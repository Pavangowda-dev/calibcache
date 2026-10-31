# Collects answers from the cheap and the strong model for:
#   - the hard questions (calculation, multi_issue, timeline), and
#   - an equal number of easy questions (ordinary single-topic queries).
# Saved in data/answers.jsonl. Safe to restart: finished answers are skipped.
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import CHEAP_MODEL, STRONG_MODEL
from calibcache.dataset import load_queries
from calibcache.openai_calls import call_chat
from calibcache.policy import load_policy, support_system_prompt
from calibcache.pricing import chat_cost
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

SEED = 11
WORKERS = 6
MAX_TOKENS = 3000                  # room for models that think before answering
EXCLUDED_TYPES = {"not_covered"}   # set aside until its reference answers are fixed
MODELS = [CHEAP_MODEL, STRONG_MODEL]
OUTPUT_FILE = Path("data/answers.jsonl")


def build_items(policy):
    """The questions to answer: all good hard ones + an equal number of easy ones."""
    policy_answers = {i["id"]: i["answer"] for i in policy["intents"]}
    hard = [r for r in load_queries("data/hard_queries_clean.jsonl")
            if r["type"] not in EXCLUDED_TYPES]

    items = []
    for r in hard:
        items.append({"query_id": r["query_id"], "kind": "hard", "type": r["type"],
                      "split": r["split"], "text": r["text"],
                      "reference_answer": r["reference_answer"]})

    easy_pool = load_queries("data/queries_clean.jsonl")
    rng = random.Random(SEED)
    for split in ("calibration", "test"):
        n_hard = sum(1 for r in hard if r["split"] == split)
        pool = [r for r in easy_pool if r["split"] == split]
        for r in rng.sample(pool, n_hard):
            items.append({"query_id": r["query_id"], "kind": "easy", "type": "easy",
                          "split": r["split"], "text": r["text"],
                          "reference_answer": policy_answers[r["intent_id"]]})
    return items


def read_rows():
    """All answers saved so far (damaged lines are skipped)."""
    rows = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.replace("\x00", "").strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def ask_one(client, log, system_message, item, model):
    """One answer from one model. Tries up to 3 times."""
    messages = [system_message, {"role": "user", "content": item["text"]}]
    last_error = None
    for attempt in range(3):
        try:
            result = call_chat(client, model, messages, log, max_completion_tokens=MAX_TOKENS)
            return {
                "query_id": item["query_id"], "kind": item["kind"], "type": item["type"],
                "split": item["split"], "text": item["text"],
                "reference_answer": item["reference_answer"],
                "model_requested": model, "model_used": result["model"],
                "answer": result["answer"], "finish_reason": result["finish_reason"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "thinking_tokens": result["thinking_tokens"],
                "cost_usd": result["cost_usd"], "seconds": result["seconds"],
            }
        except Exception as error:
            last_error = error
            time.sleep(2)
    raise last_error


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

    client = OpenAI(api_key=api_key)
    log = UsageLog()
    spent_before = log.total_cost()

    policy = load_policy()
    system_message = {"role": "system", "content": support_system_prompt(policy)}
    items = build_items(policy)

    done = {(r["query_id"], r["model_requested"]) for r in read_rows()}
    todo = [(item, model) for item in items for model in MODELS
            if (item["query_id"], model) not in done]

    n_hard = sum(1 for i in items if i["kind"] == "hard")
    print(f"Questions: {len(items)} ({n_hard} hard, {len(items) - n_hard} easy), "
          f"models: {', '.join(MODELS)}")
    print(f"Answers already saved: {len(done)}, to collect now: {len(todo)}")

    if todo:
        per_call = {CHEAP_MODEL: chat_cost(CHEAP_MODEL, 2100, 150),
                    STRONG_MODEL: chat_cost(STRONG_MODEL, 2100, 300)}
        estimate = sum(per_call[model] for _, model in todo)
        print(f"Estimated cost: about ${estimate:.2f} (more if the strong model thinks a lot)")
        input("Press Enter to start, or Ctrl+C to stop... ")

        failed = 0
        finished = 0
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(ask_one, client, log, system_message, item, model)
                       for item, model in todo]
            for future in as_completed(futures):
                try:
                    row = future.result()
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                except Exception as error:
                    failed += 1
                    if failed <= 3:
                        print(f"  an answer failed: {error}")
                finished += 1
                if finished % 50 == 0:
                    print(f"  finished {finished}/{len(todo)}")
        if failed:
            print(f"{failed} answers failed. Run the script again to finish them.")

    # ---------- summary ----------
    wanted = {i["query_id"] for i in items}
    rows = [r for r in read_rows() if r["query_id"] in wanted]
    groups = {}
    for r in rows:
        g = groups.setdefault((r["model_requested"], r["kind"]),
                              {"n": 0, "cost": 0.0, "seconds": 0.0, "out": 0, "think": 0,
                               "cut": 0})
        g["n"] += 1
        g["cost"] += r["cost_usd"]
        g["seconds"] += r["seconds"]
        g["out"] += r["output_tokens"]
        g["think"] += r["thinking_tokens"]
        if r["finish_reason"] != "stop" or not r["answer"].strip():
            g["cut"] += 1

    print("\nSummary of saved answers:")
    print(f"{'model':<14}{'kind':<6}{'answers':>8}{'avg cost':>11}{'avg sec':>9}"
          f"{'avg out tok':>12}{'avg think':>10}{'cut/empty':>10}")
    for (model, kind), g in sorted(groups.items()):
        print(f"{model:<14}{kind:<6}{g['n']:>8}{g['cost'] / g['n']:>11.5f}"
              f"{g['seconds'] / g['n']:>9.2f}{g['out'] / g['n']:>12.0f}"
              f"{g['think'] / g['n']:>10.0f}{g['cut']:>10}")
    print(f"\nSpent in this run: ${log.total_cost() - spent_before:.4f}")


main()