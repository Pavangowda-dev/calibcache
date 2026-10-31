# Step A of the two-band cache: how good is a cheap model at deciding
# "can this saved answer be reused for the new question?"
# It checks every pair of calibration queries with similarity >= LOW_MIN,
# saves the verdicts (so we never pay twice), and reports how accurate it is.
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import EMBEDDING_MODEL, VERIFIER_MODEL
from calibcache.dataset import load_queries
from calibcache.embed_queries import embed_queries
from calibcache.policy import load_policy
from calibcache.pricing import chat_cost
from calibcache.simulate import wilson_interval
from calibcache.usage_log import UsageLog
from calibcache.verifier import verify_pair

sys.stdout.reconfigure(encoding="utf-8")

LOW_MIN = 0.60       # pairs below this similarity are never checked
WORKERS = 8          # calls running at the same time
MAX_PAIRS = 8000     # safety limit
VERDICTS_FILE = Path(f"results/verdicts_{VERIFIER_MODEL}.json")
BANDS = [0.60, 0.70, 0.80, 0.90, 1.01]


def load_verdicts():
    if VERDICTS_FILE.exists():
        with open(VERDICTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_verdicts(verdicts):
    """Save to a temporary file first, so a crash cannot damage the real file."""
    VERDICTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = VERDICTS_FILE.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(verdicts, f)
    temp.replace(VERDICTS_FILE)


def pair_key(rows, i, j):
    """i = the new query, j = the cached query."""
    return f"{rows[i]['query_id']}|{rows[j]['query_id']}"


def check_one(client, log, rows, answers, i, j):
    return verify_pair(client, log,
                       cached_question=rows[j]["text"],
                       cached_answer=answers[rows[j]["intent_id"]],
                       new_question=rows[i]["text"])


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

    client = OpenAI(api_key=api_key)
    log = UsageLog()
    spent_before = log.total_cost()

    policy = load_policy()
    answers = {i["id"]: i["answer"] for i in policy["intents"]}
    rows = [r for r in load_queries("data/queries_clean.jsonl") if r["split"] == "calibration"]
    print(f"Calibration split only: {len(rows)} queries")

    vectors = embed_queries(client, log, rows,
                            f"results/embeddings_calibration_{EMBEDDING_MODEL}.npz",
                            model=EMBEDDING_MODEL)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -1)  # a query is never compared with itself

    # Every ordered pair (new query i, cached query j) with similarity >= LOW_MIN
    ii, jj = np.where(sim >= LOW_MIN)
    pairs = list(zip(ii.tolist(), jj.tolist()))
    if len(pairs) > MAX_PAIRS:
        raise SystemExit(f"{len(pairs)} pairs is more than the safety limit {MAX_PAIRS}.")

    verdicts = load_verdicts()
    todo = [(i, j) for i, j in pairs if pair_key(rows, i, j) not in verdicts]

    per_call = chat_cost(VERIFIER_MODEL, 250, 3)
    print(f"Pairs with similarity >= {LOW_MIN}: {len(pairs)} "
          f"({len(pairs) - len(todo)} already checked, {len(todo)} to check)")
    print(f"Verifier model: {VERIFIER_MODEL}")
    print(f"Estimated cost: about ${len(todo) * per_call:.2f}, "
          f"time: about {len(todo) * 1.5 / WORKERS / 60:.0f} minutes")

    if todo:
        input("Press Enter to start, or Ctrl+C to stop... ")
        failed = 0
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(check_one, client, log, rows, answers, i, j): (i, j)
                       for i, j in todo}
            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    verdicts[pair_key(rows, i, j)] = future.result()
                except Exception as error:
                    failed += 1
                    if failed <= 3:
                        print(f"  a check failed: {error}")
                done += 1
                if done % 200 == 0:
                    save_verdicts(verdicts)
                    print(f"  checked {done}/{len(todo)}")
        save_verdicts(verdicts)
        if failed:
            print(f"{failed} checks failed. Run the script again to finish them.")

    # ---------- how accurate is the verifier? ----------
    same_yes = same_total = sib_yes = sib_total = other_yes = other_total = 0
    band_stats = {k: {"same": 0, "same_yes": 0, "diff": 0, "diff_yes": 0}
                  for k in range(len(BANDS) - 1)}
    costs = []
    seconds = []

    for i, j in pairs:
        verdict = verdicts.get(pair_key(rows, i, j))
        if verdict is None:
            continue
        costs.append(verdict["cost"])
        seconds.append(verdict["seconds"])
        yes = verdict["yes"]
        same = rows[i]["intent_id"] == rows[j]["intent_id"]
        sibling = (not same) and rows[i]["family"] == rows[j]["family"]

        if same:
            same_total += 1
            same_yes += yes
        elif sibling:
            sib_total += 1
            sib_yes += yes
        else:
            other_total += 1
            other_yes += yes

        s = float(sim[i, j])
        for k in range(len(BANDS) - 1):
            if BANDS[k] <= s < BANDS[k + 1]:
                key = "same" if same else "diff"
                band_stats[k][key] += 1
                band_stats[k][key + "_yes"] += yes

    print(f"\nVerifier decisions ({same_total + sib_total + other_total} pairs checked):")
    print(f"  same-topic pairs     : {same_total:>5}  verifier says YES on "
          f"{same_yes} ({100 * same_yes / max(same_total, 1):.1f}%)   <- how many it rescues")
    low, high = wilson_interval(sib_yes, sib_total)
    print(f"  sibling-topic pairs  : {sib_total:>5}  verifier says YES on "
          f"{sib_yes} ({100 * sib_yes / max(sib_total, 1):.1f}%, "
          f"95% interval {100 * low:.1f}-{100 * high:.1f}%)   <- wrong YES, the dangerous one")
    print(f"  other-family pairs   : {other_total:>5}  verifier says YES on "
          f"{other_yes} ({100 * other_yes / max(other_total, 1):.1f}%)")
    diff_yes = sib_yes + other_yes
    diff_total = sib_total + other_total
    precision_wrong = diff_yes / max(diff_yes + same_yes, 1)
    print(f"  Of all YES answers, {100 * precision_wrong:.1f}% were for a different topic")

    print("\nBy similarity band:")
    print(f"{'band':<11}{'same-topic':>11}{'YES on same':>13}{'different':>11}{'YES on different':>18}")
    for k in range(len(BANDS) - 1):
        b = band_stats[k]
        label = f"{BANDS[k]:.2f}-{min(BANDS[k + 1], 1.0):.2f}"
        print(f"{label:<11}{b['same']:>11}{b['same_yes']:>13}{b['diff']:>11}{b['diff_yes']:>18}")

    if costs:
        print(f"\nAverage per check: ${np.mean(costs):.6f}, {np.mean(seconds):.2f} seconds")
    print(f"Spent in this run: ${log.total_cost() - spent_before:.4f}")


main()