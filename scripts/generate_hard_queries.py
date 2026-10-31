# Generates "hard" support questions: they need calculation, combining topics,
# time reasoning, or an honest "not covered" answer. The strong model writes each
# question with a reference answer. Used later to test the router.
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import STRONG_MODEL
from calibcache.openai_calls import call_chat
from calibcache.policy import load_policy
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

GENERATOR_MODEL = STRONG_MODEL
PER_CALL = 20            # questions asked for in one call
CALLS_PER_TYPE = 2
MIN_PER_CALL = 17        # accept a reply with at least this many questions
GENERATED_DIR = Path("data/hard_generated")
OUTPUT_FILE = Path("data/hard_queries.jsonl")
AUDIT_FILE = Path("data/hard_audit.txt")
VERDICTS_FILE = Path("data/hard_verdicts.csv")
AUDIT_PER_TYPE = 8
SEED = 42
MAX_TRIES = 3

TYPES = {
    "calculation": (
        "The customer describes a situation with specific amounts or counts. The answer "
        "needs arithmetic or comparing numbers with limits or fees from the policy (for "
        "example: the total sent today against the daily limit, how many transactions are "
        "left, a wallet balance after adding money, the amount of a fee)."),
    "multi_issue": (
        "The customer mixes two or three different problems in one message, so the answer "
        "must cover each of them using different topics of the policy."),
    "timeline": (
        "The customer says how much time has passed (hours or working days). The answer "
        "depends on the time limits in the policy (for example: keep waiting, or raise a "
        "dispute now)."),
    "not_covered": (
        "The customer asks something about the app that the policy does NOT cover (for "
        "example: use abroad, business accounts, taxes, support hours, other banks' "
        "charges). The correct answer says the policy does not cover it and suggests "
        "Help > Contact support. Never ask about something the policy does cover."),
}

SYSTEM_PROMPT = ("You write careful test questions and correct reference answers for a "
                 "customer support assistant. You follow the output format exactly.")


def policy_lines(policy):
    """The policy as text, with the topic ids."""
    return "\n".join(f"- [{i['id']}] {i['question']} {i['answer']}" for i in policy["intents"])


def build_prompt(policy, type_name, batch_number):
    if batch_number == 1:
        focus = "Cover a wide range of situations."
    else:
        focus = ("This is the second batch: cover different situations and topics than the "
                 "most obvious ones.")
    return f"""You write test questions for the customer support assistant of PayLoom, a UPI payments app in India.
The assistant only knows this policy:

{policy_lines(policy)}

QUESTION TYPE: {type_name}
{TYPES[type_name]}

Write exactly {PER_CALL} different customer questions of this type. {focus}
For each question also write the reference answer: the correct answer based ONLY on the policy.

Rules:
- The reference answer has 1 to 4 short sentences. Show the calculation or the reasoning briefly.
- Do not use any fact that is not in the policy. Do not guess.
- Customers write in different styles: some formal, most casual, some with spelling mistakes, a few in Hinglish (Hindi and English in Roman letters).
- Each question must have exactly one correct answer. Avoid vague questions.
- Use only plain ASCII characters. Write Rs instead of the rupee symbol. No markdown.
- "topics_used" lists the ids of the policy topics needed for the answer (an empty list for not_covered).

Return ONLY a JSON object like:
{{"items": [{{"question": "...", "topics_used": ["topic_id"], "reference_answer": "..."}}]}}"""


def parse_reply(text, valid_ids, type_name):
    """Turn the model's reply into a list of items. Raises an error if broken."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
    items = json.loads(text)["items"]
    if not isinstance(items, list) or len(items) < MIN_PER_CALL:
        raise ValueError("too few items")
    clean = []
    for item in items:
        question = str(item["question"]).strip()
        answer = str(item["reference_answer"]).strip()
        topics = item.get("topics_used", [])
        if not question or not answer or not isinstance(topics, list):
            raise ValueError("a broken item")
        topics = [str(t) for t in topics]
        for topic in topics:
            if topic not in valid_ids:
                raise ValueError(f"unknown topic id: {topic}")
        if type_name != "not_covered" and not topics:
            raise ValueError("an item has no topics")
        clean.append({"question": question, "topics_used": topics, "reference_answer": answer})
    return clean


def generate_one(client, log, policy, valid_ids, type_name, batch_number):
    """One call. Saves the result in data/hard_generated/."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(policy, type_name, batch_number)},
    ]
    for attempt in range(1, MAX_TRIES + 1):
        try:
            result = call_chat(client, GENERATOR_MODEL, messages, log,
                               max_completion_tokens=16000,
                               response_format={"type": "json_object"})
            items = parse_reply(result["answer"], valid_ids, type_name)
            data = {"type": type_name, "batch": batch_number,
                    "model": result["model"], "items": items}
            path = GENERATED_DIR / f"{type_name}_{batch_number}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return
        except Exception as error:
            print(f"    attempt {attempt} failed: {error}")
    raise SystemExit(f"Could not generate {type_name} batch {batch_number}. Run the script again.")


def clean_text(text):
    """Plain ASCII only. Returns None if other odd characters remain."""
    replacements = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u20b9": "Rs "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return None
    return text


def normalize(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def assemble():
    """Read data/hard_generated/, clean, remove duplicates, split, write hard_queries.jsonl."""
    items = []
    for type_name in TYPES:
        for batch in range(1, CALLS_PER_TYPE + 1):
            with open(GENERATED_DIR / f"{type_name}_{batch}.json", encoding="utf-8") as f:
                data = json.load(f)
            for item in data["items"]:
                items.append((type_name, item))

    seen = set()
    kept = []
    removed = {"bad characters": 0, "duplicate": 0}
    for type_name, item in items:
        question = clean_text(item["question"])
        answer = clean_text(item["reference_answer"])
        if question is None or answer is None:
            removed["bad characters"] += 1
            continue
        norm = normalize(question)
        if norm in seen:
            removed["duplicate"] += 1
            continue
        seen.add(norm)
        kept.append({"type": type_name, "text": question, "reference_answer": answer,
                     "topics_used": item["topics_used"]})

    # Split each type 50/50 into calibration and test
    rng = random.Random(SEED)
    by_type = {}
    for row in kept:
        by_type.setdefault(row["type"], []).append(row)
    for rows in by_type.values():
        rng.shuffle(rows)
        half = len(rows) // 2
        for position, row in enumerate(rows):
            row["split"] = "calibration" if position < half else "test"

    final = [row for rows in by_type.values() for row in rows]
    for number, row in enumerate(final, start=1):
        row["query_id"] = f"h{number:03d}"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")
    return final, removed


def write_audit(final):
    """A reading file with some questions to check, and an empty verdicts file to fill."""
    rng = random.Random(SEED + 1)
    chosen = []
    for type_name in TYPES:
        rows = [r for r in final if r["type"] == type_name]
        chosen.extend(rng.sample(rows, min(AUDIT_PER_TYPE, len(rows))))

    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        f.write("AUDIT: for each item, check the reference answer against the policy.\n")
        f.write("Verdict ok      = the reference answer is correct and follows from the policy,\n")
        f.write("                  and the question has exactly one right answer.\n")
        f.write("Verdict wrong   = the reference answer is wrong or uses facts not in the policy.\n")
        f.write("Verdict unclear = the question is vague or has more than one right answer.\n\n")
        for row in chosen:
            f.write(f"{row['query_id']}  [{row['type']}]  topics: {', '.join(row['topics_used']) or '-'}\n")
            f.write(f"  QUESTION : {row['text']}\n")
            f.write(f"  REFERENCE: {row['reference_answer']}\n\n")

    with open(VERDICTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "verdict"])
        for row in chosen:
            writer.writerow([row["query_id"], ""])
    return len(chosen)


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

    client = OpenAI(api_key=api_key)
    log = UsageLog()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    policy = load_policy()
    valid_ids = {i["id"] for i in policy["intents"]}

    todo = [(t, b) for t in TYPES for b in range(1, CALLS_PER_TYPE + 1)
            if not (GENERATED_DIR / f"{t}_{b}.json").exists()]
    spent_before = log.total_cost()
    if todo:
        print(f"{len(todo)} calls with {GENERATOR_MODEL} to make "
              f"({PER_CALL} questions each). Estimated cost: about $0.15 to $0.30 per call, "
              f"time: a few minutes in total.")
        input("Press Enter to start, or Ctrl+C to stop... ")
    for type_name, batch in todo:
        print(f"generating {type_name}, batch {batch} ...")
        generate_one(client, log, policy, valid_ids, type_name, batch)

    final, removed = assemble()
    audit_count = write_audit(final)

    counts = {}
    splits = {}
    for row in final:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    print(f"\nHard questions written to {OUTPUT_FILE}: {len(final)}")
    print("Removed:", ", ".join(f"{k}: {v}" for k, v in removed.items()))
    print("By type:", counts)
    print("By split:", splits)

    rng = random.Random(1)
    print("\nTwo random examples per type:")
    for type_name in TYPES:
        rows = [r for r in final if r["type"] == type_name]
        for row in rng.sample(rows, min(2, len(rows))):
            print(f"\n[{type_name}] {row['text']}")
            print(f"   reference: {row['reference_answer']}")

    print(f"\nAudit: {audit_count} questions to check are in {AUDIT_FILE}")
    print(f"Write your verdicts in {VERDICTS_FILE}")
    print(f"Spent in this run: ${log.total_cost() - spent_before:.4f}")


main()