# Builds the 100 answers that I will grade by hand, to test the LLM judge.
#   60 real answers from both models (hard and easy questions)
#   30 wrong-topic answers (what a false cache hit would serve)
#   10 right-topic cached-style answers
# Only calibration-split questions are used. No API calls.
import csv
import random
import sys
from pathlib import Path

from calibcache.dataset import load_queries, save_queries
from calibcache.judge import RUBRIC
from calibcache.policy import load_policy

sys.stdout.reconfigure(encoding="utf-8")

SEED = 21
N_REAL_PER_GROUP = 15   # per (model, kind); there are 4 groups -> 60 items
N_WRONG_TOPIC = 30
N_RIGHT_TOPIC = 10
SET_FILE = "data/judge_set_private.jsonl"   # includes the source of each item: do not open
SHEET_FILE = Path("data/judge_label_sheet.txt")
LABELS_FILE = Path("data/judge_labels.csv")


def main():
    rng = random.Random(SEED)
    policy = load_policy()
    intents = policy["intents"]
    by_id = {i["id"]: i for i in intents}
    siblings = {i["id"]: [j["id"] for j in intents
                          if j["family"] == i["family"] and j["id"] != i["id"]]
                for i in intents}

    items = []

    # ---- real answers from both models ----
    answers = [r for r in load_queries("data/answers.jsonl") if r["split"] == "calibration"]
    groups = {}
    for r in answers:
        groups.setdefault((r["model_requested"], r["kind"]), []).append(r)
    for (model, kind), rows in sorted(groups.items()):
        for r in rng.sample(rows, min(N_REAL_PER_GROUP, len(rows))):
            items.append({"source": f"real:{model}:{kind}", "query_id": r["query_id"],
                          "text": r["text"], "reference_answer": r["reference_answer"],
                          "candidate_answer": r["answer"].strip()})

    # ---- cache-style answers ----
    queries = [q for q in load_queries("data/queries_clean.jsonl")
               if q["split"] == "calibration"]
    chosen = rng.sample(queries, N_WRONG_TOPIC + N_RIGHT_TOPIC)
    for k, q in enumerate(chosen):
        own_answer = by_id[q["intent_id"]]["answer"]
        if k < N_WRONG_TOPIC:
            sibling_id = rng.choice(siblings[q["intent_id"]])
            candidate = by_id[sibling_id]["answer"]
            source = "cache_wrong_topic"
        else:
            candidate = own_answer
            source = "cache_right_topic"
        items.append({"source": source, "query_id": q["query_id"], "text": q["text"],
                      "reference_answer": own_answer, "candidate_answer": candidate})

    # ---- shuffle, number, write ----
    rng.shuffle(items)
    for number, item in enumerate(items, start=1):
        item["item_id"] = f"j{number:03d}"
    save_queries(items, SET_FILE)

    with open(SHEET_FILE, "w", encoding="utf-8") as f:
        f.write("LABEL SHEET: for each item, decide if the ANSWER is correct or incorrect.\n")
        f.write("Use only the QUESTION and the REFERENCE. Work on your own.\n\n")
        f.write("RULES\n-----\n" + RUBRIC + "\n")
        f.write("If you truly cannot decide, label it incorrect.\n\n")
        f.write("Write your labels in data/judge_labels.csv, one line per item, like:\n")
        f.write("   j001,correct\n   j002,incorrect\n\n")
        f.write("=" * 70 + "\n\n")
        for item in items:
            f.write(f"{item['item_id']}\n")
            f.write(f"  QUESTION : {item['text']}\n")
            f.write(f"  REFERENCE: {item['reference_answer']}\n")
            f.write(f"  ANSWER   : {' '.join(item['candidate_answer'].split())}\n\n")

    with open(LABELS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "label"])
        for item in items:
            writer.writerow([item["item_id"], ""])

    print(f"{len(items)} items written.")
    print(f"Read the items in : {SHEET_FILE}")
    print(f"Write your labels in: {LABELS_FILE}")
    print(f"Do NOT open {SET_FILE} until you have finished labelling.")


main()