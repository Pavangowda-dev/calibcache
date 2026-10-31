# Fixes two policy answers that start with "No." / "Yes," (the right polarity depends on
# how the customer words the question), and refreshes everything that copied them.
import json
from pathlib import Path

from calibcache.dataset import load_queries, save_queries

POLICY_FILE = "data/policy_intents.json"
ANSWERS_FILE = "data/answers.jsonl"
JUDGE_SET_FILE = "data/judge_set_private.jsonl"
GRADES_FILE = Path("data/answer_grades.jsonl")
VALIDATION_FILE = Path("results/judge_validation.json")

FIXES = {
    "p2p_fee": (
        "No. Sending money to friends and family from your bank account is free.",
        "Sending money to friends and family from your bank account is free. "
        "PayLoom charges no fee for it.",
    ),
    "wallet_credit_card_fee": (
        "Yes, adding money with a credit card has a 2 percent fee, while adding money "
        "from a bank account is free.",
        "Adding money to the wallet with a credit card has a 2 percent fee, while adding "
        "money from a bank account is free.",
    ),
}


def main():
    replacements = {old: new for old, new in FIXES.values()}

    # 1) the policy file
    with open(POLICY_FILE, encoding="utf-8") as f:
        policy = json.load(f)
    changed = 0
    for intent in policy["intents"]:
        if intent["id"] in FIXES:
            old, new = FIXES[intent["id"]]
            if intent["answer"] == old:
                intent["answer"] = new
                changed += 1
            elif intent["answer"] != new:
                raise SystemExit(f"The answer of {intent['id']} is not what I expected. "
                                 "I did not change anything. Send me this message.")
    with open(POLICY_FILE, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    print(f"Policy answers updated: {changed}")

    # 2) reference answers copied into the collected answers
    answers = load_queries(ANSWERS_FILE)
    affected_pairs = set()
    for row in answers:
        if row["reference_answer"] in replacements:
            row["reference_answer"] = replacements[row["reference_answer"]]
            affected_pairs.add((row["query_id"], row["model_requested"]))
    save_queries(answers, ANSWERS_FILE)
    print(f"Answer rows with a fixed reference: {len(affected_pairs)}")

    # 3) reference answers in the judge set (the answers I labelled stay untouched)
    items = load_queries(JUDGE_SET_FILE)
    affected_items = set()
    for item in items:
        if item["reference_answer"] in replacements:
            item["reference_answer"] = replacements[item["reference_answer"]]
            affected_items.add(item["item_id"])
    save_queries(items, JUDGE_SET_FILE)
    print(f"Judge-set items with a fixed reference: {len(affected_items)}")

    # 4) delete the stale grades so the judge grades them again
    removed = 0
    if GRADES_FILE.exists():
        grades = load_queries(str(GRADES_FILE))
        kept = [g for g in grades if (g["query_id"], g["model_requested"]) not in affected_pairs]
        removed = len(grades) - len(kept)
        save_queries(kept, str(GRADES_FILE))
    print(f"Stale answer grades removed: {removed}")

    removed_validation = 0
    if VALIDATION_FILE.exists():
        with open(VALIDATION_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        for item_id in affected_items:
            if saved["verdicts"].pop(item_id, None) is not None:
                removed_validation += 1
        with open(VALIDATION_FILE, "w", encoding="utf-8") as f:
            json.dump(saved, f)
    print(f"Stale judge-check verdicts removed: {removed_validation}")


main()