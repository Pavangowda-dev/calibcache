# Applies my review decisions and writes the clean test set.
# data/review.csv lists the flagged queries.
# data/decisions.csv has one line per flagged query:  query_id,decision
# Queries the checker agreed with are kept as they are.
import csv
import sys

from calibcache.dataset import load_queries, save_queries
from calibcache.policy import load_policy

sys.stdout.reconfigure(encoding="utf-8")

REVIEW_FILE = "data/review.csv"
DECISIONS_FILE = "data/decisions.csv"
OUTPUT_FILE = "data/queries_clean.jsonl"


def read_flagged_ids():
    """The query ids that were flagged by the label check."""
    with open(REVIEW_FILE, encoding="utf-8-sig", newline="") as f:
        return [row["query_id"] for row in csv.DictReader(f)]


def read_decisions():
    """Read decisions.csv. Returns {query_id: decision}."""
    decisions = {}
    with open(DECISIONS_FILE, encoding="utf-8-sig", newline="") as f:
        for line_number, row in enumerate(csv.reader(f), start=1):
            if not row or not "".join(row).strip():
                continue  # skip empty lines
            if len(row) < 2:
                raise SystemExit(f"decisions.csv line {line_number}: "
                                 "expected 'query_id,decision'")
            query_id = row[0].strip()
            decision = row[1].strip().lower()
            if query_id.lower() == "query_id":
                continue  # header line
            if query_id in decisions:
                raise SystemExit(f"decisions.csv: {query_id} appears twice")
            decisions[query_id] = decision
    return decisions


def count_by(rows, field):
    counts = {}
    for row in rows:
        counts[row[field]] = counts.get(row[field], 0) + 1
    return counts


def main():
    policy = load_policy()
    family_of = {i["id"]: i["family"] for i in policy["intents"]}
    rows = load_queries()
    flagged = set(read_flagged_ids())
    decisions = read_decisions()

    # Check the decisions before changing anything
    missing = sorted(flagged - set(decisions))
    if missing:
        raise SystemExit(f"{len(missing)} flagged queries have no decision yet "
                         f"(first ones: {', '.join(missing[:5])}).")

    extra = sorted(set(decisions) - flagged)
    if extra:
        raise SystemExit(f"decisions.csv has ids that were not flagged: {', '.join(extra[:5])}")

    bad = [(qid, d) for qid, d in decisions.items()
           if d not in ("keep", "drop") and d not in family_of]
    if bad:
        raise SystemExit(f"Unknown decision '{bad[0][1]}' for {bad[0][0]}. "
                         "Use keep, drop, or a topic id from policy_intents.json.")

    # Apply the decisions
    clean = []
    dropped = []
    kept_by_me = 0
    relabeled = 0
    for row in rows:
        decision = decisions.get(row["query_id"])
        if decision is None:
            row["label_source"] = "checker_agreed"
            clean.append(row)
        elif decision == "drop":
            dropped.append(row)
        elif decision == "keep":
            row["label_source"] = "human_kept"
            kept_by_me += 1
            clean.append(row)
        else:
            # the decision is a topic id: relabel the query
            row["intent_id"] = decision
            row["family"] = family_of[decision]
            row["label_source"] = "human_relabeled"
            relabeled += 1
            clean.append(row)

    save_queries(clean, OUTPUT_FILE)

    # Summary
    print(f"Queries before: {len(rows)}")
    print(f"Reviewed: {len(decisions)} "
          f"(kept {kept_by_me}, relabeled {relabeled}, dropped {len(dropped)})")
    print(f"Queries after : {len(clean)}  -> written to {OUTPUT_FILE}\n")

    before = count_by(rows, "style")
    after = count_by(clean, "style")
    print("By style (before -> after):")
    for style in before:
        print(f"  {style:<13} {before[style]} -> {after.get(style, 0)}")

    print("\nBy split:", count_by(clean, "split"))
    print("By label source:", count_by(clean, "label_source"))

    per_topic = count_by(clean, "intent_id")
    print(f"Queries per topic: min {min(per_topic.values())}, max {max(per_topic.values())}")


main()