# Prints the audited questions that were marked wrong or unclear, so we can see why.
import re
import sys

from calibcache.dataset import load_queries

sys.stdout.reconfigure(encoding="utf-8")

rows = {r["query_id"]: r for r in load_queries("data/hard_queries.jsonl")}

with open("data/hard_verdicts.csv", encoding="utf-8-sig") as f:
    for line in f:
        match = re.match(r'^"?(h\d+)"?[\s,;:]*([a-zA-Z]*)', line.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        query_id, verdict = match.group(1).lower(), match.group(2).lower()
        if verdict in ("wrong", "unclear"):
            row = rows[query_id]
            print(f"{query_id}  [{row['type']}]  verdict: {verdict}")
            print(f"  topics   : {', '.join(row['topics_used']) or '-'}")
            print(f"  QUESTION : {row['text']}")
            print(f"  REFERENCE: {row['reference_answer']}\n")