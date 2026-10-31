# Small helpers to read and write the query files (one JSON object per line).
import json


def load_queries(path="data/queries.jsonl"):
    """Read a .jsonl file and return a list of dictionaries."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_queries(rows, path):
    """Write a list of dictionaries to a .jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")