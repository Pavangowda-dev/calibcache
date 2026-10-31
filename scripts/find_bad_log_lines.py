# Shows damaged lines in results/usage_log.jsonl (if there are any).
import json
from pathlib import Path

path = Path("results/usage_log.jsonl")
total = 0
bad = 0

with open(path, encoding="utf-8", errors="replace") as f:
    for number, line in enumerate(f, start=1):
        if not line.strip():
            continue
        total += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            if bad <= 5:
                print(f"line {number}: {line[:80]!r}")

print(f"{total} lines in the log, {bad} damaged")