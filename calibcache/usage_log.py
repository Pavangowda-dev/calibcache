# Saves every OpenAI call (tokens, cost, time), one line per call.
# We save only numbers, never the question text, to protect users' privacy.
# UsageLog(None) keeps the log in memory only (nothing is written to disk).
import json
import threading
import time
from pathlib import Path


class UsageLog:
    def __init__(self, path="results/usage_log.jsonl"):
        self.lock = threading.Lock()  # so several calls can write at the same time
        self.bad_lines = 0            # damaged lines found the last time we read the file
        self.memory = []              # used only when path is None
        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, kind, model, input_tokens, output_tokens, cost_usd, seconds):
        """Add one call to the log."""
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,  # "chat" or "embedding"
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "seconds": seconds,
        }
        with self.lock:
            if self.path is None:
                self.memory.append(row)
            else:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        return row

    def read_all(self):
        """Read all logged calls back as a list of dictionaries.
        Damaged lines are skipped (and counted in self.bad_lines)."""
        self.bad_lines = 0
        if self.path is None:
            return list(self.memory)
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.replace("\x00", "").strip()  # remove stray null characters
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "cost_usd" not in row:
                        raise ValueError("no cost in this line")
                except (json.JSONDecodeError, ValueError):
                    self.bad_lines += 1
                    continue
                rows.append(row)
        return rows

    def total_cost(self):
        """Total dollars spent so far."""
        return sum(row["cost_usd"] for row in self.read_all())

    def print_summary(self):
        """Print calls, tokens and cost for each model."""
        rows = self.read_all()
        if not rows:
            print("No calls logged yet.")
            return

        groups = {}
        for row in rows:
            key = (row["kind"], row["model"])
            if key not in groups:
                groups[key] = {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
            groups[key]["calls"] += 1
            groups[key]["input"] += row["input_tokens"]
            groups[key]["output"] += row["output_tokens"]
            groups[key]["cost"] += row["cost_usd"]

        print("Usage so far:")
        for (kind, model), g in groups.items():
            print(f"  {kind:<10} {model:<30} calls={g['calls']:<4} "
                  f"in={g['input']:<7} out={g['output']:<7} cost=${g['cost']:.6f}")
        print(f"  TOTAL SPENT: ${self.total_cost():.6f}")
        if self.bad_lines:
            print(f"  Note: {self.bad_lines} damaged line(s) in the log were ignored.")