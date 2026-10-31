# Applies my audit verdicts for the hard questions and writes the clean set.
# Questions marked wrong or unclear are dropped. The rest are kept.
import re
import sys

from calibcache.dataset import load_queries, save_queries
from calibcache.simulate import wilson_interval

sys.stdout.reconfigure(encoding="utf-8")

HARD_FILE = "data/hard_queries.jsonl"
VERDICTS_FILE = "data/hard_verdicts.csv"
OUTPUT_FILE = "data/hard_queries_clean.jsonl"
ALLOWED = ("ok", "wrong", "unclear")
EXPECTED_AUDITED = 32


def read_verdicts():
    """Reads lines like 'h012,ok'. Lines that do not start with a query id
    (for example the header line) are ignored."""
    verdicts = {}
    ignored = 0
    with open(VERDICTS_FILE, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^"?(h\d+)"?(.*)$', line, flags=re.IGNORECASE)
            if not match:
                ignored += 1
                continue
            query_id = match.group(1).lower()
            rest = match.group(2).strip(' ,;:\t"')
            verdict = re.split(r"[\s,;:]+", rest)[0].lower() if rest else ""
            if query_id in verdicts:
                raise SystemExit(f"{query_id} appears twice in {VERDICTS_FILE}")
            verdicts[query_id] = verdict
    return verdicts, ignored


def main():
    rows = load_queries(HARD_FILE)
    verdicts, ignored = read_verdicts()
    print(f"Read {len(verdicts)} verdict lines from {VERDICTS_FILE} "
          f"({ignored} other line(s) ignored, the header is one of them)")
    if len(verdicts) != EXPECTED_AUDITED:
        print(f"WARNING: expected {EXPECTED_AUDITED} verdicts, found {len(verdicts)}.")

    # Check the verdicts before changing anything
    blank = [qid for qid, v in verdicts.items() if v == ""]
    if blank:
        raise SystemExit(f"{len(blank)} verdicts are still empty (first one: {blank[0]}). "
                         "Write ok, wrong or unclear after the comma on every line.")
    bad = [(qid, v) for qid, v in verdicts.items() if v not in ALLOWED]
    if bad:
        raise SystemExit(f"Unknown verdict '{bad[0][1]}' for {bad[0][0]}. "
                         "Use ok, wrong or unclear.")
    known = {row["query_id"] for row in rows}
    unknown = [qid for qid in verdicts if qid not in known]
    if unknown:
        raise SystemExit(f"Unknown query id in the verdicts file: {unknown[0]}")

    # Apply
    clean = []
    dropped = []
    for row in rows:
        verdict = verdicts.get(row["query_id"])
        if verdict is None:
            row["audit_status"] = "unaudited"
            clean.append(row)
        elif verdict == "ok":
            row["audit_status"] = "audited_ok"
            clean.append(row)
        else:
            dropped.append(row)
    save_queries(clean, OUTPUT_FILE)

    # Report
    audited = len(verdicts)
    n_ok = sum(1 for v in verdicts.values() if v == "ok")
    n_wrong = sum(1 for v in verdicts.values() if v == "wrong")
    n_unclear = sum(1 for v in verdicts.values() if v == "unclear")
    print(f"\nAudited {audited} questions: ok {n_ok}, wrong {n_wrong}, unclear {n_unclear}")

    by_type = {}
    for row in rows:
        verdict = verdicts.get(row["query_id"])
        if verdict is None:
            continue
        t = by_type.setdefault(row["type"], {"ok": 0, "wrong": 0, "unclear": 0})
        t[verdict] += 1
    print("\nBy type:")
    for type_name, t in by_type.items():
        print(f"  {type_name:<12} ok {t['ok']}  wrong {t['wrong']}  unclear {t['unclear']}")

    problems = n_wrong + n_unclear
    low, high = wilson_interval(problems, audited)
    print(f"\nShare of audited questions that were wrong or unclear: "
          f"{100 * problems / audited:.1f}% (95% interval {100 * low:.0f}-{100 * high:.0f}%)")
    print("The unaudited questions may have problems at a similar rate.")
    if problems / audited > 0.15:
        print("WARNING: more than 15% had problems. Tell me before we continue.")

    print(f"\nKept {len(clean)} questions, dropped {len(dropped)}  -> {OUTPUT_FILE}")
    counts = {}
    for row in clean:
        key = f"{row['type']}/{row['split']}"
        counts[key] = counts.get(key, 0) + 1
    print("Kept by type and split:", counts)


main()