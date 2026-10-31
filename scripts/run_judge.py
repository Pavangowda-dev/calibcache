# Step 1: check the judge against my 100 hand labels.
# Step 2: if it agrees well enough, grade all collected answers and show how often
#         the cheap model is wrong (calibration half only).
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import CHEAP_MODEL, STRONG_MODEL
from calibcache.dataset import load_queries
from calibcache.judge import JUDGE_MODEL, build_messages, judge_answer
from calibcache.pricing import chat_cost
from calibcache.simulate import wilson_interval
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

WORKERS = 6
MIN_AGREEMENT = 0.90
MAX_FALSE_ACCEPT = 0.10
SET_FILE = "data/judge_set_private.jsonl"
LABELS_FILE = "data/judge_labels.csv"
ANSWERS_FILE = "data/answers.jsonl"
GRADES_FILE = Path("data/answer_grades.jsonl")
VALIDATION_FILE = Path("results/judge_validation.json")
MODELS = [CHEAP_MODEL, STRONG_MODEL]

# Identifies the judge's prompt and model. If it changes, old verdicts are not reused.
FINGERPRINT = hashlib.sha256(
    (JUDGE_MODEL + json.dumps(build_messages("q", "r", "c"))).encode()
).hexdigest()[:12]


def interval_text(k, n):
    """'94.0% (88-97%)' for k out of n."""
    if n == 0:
        return "-"
    low, high = wilson_interval(k, n)
    return f"{100 * k / n:.1f}% ({100 * low:.0f}-{100 * high:.0f}%)"


def read_labels():
    """Reads lines like 'j001,correct'. Other lines (the header) are ignored."""
    labels = {}
    with open(LABELS_FILE, encoding="utf-8-sig") as f:
        for line in f:
            match = re.match(r'^"?(j\d+)"?[\s,;:]*([a-zA-Z]*)', line.strip(), flags=re.IGNORECASE)
            if match:
                labels[match.group(1).lower()] = match.group(2).lower()
    return labels


def judge_tasks(client, log, tasks):
    """tasks: list of (key, question, reference, candidate). Returns {key: result}."""
    results = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(judge_answer, client, log, q, r, c): key
                   for key, q, r, c in tasks}
        done = 0
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as error:
                failed += 1
                if failed <= 3:
                    print(f"  a judgement failed: {error}")
            done += 1
            if done % 100 == 0:
                print(f"  judged {done}/{len(tasks)}")
    if failed:
        print(f"{failed} judgements failed. Run the script again to finish them.")
    return results


def read_grades():
    """Saved grades that were made with the same judge. Returns {(query_id, model): correct}."""
    grades = {}
    if GRADES_FILE.exists():
        with open(GRADES_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.replace("\x00", "").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("fingerprint") == FINGERPRINT:
                    grades[(row["query_id"], row["model_requested"])] = row["correct"]
    return grades


# ------------------------------------------------------------------ part 1
def validate_judge(client, log):
    items = load_queries(SET_FILE)
    labels = read_labels()

    ids = {item["item_id"] for item in items}
    missing = sorted(ids - set(labels))
    if missing:
        raise SystemExit(f"{len(missing)} items have no label yet (first: {missing[0]}).")
    bad = [(i, v) for i, v in labels.items() if v not in ("correct", "incorrect")]
    if bad:
        raise SystemExit(f"Unknown label '{bad[0][1]}' for {bad[0][0]}. Use correct or incorrect.")

    # Judge the 100 items (verdicts are saved, so a restart costs nothing)
    cache = {}
    if VALIDATION_FILE.exists():
        with open(VALIDATION_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("fingerprint") == FINGERPRINT:
            cache = saved["verdicts"]
    todo = [(i["item_id"], i["text"], i["reference_answer"], i["candidate_answer"])
            for i in items if i["item_id"] not in cache]
    if todo:
        print(f"Judging {len(todo)} items with {JUDGE_MODEL} "
              f"(about ${len(todo) * chat_cost(JUDGE_MODEL, 450, 80):.2f}) ...")
        cache.update(judge_tasks(client, log, todo))
        VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_FILE, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": FINGERPRINT, "verdicts": cache}, f)
    if any(i["item_id"] not in cache for i in items):
        raise SystemExit("Some items could not be judged. Run the script again.")

    # Compare the judge with my labels
    a = b = c = d = 0   # a: me correct/judge correct, b: me correct/judge incorrect,
    #                     c: me incorrect/judge correct, d: me incorrect/judge incorrect
    by_source = defaultdict(lambda: {"n": 0, "me_bad": 0, "judge_bad": 0, "agree": 0})
    disagreements = []
    for item in items:
        me = labels[item["item_id"]] == "correct"
        verdict = cache[item["item_id"]]
        judge = verdict["correct"]
        if me and judge:
            a += 1
        elif me and not judge:
            b += 1
        elif not me and judge:
            c += 1
        else:
            d += 1
        s = by_source[item["source"]]
        s["n"] += 1
        s["me_bad"] += (not me)
        s["judge_bad"] += (not judge)
        s["agree"] += (me == judge)
        if me != judge:
            disagreements.append((item, me, verdict))

    n = a + b + c + d
    agreement = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    kappa = (agreement - pe) / (1 - pe) if pe < 1 else 1.0
    false_accept = c / (c + d) if (c + d) else 0.0

    print(f"\nJUDGE VERSUS MY LABELS ({n} items)")
    print(f"  agreement : {a + d}/{n} = {interval_text(a + d, n)}")
    print(f"  Cohen's kappa: {kappa:.2f}")
    print(f"  I said correct   -> judge correct {a}, judge incorrect {b}")
    print(f"  I said incorrect -> judge correct {c}, judge incorrect {d}")
    print(f"  Judge wrongly ACCEPTS (I said incorrect): {c}/{c + d} = {interval_text(c, c + d)}")
    print(f"  Judge wrongly REJECTS (I said correct)  : {b}/{a + b} = {interval_text(b, a + b)}")

    print("\nBy kind of item:")
    print(f"  {'source':<28}{'n':>4}{'I: incorrect':>14}{'judge: incorrect':>18}{'agree':>7}")
    for source, s in sorted(by_source.items()):
        print(f"  {source:<28}{s['n']:>4}{s['me_bad']:>14}{s['judge_bad']:>18}{s['agree']:>7}")

    if disagreements:
        print(f"\nDISAGREEMENTS ({len(disagreements)}), showing up to 15:")
        for item, me, verdict in disagreements[:15]:
            print(f"\n{item['item_id']} [{item['source']}]  me: {'correct' if me else 'incorrect'}"
                  f"   judge: {'correct' if verdict['correct'] else 'incorrect'}")
            print(f"  QUESTION : {item['text']}")
            print(f"  REFERENCE: {item['reference_answer']}")
            print(f"  ANSWER   : {' '.join(item['candidate_answer'].split())}")
            print(f"  JUDGE SAID: {verdict['reason']}")

    if agreement < MIN_AGREEMENT or false_accept > MAX_FALSE_ACCEPT:
        raise SystemExit(f"\nSTOP: the judge needs agreement >= {100 * MIN_AGREEMENT:.0f}% and "
                         f"false accepts <= {100 * MAX_FALSE_ACCEPT:.0f}%. "
                         "I am not grading the answers. Send me this output.")
    print("\nThe judge passes the check.")


# ------------------------------------------------------------------ part 2
def grade_answers(client, log):
    answers = load_queries(ANSWERS_FILE)
    grades = read_grades()
    todo = [r for r in answers if (r["query_id"], r["model_requested"]) not in grades]
    print(f"\nAnswers: {len(answers)}, already graded: {len(answers) - len(todo)}, "
          f"to grade now: {len(todo)}")

    if todo:
        print(f"Estimated cost: about ${len(todo) * chat_cost(JUDGE_MODEL, 450, 80):.2f}")
        input("Press Enter to start grading, or Ctrl+C to stop... ")
        tasks = [((r["query_id"], r["model_requested"]), r["text"],
                  r["reference_answer"], r["answer"]) for r in todo]
        by_key = {(r["query_id"], r["model_requested"]): r for r in todo}
        results = judge_tasks(client, log, tasks)
        with open(GRADES_FILE, "a", encoding="utf-8") as f:
            for key, verdict in results.items():
                r = by_key[key]
                f.write(json.dumps({"query_id": r["query_id"],
                                    "model_requested": r["model_requested"],
                                    "kind": r["kind"], "type": r["type"], "split": r["split"],
                                    "correct": verdict["correct"], "reason": verdict["reason"],
                                    "fingerprint": FINGERPRINT}) + "\n")
        grades = read_grades()

    # ---- summary: calibration half only ----
    calibration = [r for r in answers if r["split"] == "calibration"]
    n_test = sum(1 for r in answers if r["split"] == "test")
    print(f"\nAccuracy is shown for the CALIBRATION half only "
          f"({n_test} test-half answers are graded but not shown).")

    print("\nTABLE 1: how often is each model correct? (calibration half)")
    print(f"{'question type':<14}{'model':<14}{'n':>4}{'correct':>9}   accuracy (95% interval)")
    for type_name in ["easy", "calculation", "multi_issue", "timeline"]:
        for model in MODELS:
            rows = [r for r in calibration
                    if r["type"] == type_name and r["model_requested"] == model
                    and (r["query_id"], model) in grades]
            k = sum(1 for r in rows if grades[(r["query_id"], model)])
            print(f"{type_name:<14}{model:<14}{len(rows):>4}{k:>9}   {interval_text(k, len(rows))}")

    # ---- pair the two models' answers to the same question ----
    pairs = {}
    for r in calibration:
        pairs.setdefault(r["query_id"], {})[r["model_requested"]] = r
    full = [p for p in pairs.values()
            if all(m in p and (p[m]["query_id"], m) in grades for m in MODELS)]

    print("\nTABLE 2: does the strong model fix the cheap model's mistakes? (calibration half)")
    print(f"{'kind':<6}{'n':>4}{'both ok':>9}{'cheap ok only':>15}{'strong ok only':>16}{'neither':>9}")
    for kind in ["easy", "hard"]:
        group = [p for p in full if p[CHEAP_MODEL]["kind"] == kind]
        counts = {"both": 0, "cheap": 0, "strong": 0, "neither": 0}
        for p in group:
            cheap_ok = grades[(p[CHEAP_MODEL]["query_id"], CHEAP_MODEL)]
            strong_ok = grades[(p[STRONG_MODEL]["query_id"], STRONG_MODEL)]
            if cheap_ok and strong_ok:
                counts["both"] += 1
            elif cheap_ok:
                counts["cheap"] += 1
            elif strong_ok:
                counts["strong"] += 1
            else:
                counts["neither"] += 1
        print(f"{kind:<6}{len(group):>4}{counts['both']:>9}{counts['cheap']:>15}"
              f"{counts['strong']:>16}{counts['neither']:>9}")

    print("\nTABLE 3: the most a router could gain (calibration half, average per question)")
    print("A perfect router uses the cheap model whenever its answer is correct, else the strong one.")
    print(f"{'kind':<6}{'n':>4} | {'cheap: acc  $':<20}| {'strong: acc  $':<21}| perfect router: acc  $ ")
    for kind in ["easy", "hard", "all"]:
        group = [p for p in full if kind == "all" or p[CHEAP_MODEL]["kind"] == kind]
        if not group:
            continue
        n = len(group)
        cheap_ok = [grades[(p[CHEAP_MODEL]["query_id"], CHEAP_MODEL)] for p in group]
        strong_ok = [grades[(p[STRONG_MODEL]["query_id"], STRONG_MODEL)] for p in group]
        cheap_cost = [p[CHEAP_MODEL]["cost_usd"] for p in group]
        strong_cost = [p[STRONG_MODEL]["cost_usd"] for p in group]
        perfect_ok = [c or s for c, s in zip(cheap_ok, strong_ok)]
        perfect_cost = [cc if c else sc for c, cc, sc in zip(cheap_ok, cheap_cost, strong_cost)]
        print(f"{kind:<6}{n:>4} | {100 * sum(cheap_ok) / n:>5.1f}% ${sum(cheap_cost) / n:.5f}  "
              f"| {100 * sum(strong_ok) / n:>5.1f}% ${sum(strong_cost) / n:.5f}  "
              f"| {100 * sum(perfect_ok) / n:>5.1f}% ${sum(perfect_cost) / n:.5f}")


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")
    client = OpenAI(api_key=api_key)
    log = UsageLog()
    spent_before = log.total_cost()

    validate_judge(client, log)
    grade_answers(client, log)
    print(f"\nSpent in this run: ${log.total_cost() - spent_before:.4f}")


main()