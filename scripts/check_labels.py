# Label check: a second model (blind to the intended topic) picks the topic of
# every query. Disagreements go to data/review.csv for me to review by hand.
import csv
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import STRONG_MODEL
from calibcache.dataset import load_queries
from calibcache.openai_calls import call_chat
from calibcache.policy import load_policy
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

VERIFIER_MODEL = STRONG_MODEL      # not the same model that wrote the queries
BATCH_SIZE = 10                    # queries checked per call
MAX_TRIES = 3
SEED = 7
PREDICTIONS_FILE = Path("data/label_check.jsonl")
REVIEW_FILE = Path("data/review.csv")

SYSTEM_PROMPT = ("You are a careful quality checker for a customer support dataset. "
                 "You follow the output format exactly.")


def build_prompt(intents, batch):
    """Instructions plus the policy plus the numbered messages."""
    topic_lines = "\n".join(
        f"- {i['id']} | {i['question']} | {i['answer']}" for i in intents
    )
    message_lines = "\n".join(f"{n}. {row['text']}" for n, row in enumerate(batch, start=1))
    return f"""Below is the support policy of PayLoom, a UPI payments app.
Each topic has: id | typical question | correct answer.

{topic_lines}

Now there are {len(batch)} customer messages. For each message, choose the ONE topic id
whose answer fully and correctly answers the message.
Reply "ambiguous" if two or more topics could answer it, or if no topic's answer fits exactly.
Judge only by meaning. Ignore spelling mistakes and the mix of languages.

Messages:
{message_lines}

Return ONLY a JSON object like {{"labels": ["topic_id", "ambiguous", "topic_id"]}}
with exactly {len(batch)} labels, in the same order as the messages."""


def classify_batch(client, log, intents, valid_ids, batch):
    """Ask the verifier model for the topic of each message in the batch."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(intents, batch)},
    ]
    for attempt in range(1, MAX_TRIES + 1):
        try:
            result = call_chat(client, VERIFIER_MODEL, messages, log,
                               max_completion_tokens=2000,
                               response_format={"type": "json_object"})
            labels = json.loads(result["answer"])["labels"]
            if len(labels) != len(batch):
                raise ValueError("wrong number of labels")
            clean = []
            for label in labels:
                label = str(label).strip()
                if label not in valid_ids and label != "ambiguous":
                    raise ValueError(f"unknown label: {label}")
                clean.append(label)
            return clean
        except Exception as error:
            print(f"    attempt {attempt} failed: {error}")
    raise SystemExit("Batch failed 3 times. Run the script again.")


def load_predictions():
    """Read the labels saved so far (so a restart skips finished work)."""
    done = {}
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    done[item["query_id"]] = item["predicted"]
    return done


def print_summary(rows, done):
    counts = {"agree": 0, "ambiguous": 0, "disagree": 0}
    by_style = {}
    by_family = {}
    confusion = {}
    flagged = []

    for row in rows:
        predicted = done[row["query_id"]]
        if predicted == row["intent_id"]:
            status = "agree"
        elif predicted == "ambiguous":
            status = "ambiguous"
        else:
            status = "disagree"
        counts[status] += 1

        for table, key in [(by_style, row["style"]), (by_family, row["family"])]:
            if key not in table:
                table[key] = [0, 0]  # [total, agree]
            table[key][0] += 1
            if status == "agree":
                table[key][1] += 1

        if status != "agree":
            flagged.append((row, predicted))
            pair = (row["intent_id"], predicted)
            confusion[pair] = confusion.get(pair, 0) + 1

    total = len(rows)
    print(f"\nChecked {total} queries with {VERIFIER_MODEL}")
    print(f"  agree    : {counts['agree']} ({100 * counts['agree'] / total:.1f}%)")
    print(f"  ambiguous: {counts['ambiguous']} ({100 * counts['ambiguous'] / total:.1f}%)")
    print(f"  disagree : {counts['disagree']} ({100 * counts['disagree'] / total:.1f}%)")

    print("\nAgreement by style:")
    for style, (n, ok) in by_style.items():
        print(f"  {style:<13} {ok}/{n} = {100 * ok / n:.1f}%")

    print("\nAgreement by family:")
    for family, (n, ok) in by_family.items():
        print(f"  {family:<18} {ok}/{n} = {100 * ok / n:.1f}%")

    print("\nMost common mix-ups (intended topic -> checker's answer):")
    ranked = sorted(confusion.items(), key=lambda item: item[1], reverse=True)
    for (intended, predicted), n in ranked[:10]:
        print(f"  {n:>3}  {intended} -> {predicted}")

    return flagged


def write_review_file(flagged):
    """One row per flagged query. I fill the last column in Excel."""
    flagged = sorted(flagged, key=lambda item: (item[0]["intent_id"], item[1]))
    with open(REVIEW_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "style", "text", "intended_topic",
                         "checker_topic", "your_decision"])
        for row, predicted in flagged:
            writer.writerow([row["query_id"], row["style"], row["text"],
                             row["intent_id"], predicted, ""])
    print(f"\n{len(flagged)} flagged queries written to {REVIEW_FILE}")


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

    client = OpenAI(api_key=api_key)
    log = UsageLog()
    intents = load_policy()["intents"]
    valid_ids = {i["id"] for i in intents}

    rows = load_queries()
    done = load_predictions()
    todo = [row for row in rows if row["query_id"] not in done]
    random.Random(SEED).shuffle(todo)  # shuffled, so neighbours do not hint at the topic

    spent_before = log.total_cost()
    print(f"{len(rows)} queries, {len(done)} already checked, {len(todo)} to check")

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    for number, batch in enumerate(batches, start=1):
        labels = classify_batch(client, log, intents, valid_ids, batch)
        with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
            for row, label in zip(batch, labels):
                f.write(json.dumps({"query_id": row["query_id"], "predicted": label}) + "\n")
                done[row["query_id"]] = label
        if number % 5 == 0 or number == len(batches):
            print(f"  checked {min(number * BATCH_SIZE, len(todo))}/{len(todo)}")

    flagged = print_summary(rows, done)
    write_review_file(flagged)
    print(f"\nSpent in this run: ${log.total_cost() - spent_before:.4f}")


main()