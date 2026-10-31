# Generates the test queries: about 25 per topic, in 5 styles.
# Step 1: ask the cheap model for queries, one topic at a time (saved in data/generated/)
# Step 2: clean them, remove duplicates, split into calibration/test,
#         and write everything to data/queries.jsonl
import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import CHEAP_MODEL
from calibcache.openai_calls import call_chat
from calibcache.policy import load_policy
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")  # avoids crashes when printing odd characters

GENERATOR_MODEL = CHEAP_MODEL
STYLE_COUNTS = {"formal": 5, "casual": 6, "typos": 5, "hinglish": 5, "minimal_edit": 4}
GENERATED_DIR = Path("data/generated")
OUTPUT_FILE = Path("data/queries.jsonl")
SEED = 42        # fixed seed, so the split is the same every time
MAX_TRIES = 3    # tries per topic if the model's reply is broken

SYSTEM_PROMPT = ("You write realistic customer support messages for a UPI payments app "
                 "in India. You follow the output format exactly.")


def build_prompt(intent, siblings):
    """The instructions for one topic."""
    sibling_lines = "\n".join(f"- {s['question']}" for s in siblings)
    return f"""App: PayLoom (a UPI payments app in India).

TARGET TOPIC
Typical question: {intent['question']}
Correct answer for this topic: {intent['answer']}

OTHER TOPICS FROM THE SAME FAMILY (messages about these are NOT allowed):
{sibling_lines}

Write messages that real customers would send to support about the TARGET TOPIC only.
Every message must be a question or request that the correct answer above answers fully and correctly.
Do not mention specific amounts, dates or limits unless the target topic is about them.

Write exactly these groups:
- "formal": {STYLE_COUNTS['formal']} polite messages in complete sentences
- "casual": {STYLE_COUNTS['casual']} short everyday messages, as typed on a phone, lowercase is fine
- "typos": {STYLE_COUNTS['typos']} messages with realistic spelling mistakes and missing punctuation
- "hinglish": {STYLE_COUNTS['hinglish']} messages mixing Hindi and English in Roman letters (for example: paise kat gaye lekin payment fail ho gaya)
- "minimal_edit": {STYLE_COUNTS['minimal_edit']} messages made by taking one of the OTHER TOPICS questions above and changing as few words as possible, so that the message now belongs to the TARGET TOPIC

Rules:
- All messages must be different from each other and from the typical question.
- Use only plain ASCII characters. Write Rs or Rs. instead of the rupee symbol.
- Use greetings like hello or hi in at most 2 messages.
- Return ONLY a JSON object with the five keys above, each holding a list of strings."""


def parse_reply(text):
    """Turn the model's reply into {style: [messages]}. Raises an error if broken."""
    text = text.strip()
    if text.startswith("```"):
        # remove ``` fences if the model added them
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
    data = json.loads(text)
    result = {}
    for style, count in STYLE_COUNTS.items():
        items = data.get(style)
        if not isinstance(items, list) or len(items) < count:
            raise ValueError(f"style '{style}' has too few messages")
        result[style] = [str(item).strip() for item in items[:count]]
    return result


def generate_for_intent(client, log, intent, siblings):
    """Ask the model for one topic. Saves the result in data/generated/."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(intent, siblings)},
    ]
    for attempt in range(1, MAX_TRIES + 1):
        try:
            result = call_chat(client, GENERATOR_MODEL, messages, log,
                               max_completion_tokens=4000,
                               response_format={"type": "json_object"})
            styles = parse_reply(result["answer"])
            data = {"intent_id": intent["id"], "model": result["model"], "styles": styles}
            path = GENERATED_DIR / f"{intent['id']}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return
        except Exception as error:
            print(f"    attempt {attempt} failed: {error}")
    raise SystemExit(f"Could not generate topic {intent['id']}. Run the script again.")


def clean_text(text):
    """Make quotes and dashes plain ASCII. Returns None if other odd characters remain."""
    replacements = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u20b9": "Rs "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return None
    return text


def normalize(text):
    """Lowercase, no punctuation: used only to find duplicates."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def assemble(intents):
    """Read data/generated/, clean, remove duplicates, split, write queries.jsonl."""
    # Collect every generated message
    items = []  # (intent, style, text)
    for intent in intents:
        path = GENERATED_DIR / f"{intent['id']}.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for style, texts in data["styles"].items():
            for text in texts:
                items.append((intent, style, text))

    canonical = {normalize(i["question"]) for i in intents}
    topics_of_text = {}  # normalized text -> set of topics that produced it
    for intent, style, text in items:
        topics_of_text.setdefault(normalize(text), set()).add(intent["id"])

    removed = {"bad characters": 0, "same as policy question": 0,
               "text under 2 topics": 0, "duplicate": 0}
    seen = set()
    kept = []
    for intent, style, text in items:
        text = clean_text(text)
        if text is None or len(text) > 300 or not normalize(text):
            removed["bad characters"] += 1
            continue
        norm = normalize(text)
        if norm in canonical:
            removed["same as policy question"] += 1
        elif len(topics_of_text.get(norm, set())) > 1:
            removed["text under 2 topics"] += 1
        elif norm in seen:
            removed["duplicate"] += 1
        else:
            seen.add(norm)
            kept.append({"intent_id": intent["id"], "family": intent["family"],
                         "style": style, "text": text})

    # Split each topic 50/50 into calibration and test
    rng = random.Random(SEED)
    by_topic = {}
    for row in kept:
        by_topic.setdefault(row["intent_id"], []).append(row)
    for rows in by_topic.values():
        rng.shuffle(rows)
        half = len(rows) // 2
        for position, row in enumerate(rows):
            row["split"] = "calibration" if position < half else "test"

    # Give every query an id and write the file
    final = [row for rows in by_topic.values() for row in rows]
    for number, row in enumerate(final, start=1):
        row["query_id"] = f"q{number:04d}"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")

    return final, removed


def print_summary(final, removed):
    print(f"\nQueries written to {OUTPUT_FILE}: {len(final)}")
    print("Removed:", ", ".join(f"{k}: {v}" for k, v in removed.items()))

    for field in ["style", "split"]:
        counts = {}
        for row in final:
            counts[row[field]] = counts.get(row[field], 0) + 1
        print(f"By {field}:", counts)

    per_topic = {}
    for row in final:
        per_topic[row["intent_id"]] = per_topic.get(row["intent_id"], 0) + 1
    print(f"Queries per topic: min {min(per_topic.values())}, max {max(per_topic.values())}")

    rng = random.Random(1)
    print("\n12 random queries:")
    for row in rng.sample(final, 12):
        print(f"  [{row['style']:<12}] ({row['intent_id']}) {row['text']}")

    near_misses = [row for row in final if row["style"] == "minimal_edit"]
    print("\n8 random minimal_edit queries (the hard near-misses):")
    for row in rng.sample(near_misses, 8):
        print(f"  ({row['intent_id']}) {row['text']}")


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

    client = OpenAI(api_key=api_key)
    log = UsageLog()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    policy = load_policy()
    intents = policy["intents"]
    by_family = {}
    for intent in intents:
        by_family.setdefault(intent["family"], []).append(intent)

    spent_before = log.total_cost()
    for number, intent in enumerate(intents, start=1):
        path = GENERATED_DIR / f"{intent['id']}.json"
        if path.exists():
            print(f"[{number}/{len(intents)}] {intent['id']}: already generated, skipping")
            continue
        print(f"[{number}/{len(intents)}] generating {intent['id']} ...")
        siblings = [s for s in by_family[intent["family"]] if s["id"] != intent["id"]]
        generate_for_intent(client, log, intent, siblings)

    final, removed = assemble(intents)
    print_summary(final, removed)
    print(f"\nSpent in this run: ${log.total_cost() - spent_before:.4f}")


main()