# Cache demo (cheap: only embeddings, answers are fixed fake text).
# Shows the best similarity score of each test question, and whether it would
# be a HIT or MISS at different thresholds.
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calibcache.cache_store import CacheStore
from calibcache.config import CHEAP_MODEL, STRONG_MODEL
from calibcache.semantic_cache import SemanticCache, make_namespace
from calibcache.usage_log import UsageLog

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

client = OpenAI(api_key=api_key)

# Start from an empty demo database every time
db_path = Path("results/demo_cache.db")
if db_path.exists():
    db_path.unlink()

cache = SemanticCache(client, store=CacheStore(db_path), log=UsageLog())

system_prompt = "You are a customer support assistant for a UPI payments app."
group_cheap = make_namespace(CHEAP_MODEL, system_prompt, {})
group_strong = make_namespace(STRONG_MODEL, system_prompt, {})

# ---------- Save 3 questions (with fixed fake answers) ----------
saved = [
    ("My UPI payment failed but money was deducted. What should I do?",
     "ANSWER: failed payment, money deducted"),
    ("How do I change my UPI PIN?",
     "ANSWER: steps to change the UPI PIN"),
    ("What is the daily UPI transaction limit?",
     "ANSWER: the daily limit"),
]
print("Saving 3 questions in the cache...")
for question, answer in saved:
    cache.save(question, answer, group_cheap)
print(f"Entries saved: {cache.store.count()}\n")

# ---------- Test questions ----------
tests = [
    ("Paraphrase of saved question 1 (should HIT)",
     "Money got debited from my account but the UPI transaction failed. How do I get it back?",
     group_cheap),
    ("Near-miss of question 1: pending, not failed (should MISS)",
     "My UPI payment is still pending and money was deducted. What should I do?",
     group_cheap),
    ("Paraphrase of saved question 2 (should HIT)",
     "What are the steps to change my UPI PIN?",
     group_cheap),
    ("Near-miss of question 3: per-transaction limit, not daily (should MISS)",
     "What is the maximum amount I can send in a single UPI transaction?",
     group_cheap),
    ("Unrelated question (should MISS)",
     "How do I check my account balance?",
     group_cheap),
    ("Same words as question 1 but other model group (should MISS)",
     "Money got debited from my account but the UPI transaction failed. How do I get it back?",
     group_strong),
]

thresholds = [0.75, 0.80, 0.85, 0.90]

for label, question, group in tests:
    vector = cache.embed(question)
    entry_id, score = cache.find_best(vector, group)
    print(label)
    print(f"  question: {question}")
    if entry_id is None:
        print("  no saved questions in this group -> always MISS\n")
        continue
    matched = cache.store.get_entry(entry_id)["question"]
    print(f"  closest saved question: {matched}")
    print(f"  best score: {score:.3f}")
    decisions = []
    for t in thresholds:
        decisions.append(f"{t:.2f}: {'HIT ' if score >= t else 'miss'}")
    print("  " + " | ".join(decisions) + "\n")

# ---------- Real lookup with the default threshold ----------
print("=== Real lookup (threshold", cache.threshold, ") ===")
result = cache.lookup(
    "Money got debited from my account but the UPI transaction failed. How do I get it back?",
    group_cheap,
)
print("Hit:", result["hit"], "| score:", round(result["score"], 3))
print("Answer:", result["answer"])

print()
UsageLog().print_summary()
cache.store.close()