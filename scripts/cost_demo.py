# Cost demo:
# 1) the same question to the cheap and the strong model (compare cost and time)
# 2) embeddings for 4 questions (see how similar a paraphrase and a near-miss look)
import math
import os

from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import CHEAP_MODEL, STRONG_MODEL, EMBEDDING_MODEL
from calibcache.openai_calls import call_chat, call_embeddings
from calibcache.usage_log import UsageLog

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

client = OpenAI(api_key=api_key)
log = UsageLog()  # saves every call in results/usage_log.jsonl


def cosine(a, b):
    # Similarity of two vectors: 1.0 = same direction, near 0 = unrelated
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# ---------- Part 1: cheap model vs strong model ----------
question = "My UPI payment failed but the money was deducted. What should I do?"
messages = [
    {"role": "system",
     "content": "You are a customer support assistant for a UPI payments app. "
                "Answer in 3 short sentences."},
    {"role": "user", "content": question},
]

print("=== Part 1: same question, two models ===")
for model in [CHEAP_MODEL, STRONG_MODEL]:
    result = call_chat(client, model, messages, log)
    print(f"\nModel         : {result['model']}")
    print(f"Answer        : {result['answer']}")
    print(f"Input tokens  : {result['input_tokens']}")
    print(f"Output tokens : {result['output_tokens']} "
          f"(thinking: {result['thinking_tokens']})")
    print(f"Cost          : ${result['cost_usd']:.6f}")
    print(f"Time          : {result['seconds']:.2f} seconds")

# ---------- Part 2: embeddings and similarity ----------
texts = [
    "My UPI payment failed but money was deducted. What should I do?",           # original
    "Money got debited from my account but the UPI transaction failed. How do I get it back?",  # paraphrase
    "My UPI payment is still pending and money was deducted. What should I do?",  # near-miss
    "How do I change my UPI PIN?",                                                # unrelated
]

print("\n=== Part 2: how similar do the questions look? ===")
result = call_embeddings(client, EMBEDDING_MODEL, texts, log)
vectors = result["vectors"]
print(f"Embedding model: {result['model']}, tokens: {result['tokens']}, "
      f"cost: ${result['cost_usd']:.8f}, vector size: {len(vectors[0])}")

print(f"\nOriginal vs paraphrase (same meaning)   : {cosine(vectors[0], vectors[1]):.3f}")
print(f"Original vs near-miss (needs new answer): {cosine(vectors[0], vectors[2]):.3f}")
print(f"Original vs unrelated question          : {cosine(vectors[0], vectors[3]):.3f}")

print()
log.print_summary()