# Wrapper demo: 11 requests through CalibClient with real model answers.
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calibcache import CalibClient
from calibcache.config import CHEAP_MODEL
from calibcache.policy import load_policy, support_system_prompt
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

# Start with an empty demo cache every time
db_path = Path("results/wrapper_demo.db")
if db_path.exists():
    db_path.unlink()

client = CalibClient(OpenAI(api_key=api_key), cache_path=db_path, log=UsageLog())
system = {"role": "system", "content": support_system_prompt(load_policy())}


def ask(question, tokens=1000, **extra):
    """Same usage style as the OpenAI client."""
    return client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[system, {"role": "user", "content": question}],
        max_completion_tokens=tokens,
        **extra,
    )


Q1 = "My payment failed but the money was deducted. What should I do?"
requests = [
    ("1  first question (new)", lambda: ask(Q1)),
    ("2  paraphrase of 1 (should HIT)",
     lambda: ask("Money got debited but the UPI transaction failed. How do I get it back?")),
    ("3  near-miss of 1: pending (should MISS)",
     lambda: ask("My payment is pending and the money was deducted. What should I do?")),
    ("4  Hinglish paraphrase of 1",
     lambda: ask("paise kat gaye lekin payment fail ho gaya, kya karu?")),
    ("5  daily limit (new)", lambda: ask("What is the maximum amount I can send per day?")),
    ("6  paraphrase of 5 (should HIT)", lambda: ask("how much can i send in one day")),
    ("7  near-miss of 5: single transaction",
     lambda: ask("What is the maximum I can send in a single transaction?")),
    ("8  same as 1, other user (should MISS)", lambda: ask(Q1, cache_namespace="user_b")),
    ("9  same as 1, other settings (should MISS)", lambda: ask(Q1, tokens=900)),
    ("10 longer conversation (bypass)", lambda: client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[system,
                  {"role": "user", "content": Q1},
                  {"role": "assistant", "content": "Please wait 3 working days."},
                  {"role": "user", "content": "And if it still has not come back?"}],
        max_completion_tokens=1000)),
    ("11 exact repeat of 1 (should HIT)", lambda: ask(Q1)),
]

print(f"{'request':<45} {'result':<7} {'score':>6} {'check':>6} {'seconds':>8} {'cost':>10}")
for label, run in requests:
    response = run()
    info = client.last_call
    score = "-" if info["score"] is None else f"{info['score']:.3f}"
    check = "-" if info["verifier"] is None else ("YES" if info["verifier"] else "NO")
    print(f"{label:<45} {info['decision'].upper():<7} {score:>6} {check:>6} "
          f"{info['seconds']:>8.2f} ${info['cost']:>9.6f}")
    answer = response.choices[0].message.content.replace("\n", " ")
    print(f"      answer: {answer[:110]}")

print()
client.report()
client.close()