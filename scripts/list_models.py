# Day 1: list the models my API key can use, and check my CHAT_MODEL.
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads the .env file in the project folder

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

client = OpenAI(api_key=api_key)

# Ask OpenAI for all models this key can use
names = sorted(m.id for m in client.models.list())

# Hide models we do not need (audio, image, voice, etc.)
skip_words = ["audio", "realtime", "image", "tts", "transcribe",
              "search", "codex", "sora", "moderation", "instruct"]
useful = [
    n for n in names
    if (n.startswith("gpt-") or n.startswith("text-embedding"))
    and not any(word in n for word in skip_words)
]

print(f"Total models: {len(names)}. Showing chat/embedding models:\n")
for n in useful:
    print(n)

# Check the model I chose in .env
chat_model = os.getenv("CHAT_MODEL", "")
print()
if chat_model in names:
    print(f"OK: CHAT_MODEL '{chat_model}' is available to your key.")
else:
    print(f"PROBLEM: CHAT_MODEL '{chat_model}' is not in the list above.")