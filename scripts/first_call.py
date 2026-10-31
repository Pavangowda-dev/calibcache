# Day 1: my first OpenAI call. It prints the answer, token usage and time.
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads the .env file

api_key = os.getenv("OPENAI_API_KEY")
chat_model = os.getenv("CHAT_MODEL")

if not api_key or not api_key.startswith("sk-"):
    raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")
if not chat_model:
    raise SystemExit("CHAT_MODEL is missing in .env")

client = OpenAI(api_key=api_key)

# A typical support question for our fictional UPI app
question = "My UPI payment failed but the money was deducted. What should I do?"

start = time.time()
response = client.chat.completions.create(
    model=chat_model,
    messages=[
        {
            "role": "system",
            "content": "You are a customer support assistant for a UPI "
                       "payments app. Answer in 3 short sentences.",
        },
        {"role": "user", "content": question},
    ],
    max_completion_tokens=1000,  # some models "think" first, so keep room
)
seconds = time.time() - start

answer = response.choices[0].message.content
usage = response.usage

# Some models count hidden "thinking" tokens; read them safely
details = usage.completion_tokens_details
thinking_tokens = (details.reasoning_tokens or 0) if details else 0

print("Model used     :", response.model)
print("Question       :", question)
print("Answer         :", answer if answer else "(empty answer)")
print("Input tokens   :", usage.prompt_tokens)
print("Output tokens  :", usage.completion_tokens)
print("  of which thinking:", thinking_tokens)
print("Total tokens   :", usage.total_tokens)
print(f"Time taken     : {seconds:.2f} seconds")