# The judge: a model that grades an answer against the reference answer.
import json
import time

from calibcache.config import STRONG_MODEL
from calibcache.openai_calls import call_chat

JUDGE_MODEL = STRONG_MODEL

# The SAME rules are given to the human labeller (me) and to the judge.
RUBRIC = """An answer is CORRECT if:
- it contains the essential facts of the reference answer (the numbers, time limits, steps, and where to go in the app), and
- nothing in it contradicts the reference answer, and
- it answers the question that was asked.
Friendly wording, a different order, or harmless extra detail do not matter.
A shorter answer is fine if nothing essential is missing.
An answer is INCORRECT if it misses an essential fact, gives a wrong fact or number,
answers a different question, or covers only part of what was asked."""

SYSTEM_PROMPT = ("You are a strict but fair grader of customer support answers. "
                 "You follow the output format exactly.")


def build_messages(question, reference, candidate):
    user = f"""Grade a customer support answer.

CUSTOMER QUESTION:
{question}

REFERENCE ANSWER (correct, from the support policy):
{reference}

ANSWER TO GRADE:
{candidate}

RULES:
{RUBRIC}

Return ONLY a JSON object like {{"verdict": "correct", "reason": "one short sentence"}}
where verdict is "correct" or "incorrect"."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def judge_answer(client, log, question, reference, candidate, model=JUDGE_MODEL, tries=3):
    """Returns {"correct": True/False, "reason": text, "cost": dollars, "seconds": time}."""
    messages = build_messages(question, reference, candidate)
    last_error = None
    for attempt in range(tries):
        try:
            result = call_chat(client, model, messages, log, max_completion_tokens=1000,
                               response_format={"type": "json_object"})
            data = json.loads(result["answer"])
            verdict = str(data["verdict"]).strip().lower()
            if verdict not in ("correct", "incorrect"):
                raise ValueError(f"unexpected verdict: {verdict!r}")
            return {"correct": verdict == "correct",
                    "reason": str(data.get("reason", "")).strip(),
                    "cost": result["cost_usd"], "seconds": result["seconds"]}
        except Exception as error:
            last_error = error
            time.sleep(2)
    raise last_error