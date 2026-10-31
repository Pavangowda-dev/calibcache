# A cheap second opinion for the middle band:
# "Is this saved answer a correct and complete answer to the new question?"
import time

from calibcache.config import VERIFIER_MODEL
from calibcache.openai_calls import call_chat

SYSTEM_PROMPT = ("You check whether a saved customer support answer can be reused "
                 "for a new question. You answer with one word.")


def build_messages(cached_question, cached_answer, new_question):
    """The instructions for one check."""
    user = (
        "A customer asked this question:\n"
        f"{cached_question}\n\n"
        "Support replied:\n"
        f"{cached_answer}\n\n"
        "A new customer now asks:\n"
        f"{new_question}\n\n"
        "Is the reply above a fully correct and complete answer to the NEW question, "
        "with nothing missing and nothing wrong for the new customer's situation? "
        "Be strict: if the new question is about a different situation, a different "
        "feature, or needs different facts, answer NO. "
        "Answer with exactly one word: YES or NO."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def verify_pair(client, log, cached_question, cached_answer, new_question,
                model=VERIFIER_MODEL, tries=3):
    """Returns {"yes": True/False, "cost": dollars, "seconds": time}."""
    messages = build_messages(cached_question, cached_answer, new_question)
    last_error = None
    for attempt in range(tries):
        try:
            result = call_chat(client, model, messages, log, max_completion_tokens=300)
            word = result["answer"].strip().upper()
            if word.startswith("YES"):
                yes = True
            elif word.startswith("NO"):
                yes = False
            else:
                raise ValueError(f"unexpected reply: {result['answer'][:40]!r}")
            return {"yes": yes, "cost": result["cost_usd"], "seconds": result["seconds"]}
        except Exception as error:
            last_error = error
            time.sleep(2)  # wait a little, then try again
    raise last_error