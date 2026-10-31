# Loads the fictional PayLoom support policy, checks it, and turns it into
# the system prompt for the support assistant.
import json

POLICY_PATH = "data/policy_intents.json"
REQUIRED_KEYS = ["id", "family", "question", "answer"]


def check_policy(policy):
    """Stop with a clear message if the policy file has a mistake."""
    seen_ids = set()
    for intent in policy["intents"]:
        for key in REQUIRED_KEYS:
            if not str(intent.get(key, "")).strip():
                raise ValueError(f"Intent {intent.get('id')} has an empty '{key}'")
        if intent["id"] in seen_ids:
            raise ValueError(f"Duplicate intent id: {intent['id']}")
        seen_ids.add(intent["id"])


def load_policy(path=POLICY_PATH):
    """Read the policy file (JSON) and check it."""
    with open(path, encoding="utf-8") as f:
        policy = json.load(f)
    check_policy(policy)
    return policy


def policy_as_text(policy):
    """All topics as plain text lines: question + answer."""
    lines = [f"Support policy for {policy['app_name']}:"]
    for intent in policy["intents"]:
        lines.append(f"- {intent['question']} {intent['answer']}")
    return "\n".join(lines)


def support_system_prompt(policy):
    """The system prompt: instructions + the whole policy."""
    intro = (
        f"You are the customer support assistant for {policy['app_name']}, "
        "a UPI payments app. Answer only using the support policy below. "
        "Be short and clear. If the policy does not cover the question, say you "
        "do not know and suggest Help > Contact support.\n\n"
    )
    return intro + policy_as_text(policy)