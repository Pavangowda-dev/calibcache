# Checks the policy file and prints a summary: topics per family,
# and a rough size of the system prompt.
from calibcache.policy import load_policy, policy_as_text, support_system_prompt

policy = load_policy()
intents = policy["intents"]

print(f"App: {policy['app_name']}")
print(f"Topics (intents): {len(intents)}")

# Group topic ids by family
families = {}
for intent in intents:
    families.setdefault(intent["family"], []).append(intent["id"])

print(f"Families: {len(families)}\n")
for family, ids in families.items():
    print(f"{family} ({len(ids)}):")
    for intent_id in ids:
        print(f"    {intent_id}")

prompt = support_system_prompt(policy)
print(f"\nSystem prompt: {len(prompt)} characters, "
      f"about {len(prompt) // 4} tokens (rough guess)")
print("\nFirst 500 characters of the system prompt:\n")
print(prompt[:500])