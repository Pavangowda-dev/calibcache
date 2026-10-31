# Price table and cost calculator.
# Prices are in US dollars per 1 million tokens.
# Checked on 28 Oct 2026 from OpenAI's launch post and third-party tables.
# Always compare with https://developers.openai.com/api/docs/pricing/ before final runs.

CHAT_PRICES = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
}

EMBEDDING_PRICES = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


def find_model_key(model, table):
    """Match an exact version name like 'gpt-5.4-mini-2026-03-17' to 'gpt-5.4-mini'."""
    best = None
    for key in table:
        if model == key or model.startswith(key + "-"):
            # keep the longest match
            if best is None or len(key) > len(best):
                best = key
    if best is None:
        # never guess a price: stop and tell me to add it
        raise ValueError(f"No price for model '{model}'. Add it in calibcache/pricing.py")
    return best


def chat_cost(model, input_tokens, output_tokens):
    """Dollar cost of one chat call."""
    key = find_model_key(model, CHAT_PRICES)
    price = CHAT_PRICES[key]
    total = input_tokens * price["input"] + output_tokens * price["output"]
    return total / 1_000_000


def embedding_cost(model, tokens):
    """Dollar cost of one embedding call."""
    key = find_model_key(model, EMBEDDING_PRICES)
    return tokens * EMBEDDING_PRICES[key] / 1_000_000