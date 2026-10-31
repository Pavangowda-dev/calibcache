# The two plain OpenAI calls used everywhere: chat and embeddings.
# Each call is timed, priced and saved in the usage log.
import time

from calibcache.pricing import chat_cost, embedding_cost


def call_chat(client, model, messages, log, max_completion_tokens=1000,
              response_format=None):
    """Ask one chat question. Returns the answer plus tokens, cost and time.
    response_format={"type": "json_object"} makes the model reply in JSON."""
    extra = {}
    if response_format is not None:
        extra["response_format"] = response_format

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        **extra,
    )
    seconds = time.time() - start

    usage = response.usage
    # Thinking tokens are already inside completion_tokens (billed as output)
    details = usage.completion_tokens_details
    thinking_tokens = (details.reasoning_tokens or 0) if details else 0

    model_used = response.model  # exact version, e.g. gpt-5.4-mini-2026-03-17
    cost = chat_cost(model_used, usage.prompt_tokens, usage.completion_tokens)
    log.add("chat", model_used, usage.prompt_tokens,
            usage.completion_tokens, cost, seconds)

    return {
        "answer": response.choices[0].message.content or "",
        "finish_reason": response.choices[0].finish_reason,  # "stop" = finished normally
        "model": model_used,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "thinking_tokens": thinking_tokens,
        "cost_usd": cost,
        "seconds": seconds,
    }


def call_embeddings(client, model, texts, log):
    """Turn a list of texts into vectors. Returns vectors plus tokens, cost and time."""
    start = time.time()
    response = client.embeddings.create(model=model, input=texts)
    seconds = time.time() - start

    tokens = response.usage.prompt_tokens
    model_used = response.model
    cost = embedding_cost(model_used, tokens)
    log.add("embedding", model_used, tokens, 0, cost, seconds)

    vectors = [item.embedding for item in response.data]
    return {
        "vectors": vectors,
        "model": model_used,
        "tokens": tokens,
        "cost_usd": cost,
        "seconds": seconds,
    }