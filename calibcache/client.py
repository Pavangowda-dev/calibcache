# CalibClient: a drop-in wrapper around the OpenAI client.
#   1) If a saved answer is verified to fit the new question, it is returned from the cache.
#   2) Otherwise OpenAI is asked, and the answer is saved.
# Only single-question requests are cached (optional system message + one user message).
# Longer conversations, streaming and tool use go straight to OpenAI.
import time
from types import SimpleNamespace

from calibcache.cache_store import CacheStore
from calibcache.config import EMBEDDING_MODEL, VERIFIER_MODEL
from calibcache.pricing import chat_cost
from calibcache.semantic_cache import SemanticCache, make_namespace
from calibcache.usage_log import UsageLog
from calibcache.verifier import verify_pair

# Requests using these options are not cached
UNSUPPORTED = ("tools", "functions", "tool_choice", "function_call", "stream", "logprobs")
# These are not "settings" (they do not change which answers are compatible)
SKIP_IN_SETTINGS = {"model", "messages"}


def read_question(kwargs):
    """Returns (question, system_text) if this request can use the cache, else None."""
    for key in UNSUPPORTED:
        if kwargs.get(key):
            return None
    if kwargs.get("n", 1) not in (None, 1):
        return None

    messages = kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    system_parts = []
    user_messages = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            return None
        role = message.get("role")
        if role in ("system", "developer"):
            system_parts.append(message["content"])
        elif role == "user":
            user_messages.append(message["content"])
        else:
            return None  # assistant or tool messages: a longer conversation

    if len(user_messages) != 1 or messages[-1].get("role") != "user":
        return None
    return user_messages[0], "\n".join(system_parts)


def safe_cost(model, usage):
    """Dollar cost of an answer. 0 if the model is not in our price table."""
    try:
        return chat_cost(model, usage.prompt_tokens, usage.completion_tokens)
    except ValueError:
        return 0.0


def make_cached_response(answer, model, entry_id):
    """A small object shaped like an OpenAI chat response."""
    message = SimpleNamespace(role="assistant", content=answer)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return SimpleNamespace(id=f"calibcache-{entry_id}", object="chat.completion",
                           model=model, choices=[choice], usage=usage)


class _Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        return self.owner._create(kwargs)


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class CalibClient:
    def __init__(self, client, cache_path="calibcache.db",
                 low=0.65, high=0.90,
                 embedding_model=EMBEDDING_MODEL, verifier_model=VERIFIER_MODEL,
                 ttl_seconds=None, log=None):
        """client: your OpenAI() client.
        low : below this similarity -> always ask the model
        high: at or above this -> answer from the cache without a check
        between: a cheap model checks whether the saved answer fits."""
        self._client = client
        self.log = log if log is not None else UsageLog(None)  # in memory only
        self.cache = SemanticCache(client, store=CacheStore(cache_path), log=self.log,
                                   embedding_model=embedding_model, ttl_seconds=ttl_seconds)
        self.low = low
        self.high = high
        self.verifier_model = verifier_model
        self.last_call = None
        self.stats = {
            "requests": 0, "hits": 0, "misses": 0, "bypassed": 0,
            "verifier_checks": 0, "verifier_yes": 0,
            "model_cost": 0.0,      # dollars paid for model answers
            "overhead_cost": 0.0,   # dollars paid for embeddings and verifier checks
            "saved_cost": 0.0,      # what the answers served from the cache originally cost
            "hit_seconds": 0.0, "miss_seconds": 0.0,
        }
        self.chat = _Chat(self)

    def __getattr__(self, name):
        # Anything else (embeddings, models, ...) goes straight to the real client
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._client, name)

    def _create(self, kwargs):
        start = time.time()
        tag = kwargs.pop("cache_namespace", "")      # e.g. "user_123"
        bypass = kwargs.pop("cache_bypass", False)   # True = skip the cache
        self.stats["requests"] += 1

        parsed = None if bypass else read_question(kwargs)
        if parsed is None:
            return self._ask_model(kwargs, start, decision="bypass")

        question, system_text = parsed
        settings = {k: v for k, v in kwargs.items() if k not in SKIP_IN_SETTINGS}
        settings["cache_namespace"] = tag
        namespace = make_namespace(kwargs["model"], system_text, settings)

        vector = self.cache.embed(question)
        overhead = self.cache.last_embed_cost
        entry_id, score = self.cache.find_best(vector, namespace)

        hit = False
        verifier_said = None
        if entry_id is not None and score >= self.high:
            hit = True
        elif entry_id is not None and score >= self.low:
            entry = self.cache.store.get_entry(entry_id)
            check = verify_pair(self._client, self.log, entry["question"], entry["answer"],
                                question, model=self.verifier_model)
            overhead += check["cost"]
            verifier_said = check["yes"]
            self.stats["verifier_checks"] += 1
            if check["yes"]:
                self.stats["verifier_yes"] += 1
                hit = True

        if hit:
            entry = self.cache.store.get_entry(entry_id)
            self.cache.store.add_hit(entry_id)
            elapsed = time.time() - start
            s = self.stats
            s["hits"] += 1
            s["overhead_cost"] += overhead
            s["saved_cost"] += entry["gen_cost"]
            s["hit_seconds"] += elapsed
            self.last_call = {"decision": "hit", "score": score, "verifier": verifier_said,
                              "seconds": elapsed, "cost": overhead,
                              "saved": entry["gen_cost"] - overhead}
            return make_cached_response(entry["answer"], kwargs["model"], entry_id)

        return self._ask_model(kwargs, start, decision="miss", question=question,
                               namespace=namespace, vector=vector, score=score,
                               verifier_said=verifier_said, overhead=overhead)

    def _ask_model(self, kwargs, start, decision, question=None, namespace=None,
                   vector=None, score=None, verifier_said=None, overhead=0.0):
        """Ask the real model. On a miss, save the answer in the cache."""
        t0 = time.time()
        response = self._client.chat.completions.create(**kwargs)
        gen_seconds = time.time() - t0
        cost = safe_cost(response.model, response.usage)

        if decision == "miss":
            choice = response.choices[0]
            answer = choice.message.content or ""
            # never save empty or cut-off answers
            if answer and choice.finish_reason == "stop":
                self.cache.save(question, answer, namespace, vector,
                                gen_cost=cost, gen_seconds=gen_seconds)

        elapsed = time.time() - start
        s = self.stats
        s["misses" if decision == "miss" else "bypassed"] += 1
        s["model_cost"] += cost
        s["overhead_cost"] += overhead
        if decision == "miss":
            s["miss_seconds"] += elapsed
        self.last_call = {"decision": decision, "score": score, "verifier": verifier_said,
                          "seconds": elapsed, "cost": cost + overhead, "saved": 0.0}
        return response

    def report(self):
        """Print a summary of hits, misses, money and time."""
        s = self.stats
        n = max(s["requests"], 1)
        print(f"Requests: {s['requests']}  |  hits: {s['hits']} ({100 * s['hits'] / n:.1f}%)  "
              f"misses: {s['misses']}  bypassed: {s['bypassed']}")
        print(f"Verifier checks: {s['verifier_checks']} (accepted {s['verifier_yes']})")
        print(f"Paid for model answers      : ${s['model_cost']:.6f}")
        print(f"Paid for embeddings + checks: ${s['overhead_cost']:.6f}")
        print(f"Original cost of cached answers served: ${s['saved_cost']:.6f}")
        net = s["saved_cost"] - s["overhead_cost"]
        print(f"Net saving (served minus overhead)    : ${net:.6f}")
        if s["hits"]:
            print(f"Average time of a hit : {s['hit_seconds'] / s['hits']:.2f} seconds")
        if s["misses"]:
            print(f"Average time of a miss: {s['miss_seconds'] / s['misses']:.2f} seconds")

    def close(self):
        self.cache.store.close()