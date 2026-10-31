# The semantic cache: embed a question, compare it with saved questions,
# and find the closest one. The decision (hit or miss) is made by the caller.
import hashlib
import json

import numpy as np

from calibcache.cache_store import CacheStore
from calibcache.config import EMBEDDING_MODEL
from calibcache.openai_calls import call_embeddings
from calibcache.usage_log import UsageLog


def make_namespace(model, system_prompt="", settings=None):
    """Same model + same system prompt + same settings -> same group name.
    We only compare questions inside the same group."""
    settings = settings or {}
    text = json.dumps(
        {"model": model, "system": system_prompt, "settings": settings},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class SemanticCache:
    def __init__(self, client, store=None, log=None,
                 embedding_model=EMBEDDING_MODEL,
                 threshold=0.90,      # only used by lookup(); CalibClient uses low/high
                 ttl_seconds=None):   # None = entries never expire
        self.client = client
        self.store = store or CacheStore()
        self.log = log if log is not None else UsageLog()
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.last_embed_cost = 0.0     # cost of the most recent embedding (dollars)
        self.last_embed_seconds = 0.0

    def embed(self, question):
        """Turn a question into a vector of length 1 (so dot product = cosine similarity)."""
        result = call_embeddings(self.client, self.embedding_model, [question], self.log)
        self.last_embed_cost = result["cost_usd"]
        self.last_embed_seconds = result["seconds"]
        vector = np.asarray(result["vectors"][0], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def find_best(self, vector, namespace):
        """Find the most similar saved question in this group.
        Returns (entry_id, score), or (None, None) if the group is empty.
        This does NOT use any threshold."""
        ids, matrix = self.store.load_vectors(
            namespace, self.embedding_model, self.ttl_seconds
        )
        if matrix is None:
            return None, None
        scores = matrix @ vector            # one similarity score per saved question
        best = int(np.argmax(scores))
        return ids[best], float(scores[best])

    def lookup(self, question, namespace):
        """Plain one-threshold check (used by the early demo)."""
        vector = self.embed(question)
        entry_id, score = self.find_best(vector, namespace)

        if entry_id is not None and score >= self.threshold:
            entry = self.store.get_entry(entry_id)
            self.store.add_hit(entry_id)
            return {"hit": True, "answer": entry["answer"], "score": score,
                    "matched_question": entry["question"], "vector": vector}

        return {"hit": False, "answer": None, "score": score,
                "matched_question": None, "vector": vector}

    def save(self, question, answer, namespace, vector=None,
             gen_cost=0.0, gen_seconds=0.0):
        """Save a question and its answer in the cache."""
        if vector is None:
            vector = self.embed(question)
        return self.store.add(namespace, self.embedding_model, question, answer,
                              vector, gen_cost, gen_seconds)