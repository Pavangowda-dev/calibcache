# Turns many queries into vectors (in batches) and saves them in a file,
# so we never pay twice for the same queries.
from pathlib import Path

import numpy as np

from calibcache.config import EMBEDDING_MODEL
from calibcache.openai_calls import call_embeddings


def embed_queries(client, log, rows, cache_path, model=EMBEDDING_MODEL, batch_size=100):
    """Returns a matrix with one normalized vector per query (same order as rows)."""
    ids = [row["query_id"] for row in rows]
    path = Path(cache_path)

    # If we already embedded exactly these queries, load them from the file
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if list(saved["ids"]) == ids and str(saved["model"]) == model:
                return saved["vectors"]

    vectors = []
    for start in range(0, len(rows), batch_size):
        texts = [row["text"] for row in rows[start:start + batch_size]]
        result = call_embeddings(client, model, texts, log)
        vectors.extend(result["vectors"])

    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)  # length 1

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), vectors=matrix, model=np.array(model))
    return matrix