"""Embeddings con OpenAI, en batches."""

import numpy as np

from .config import EMBEDDING_MODEL

BATCH_SIZE = 100


def embed_texts(client, texts: list[str]) -> np.ndarray:
    """Convierte una lista de textos en una matriz (n, dim) de float32."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
        if len(texts) > BATCH_SIZE:
            print(f"  embeddings: {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.asarray(vectors, dtype=np.float32)
