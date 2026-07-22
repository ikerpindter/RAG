"""Almacén vectorial local: numpy (.npz) + JSON en data/index/, sin infra."""

import json
import sys

import numpy as np

from .config import CHUNKS_PATH, INDEX_DIR, VECTORS_PATH


def save(vectors: np.ndarray, chunks: list[str], meta: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Se guardan normalizados: la similitud coseno queda en un producto punto.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.savez_compressed(VECTORS_PATH, vectors=vectors / norms)
    CHUNKS_PATH.write_text(
        json.dumps({"meta": meta, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )


def load() -> tuple[np.ndarray, list[str]]:
    if not (VECTORS_PATH.exists() and CHUNKS_PATH.exists()):
        sys.exit(
            "ERROR: no existe el índice local. Corre primero la ingesta:\n"
            "    uv run python -m src.ingest"
        )
    vectors = np.load(VECTORS_PATH)["vectors"]
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]
    return vectors, chunks


def top_k(query_vector: np.ndarray, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Índices y similitudes de los k chunks más parecidos a la consulta."""
    q = query_vector / np.linalg.norm(query_vector)
    scores = vectors @ q
    idx = np.argsort(scores)[::-1][:k]
    return idx, scores[idx]
