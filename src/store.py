"""Almacén vectorial local por documento: numpy (.npz) + JSON en data/index/.

Cada documento del corpus tiene su propio par de archivos
(data/index/<doc_id>.npz y .json), lo que hace la ingesta idempotente por
documento. El loader concatena todos en una sola matriz y una sola lista de
chunks (dicts con text, company, fiscal_year, doc_id y chunk_no); los índices
globales de retrieval son posiciones en esa concatenación.
"""

import json
import sys

import numpy as np

from .config import FILINGS, INDEX_DIR


def _paths(doc_id: str):
    return INDEX_DIR / f"{doc_id}.npz", INDEX_DIR / f"{doc_id}.json"


def doc_indexed(doc_id: str) -> bool:
    vectors_path, chunks_path = _paths(doc_id)
    return vectors_path.exists() and chunks_path.exists()


def save_doc(doc: dict, vectors: np.ndarray, chunk_texts: list[str], meta: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path, chunks_path = _paths(doc["doc_id"])
    # Se guardan normalizados: la similitud coseno queda en un producto punto.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.savez_compressed(vectors_path, vectors=vectors / norms)
    chunks = [
        {
            "doc_id": doc["doc_id"],
            "company": doc["company"],
            "fiscal_year": doc["fiscal_year"],
            "chunk_no": i,
            "text": text,
        }
        for i, text in enumerate(chunk_texts)
    ]
    chunks_path.write_text(
        json.dumps({"meta": meta, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )


def load() -> tuple[np.ndarray, list[dict]]:
    all_vectors: list[np.ndarray] = []
    all_chunks: list[dict] = []
    for doc in FILINGS:
        vectors_path, chunks_path = _paths(doc["doc_id"])
        if not doc_indexed(doc["doc_id"]):
            sys.exit(
                f"ERROR: falta el índice de {doc['doc_id']}. Corre primero la ingesta:\n"
                "    uv run python -m src.ingest"
            )
        all_vectors.append(np.load(vectors_path)["vectors"])
        all_chunks.extend(json.loads(chunks_path.read_text(encoding="utf-8"))["chunks"])
    return np.vstack(all_vectors), all_chunks


def top_k(query_vector: np.ndarray, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Índices y similitudes de los k chunks más parecidos a la consulta."""
    q = query_vector / np.linalg.norm(query_vector)
    scores = vectors @ q
    idx = np.argsort(scores)[::-1][:k]
    return idx, scores[idx]
