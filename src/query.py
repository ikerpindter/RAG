"""CLI de consulta: recuperación híbrida (dense + BM25 fusionados con RRF) -> respuesta."""

import argparse

from . import store
from .bm25 import BM25Index
from .config import GENERATION_MODEL, TOP_K, get_openai_client
from .embeddings import embed_texts
from .fusion import rrf

# Candidatos por método antes de fusionar: holgura suficiente para que RRF
# pueda subir chunks que un método ranquea bien y el otro ignora.
CANDIDATES_PER_METHOD = 20

INSTRUCTIONS = (
    "Respondes preguntas sobre el reporte 10-K de Lennar del año fiscal 2024. "
    "Responde ÚNICAMENTE con la información de los extractos proporcionados. "
    "Si la respuesta no está en los extractos, di claramente que no aparece en el "
    "contexto recuperado. No inventes cifras ni datos."
)


def print_debug(
    dense: list[tuple[int, float]],
    bm25: list[tuple[int, float]],
    fused: list[tuple[int, float]],
    top: list[int],
) -> None:
    print("== ranking dense (chunk: coseno) ==")
    print("  " + ", ".join(f"{i}: {s:.2f}" for i, s in dense))
    print("== ranking BM25 (chunk: score) ==")
    if bm25:
        print("  " + ", ".join(f"{i}: {s:.1f}" for i, s in bm25))
    else:
        print("  (ningún chunk contiene términos de la consulta)")
    print("== ranking fusionado RRF (chunk: score) ==")
    print("  " + ", ".join(f"{i}: {s:.4f}" for i, s in fused[: 2 * TOP_K]))
    dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
    bm25_rank = {i: r for r, (i, _) in enumerate(bm25, start=1)}
    print("== top-5 final y su origen ==")
    for i in top:
        in_dense, in_bm25 = i in dense_rank, i in bm25_rank
        if in_dense and in_bm25:
            origin = f"ambos (dense #{dense_rank[i]}, bm25 #{bm25_rank[i]})"
        elif in_dense:
            origin = f"solo dense (#{dense_rank[i]})"
        else:
            origin = f"solo bm25 (#{bm25_rank[i]})"
        print(f"  chunk {i}: {origin}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.query",
        description="Pregunta sobre el 10-K con recuperación híbrida dense+BM25.",
    )
    parser.add_argument("question", help="pregunta sobre el 10-K")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="imprime los rankings dense, BM25 y fusionado, y el origen del top-5",
    )
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        parser.error("la pregunta está vacía")

    vectors, chunks = store.load()
    client = get_openai_client()

    # Ranking dense: embeddings + coseno (lo que ya existía).
    query_vector = embed_texts(client, [question])[0]
    dense_idx, dense_scores = store.top_k(query_vector, vectors, k=CANDIDATES_PER_METHOD)
    dense = [(int(i), float(s)) for i, s in zip(dense_idx, dense_scores)]

    # Ranking BM25: keyword literal sobre los mismos chunks.
    bm25 = BM25Index(chunks).search(question, k=CANDIDATES_PER_METHOD)

    # Fusión por posiciones y top-5 final.
    fused = rrf([[i for i, _ in dense], [i for i, _ in bm25]])
    top = [doc_id for doc_id, _ in fused[:TOP_K]]

    if args.debug:
        print_debug(dense, bm25, fused, top)

    print(f"(top-{len(top)} híbrido: dense+BM25 fusionados con RRF)\n")
    context = "\n\n".join(
        f"[Extracto {rank + 1}]\n{chunks[i]}" for rank, i in enumerate(top)
    )
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=INSTRUCTIONS,
        input=f"{context}\n\nPregunta: {question}",
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
