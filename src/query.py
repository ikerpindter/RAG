"""CLI de consulta: recuperación híbrida (dense + BM25 fusionados con RRF) -> respuesta.

`retrieve()` y `generate()` son la tubería real y las importa también el
harness de evals, para que lo evaluado sea exactamente lo que corre el CLI.
"""

import argparse

from . import store
from .bm25 import BM25Index
from .citations import CITATION_INSTRUCTIONS, render_sources, source_label
from .config import GENERATION_MODEL, TOP_K, get_openai_client
from .embeddings import embed_texts
from .fusion import rrf
from .verify import print_verification, verify_answer

# Candidatos por método antes de fusionar: holgura suficiente para que RRF
# pueda subir chunks que un método ranquea bien y el otro ignora.
CANDIDATES_PER_METHOD = 20

INSTRUCTIONS = (
    "Responde SIEMPRE en el idioma en el que esté formulada la pregunta: "
    "pregunta en inglés, respuesta en inglés. "
    "Respondes preguntas sobre el reporte 10-K de Lennar del año fiscal 2024. "
    "Responde ÚNICAMENTE con la información de los extractos proporcionados. "
    "Si la respuesta no está en los extractos, di claramente que no aparece en el "
    "contexto recuperado. No inventes cifras ni datos."
)


def retrieve(
    question: str,
    client,
    vectors,
    chunks: list[dict],
    bm25_index: BM25Index,
) -> dict:
    """Recuperación híbrida. Devuelve los rankings dense, bm25, fusionado y el top final."""
    query_vector = embed_texts(client, [question])[0]
    dense_idx, dense_scores = store.top_k(query_vector, vectors, k=CANDIDATES_PER_METHOD)
    dense = [(int(i), float(s)) for i, s in zip(dense_idx, dense_scores)]
    bm25 = bm25_index.search(question, k=CANDIDATES_PER_METHOD)
    fused = rrf([[i for i, _ in dense], [i for i, _ in bm25]])
    top = [doc_id for doc_id, _ in fused[:TOP_K]]
    return {"dense": dense, "bm25": bm25, "fused": fused, "top": top}


def generate(question: str, context_chunks: list[str], client) -> str:
    """Genera la respuesta usando únicamente los chunks dados como contexto."""
    context = "\n\n".join(
        f"[Extracto {rank + 1}]\n{chunk}" for rank, chunk in enumerate(context_chunks)
    )
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=f"{INSTRUCTIONS} {CITATION_INSTRUCTIONS}",
        input=f"{context}\n\nPregunta: {question}",
    )
    return response.output_text


def print_debug(retrieval: dict, chunks: list[dict]) -> None:
    dense, bm25 = retrieval["dense"], retrieval["bm25"]
    print("== ranking dense (chunk global: coseno) ==")
    print("  " + ", ".join(f"{i}: {s:.2f}" for i, s in dense))
    print("== ranking BM25 (chunk global: score) ==")
    if bm25:
        print("  " + ", ".join(f"{i}: {s:.1f}" for i, s in bm25))
    else:
        print("  (ningún chunk contiene términos de la consulta)")
    print("== ranking fusionado RRF (chunk global: score) ==")
    print("  " + ", ".join(f"{i}: {s:.4f}" for i, s in retrieval["fused"][: 2 * TOP_K]))
    dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
    bm25_rank = {i: r for r, (i, _) in enumerate(bm25, start=1)}
    print("== top-5 final y su origen ==")
    for i in retrieval["top"]:
        in_dense, in_bm25 = i in dense_rank, i in bm25_rank
        if in_dense and in_bm25:
            origin = f"ambos (dense #{dense_rank[i]}, bm25 #{bm25_rank[i]})"
        elif in_dense:
            origin = f"solo dense (#{dense_rank[i]})"
        else:
            origin = f"solo bm25 (#{bm25_rank[i]})"
        print(f"  [{source_label(chunks[i])}]: {origin}")
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verifica con un LLM barato que cada afirmación citada esté respaldada",
    )
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        parser.error("la pregunta está vacía")

    vectors, chunks = store.load()
    client = get_openai_client()

    bm25_index = BM25Index([c["text"] for c in chunks])
    retrieval = retrieve(question, client, vectors, chunks, bm25_index)
    if args.debug:
        print_debug(retrieval, chunks)

    top = retrieval["top"]
    print(f"(top-{len(top)} híbrido: dense+BM25 fusionados con RRF)\n")
    answer = generate(question, [chunks[i]["text"] for i in top], client)
    print(answer)
    print()
    print(render_sources(top, chunks))
    if args.verify:
        print()
        print_verification(verify_answer(answer, top, chunks, client))


if __name__ == "__main__":
    main()
