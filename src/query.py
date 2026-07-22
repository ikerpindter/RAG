"""CLI de consulta: análisis de la pregunta -> retrieval híbrido (dense + BM25 +
RRF) con filtrado por metadata y decomposition comparativa -> respuesta citada.

`retrieve()` y `generate()` son la tubería real y las importa también el
harness de evals, para que lo evaluado sea exactamente lo que corre el CLI.
"""

import argparse

from . import store
from .bm25 import BM25Index
from .citations import CITATION_INSTRUCTIONS, render_sources, source_label
from .config import (
    COMPARATIVE_K_PER_COMPANY,
    GENERATION_MODEL,
    TOP_K,
    get_openai_client,
)
from .embeddings import embed_texts
from .fusion import rrf
from .query_analysis import analyze_query

# Candidatos por método antes de fusionar: holgura suficiente para que RRF
# pueda subir chunks que un método ranquea bien y el otro ignora.
CANDIDATES_PER_METHOD = 20

INSTRUCTIONS = (
    "Responde SIEMPRE en el idioma en el que esté formulada la pregunta: "
    "pregunta en inglés, respuesta en inglés. "
    "Respondes preguntas sobre reportes 10-K de constructoras de vivienda. "
    "Cada extracto viene etiquetado con su empresa y año fiscal; no mezcles "
    "datos de una empresa o año con otra. "
    "Responde ÚNICAMENTE con la información de los extractos proporcionados. "
    "Si la respuesta no está en los extractos, di claramente que no aparece en el "
    "contexto recuperado. No inventes cifras ni datos."
)


def _allowed_indices(chunks: list[dict], companies: list[str], years: list[int]) -> list[int]:
    return [
        i
        for i, chunk in enumerate(chunks)
        if (not companies or chunk["company"] in companies)
        and (not years or chunk["fiscal_year"] in years)
    ]


def _hybrid_rank(
    query_vector, question: str, vectors, chunks: list[dict], allowed: list[int], k: int
) -> dict:
    """Retrieval híbrido sobre un subconjunto de chunks (índices globales)."""
    if not allowed:
        raise RuntimeError("El filtro de metadata dejó cero chunks; esto no debería pasar.")
    n_candidates = min(CANDIDATES_PER_METHOD, len(allowed))
    sub_idx, sub_scores = store.top_k(query_vector, vectors[allowed], k=n_candidates)
    dense = [(allowed[int(i)], float(s)) for i, s in zip(sub_idx, sub_scores)]
    # BM25 se reconstruye sobre el subconjunto (IDFs correctos del subcorpus);
    # con cientos de chunks son milisegundos.
    bm25_index = BM25Index([chunks[i]["text"] for i in allowed])
    bm25 = [(allowed[i], score) for i, score in bm25_index.search(question, k=n_candidates)]
    fused = rrf([[i for i, _ in dense], [i for i, _ in bm25]])
    return {"dense": dense, "bm25": bm25, "fused": fused, "top": [i for i, _ in fused[:k]]}


def retrieve(question: str, client, vectors, chunks: list[dict]) -> dict:
    """Analiza la pregunta, filtra por metadata y recupera el top final."""
    analysis = analyze_query(question, client)
    companies, years = analysis["companies"], analysis["fiscal_years"]
    query_vector = embed_texts(client, [question])[0]

    if analysis["comparative"] and len(companies) >= 2:
        # Decomposition: sub-retrieval por empresa para que ninguna domine.
        groups = []
        top: list[int] = []
        for company in companies:
            allowed = _allowed_indices(chunks, [company], years)
            result = _hybrid_rank(
                query_vector, question, vectors, chunks, allowed, COMPARATIVE_K_PER_COMPANY
            )
            groups.append({"company": company, "allowed_count": len(allowed), **result})
            top.extend(result["top"])
        return {"analysis": analysis, "mode": "decomposition", "groups": groups, "top": top}

    allowed = _allowed_indices(chunks, companies, years)
    result = _hybrid_rank(query_vector, question, vectors, chunks, allowed, TOP_K)
    mode = "filtrado" if (companies or years) else "sin filtro"
    return {"analysis": analysis, "mode": mode, "allowed_count": len(allowed), **result}


def generate(question: str, context_chunks: list[dict], client) -> str:
    """Genera la respuesta usando únicamente los chunks dados, etiquetados por fuente."""
    context = "\n\n".join(
        f"[Extracto {rank + 1} — {source_label(chunk)}]\n{chunk['text']}"
        for rank, chunk in enumerate(context_chunks)
    )
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=f"{INSTRUCTIONS} {CITATION_INSTRUCTIONS}",
        input=f"{context}\n\nPregunta: {question}",
    )
    return response.output_text


def _print_rankings(result: dict, chunks: list[dict]) -> None:
    dense, bm25 = result["dense"], result["bm25"]
    print("  dense (chunk global: coseno):")
    print("    " + ", ".join(f"{i}: {s:.2f}" for i, s in dense))
    print("  BM25 (chunk global: score):")
    if bm25:
        print("    " + ", ".join(f"{i}: {s:.1f}" for i, s in bm25))
    else:
        print("    (ningún chunk contiene términos de la consulta)")
    dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
    bm25_rank = {i: r for r, (i, _) in enumerate(bm25, start=1)}
    print("  top final y su origen:")
    for i in result["top"]:
        in_dense, in_bm25 = i in dense_rank, i in bm25_rank
        if in_dense and in_bm25:
            origin = f"ambos (dense #{dense_rank[i]}, bm25 #{bm25_rank[i]})"
        elif in_dense:
            origin = f"solo dense (#{dense_rank[i]})"
        else:
            origin = f"solo bm25 (#{bm25_rank[i]})"
        print(f"    [{source_label(chunks[i])}]: {origin}")


def print_debug(retrieval: dict, chunks: list[dict]) -> None:
    analysis = retrieval["analysis"]
    print("== query analysis ==")
    print(
        f"  companies={analysis['companies']}  fiscal_years={analysis['fiscal_years']}  "
        f"comparative={analysis['comparative']}"
    )
    if retrieval["mode"] == "decomposition":
        print(f"  modo: decomposition ({len(retrieval['groups'])} sub-retrievals)")
        for group in retrieval["groups"]:
            print(f"== sub-retrieval: {group['company']} "
                  f"({group['allowed_count']} chunks permitidos) ==")
            _print_rankings(group, chunks)
    else:
        print(f"  modo: {retrieval['mode']} ({retrieval['allowed_count']} chunks permitidos)")
        print("== rankings ==")
        _print_rankings(retrieval, chunks)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.query",
        description="Pregunta sobre los 10-K con retrieval híbrido filtrado por metadata.",
    )
    parser.add_argument("question", help="pregunta sobre los 10-K del corpus")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="imprime el análisis de la consulta, el filtro aplicado y los rankings",
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

    retrieval = retrieve(question, client, vectors, chunks)
    if args.debug:
        print_debug(retrieval, chunks)

    top = retrieval["top"]
    print(f"(top-{len(top)} híbrido, modo: {retrieval['mode']})\n")
    answer = generate(question, [chunks[i] for i in top], client)
    print(answer)
    print()
    print(render_sources(top, chunks))
    if args.verify:
        from .verify import print_verification, verify_answer

        print()
        print_verification(verify_answer(answer, top, chunks, client))


if __name__ == "__main__":
    main()
