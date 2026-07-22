"""CLI de consulta: pregunta -> top-k chunks por coseno -> respuesta basada solo en ellos."""

import sys

from . import store
from .config import GENERATION_MODEL, TOP_K, get_openai_client
from .embeddings import embed_texts

INSTRUCTIONS = (
    "Respondes preguntas sobre el reporte 10-K de Lennar del año fiscal 2024. "
    "Responde ÚNICAMENTE con la información de los extractos proporcionados. "
    "Si la respuesta no está en los extractos, di claramente que no aparece en el "
    "contexto recuperado. No inventes cifras ni datos."
)


def main() -> None:
    if len(sys.argv) < 2 or not " ".join(sys.argv[1:]).strip():
        sys.exit('Uso: uv run python -m src.query "tu pregunta sobre el 10-K"')
    question = " ".join(sys.argv[1:]).strip()

    vectors, chunks = store.load()
    client = get_openai_client()

    query_vector = embed_texts(client, [question])[0]
    idx, scores = store.top_k(query_vector, vectors, k=TOP_K)
    similarities = ", ".join(f"{s:.2f}" for s in scores)
    print(f"(recuperados {len(idx)} chunks; similitud: {similarities})\n")

    context = "\n\n".join(
        f"[Extracto {rank + 1}]\n{chunks[i]}" for rank, i in enumerate(idx)
    )
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=INSTRUCTIONS,
        input=f"{context}\n\nPregunta: {question}",
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
