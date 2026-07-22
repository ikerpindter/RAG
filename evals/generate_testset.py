"""Genera un BORRADOR de testset sintético con Ragas desde los chunks indexados.

Es un borrador a propósito (modelo barato, gpt-5.4-nano): las preguntas y
respuestas de referencia las revisa y corrige un experto a mano en
evals/testset_draft.json antes de usarse como gold set. Cambia el campo
"status" de cada pregunta a "approved" al aprobarla.

Uso:
    uv run python -m evals.generate_testset
"""

import json

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

from src import store
from src.config import PROJECT_ROOT, TESTSET_GENERATOR_MODEL, get_openai_client

TESTSET_SIZE = 8
OUTPUT_PATH = PROJECT_ROOT / "evals" / "testset_draft.json"


def main() -> None:
    # Valida la key y deja OPENAI_API_KEY en el entorno para ChatOpenAI.
    client = get_openai_client()
    _, chunks = store.load()
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]
    print(f"Generando {TESTSET_SIZE} preguntas desde {len(documents)} chunks "
          f"con {TESTSET_GENERATOR_MODEL}...")

    generator = TestsetGenerator(
        # temperature=1: los modelos de razonamiento actuales (gpt-5.4-*) solo
        # aceptan ese valor; el default de ChatOpenAI (0.7) provoca un 400.
        llm=LangchainLLMWrapper(ChatOpenAI(model=TESTSET_GENERATOR_MODEL, temperature=1)),
        embedding_model=OpenAIEmbeddings(client=client),
    )
    dataset = generator.generate_with_langchain_docs(documents, testset_size=TESTSET_SIZE)

    questions = []
    for n, record in enumerate(dataset.to_pandas().to_dict("records"), start=1):
        questions.append(
            {
                "id": n,
                "status": "draft",
                "question": record["user_input"],
                "reference": record["reference"],
                "source_contexts": list(record["reference_contexts"]),
                "synthesizer": record.get("synthesizer_name", ""),
                "reviewer_notes": "",
            }
        )
    payload = {
        "meta": {
            "generator_model": TESTSET_GENERATOR_MODEL,
            "source": "ragas TestsetGenerator sobre data/index/chunks.json",
            "note": "Borrador sintético: revisar/corregir a mano antes de usar como gold set.",
        },
        "questions": questions,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Guardadas {len(questions)} preguntas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
