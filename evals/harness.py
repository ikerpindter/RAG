"""Harness de evaluación: corre un gold set por el pipeline RAG híbrido real y
calcula 4 métricas de Ragas (API de colecciones, Ragas 0.4):

- faithfulness:      ¿la respuesta se sostiene en los chunks recuperados?
- answer_relevancy:  ¿la respuesta responde la pregunta?
- context_precision: ¿los chunks recuperados son relevantes (vs la referencia)?
- context_recall:    ¿los chunks recuperados cubren lo que la referencia necesita?

Uso:
    uv run python -m evals.harness                     # evals/goldset.json
    uv run python -m evals.harness --goldset otro.json
    uv run python -m evals.harness --limit 1           # smoke test barato
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src import store
from src.bm25 import BM25Index
from src.citations import strip_citations
from src.config import EVAL_JUDGE_MODEL, PROJECT_ROOT, get_openai_client
from src.query import generate, retrieve

DEFAULT_GOLDSET = PROJECT_ROOT / "evals" / "goldset.json"
RESULTS_PATH = PROJECT_ROOT / "evals" / "results.json"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_goldset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    for q in questions:
        if not q.get("question") or not q.get("reference"):
            raise ValueError(
                f"La pregunta id={q.get('id')} no tiene 'question' o 'reference'; "
                "corrige el gold set antes de evaluar."
            )
    return questions


def run_pipeline(questions: list[dict], client, vectors, chunks: list[str]) -> list[dict]:
    """Corre cada pregunta por la MISMA tubería que usa el CLI (retrieve + generate)."""
    bm25_index = BM25Index(chunks)
    rows = []
    for q in questions:
        top = retrieve(q["question"], client, vectors, chunks, bm25_index)["top"]
        contexts = [chunks[i] for i in top]
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "reference": q["reference"],
                # Se evalúa la respuesta limpia: los marcadores [n] de citas no
                # deben ensuciar las métricas.
                "response": strip_citations(generate(q["question"], contexts, client)),
                "retrieved_contexts": contexts,
            }
        )
        print(f"  pipeline [{q['id']}]: {q['question'][:70]}")
    return rows


async def score_rows(rows: list[dict]) -> list[dict]:
    judge_client = AsyncOpenAI()
    judge = llm_factory(EVAL_JUDGE_MODEL, client=judge_client)
    # Ragas 0.4.3 no reconoce los modelos con versión punteada (gpt-5.4-*,
    # gpt-5.6-*) como reasoning models: su detector hace int("5.4") -> ValueError
    # y termina mandando max_tokens/top_p/temperature=0.01, que estos modelos
    # rechazan. Se sobreescriben aquí los args con los que sí aceptan
    # (max_completion_tokens>=4096 para que el output estructurado no se trunque).
    judge.model_args = {"max_completion_tokens": 4096, "temperature": 1.0}
    embeddings = OpenAIEmbeddings(client=judge_client)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }
    results = []
    for row in rows:
        scores = {
            "faithfulness": (
                await metrics["faithfulness"].ascore(
                    user_input=row["question"],
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
            ).value,
            "answer_relevancy": (
                await metrics["answer_relevancy"].ascore(
                    user_input=row["question"], response=row["response"]
                )
            ).value,
            "context_precision": (
                await metrics["context_precision"].ascore(
                    user_input=row["question"],
                    reference=row["reference"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
            ).value,
            "context_recall": (
                await metrics["context_recall"].ascore(
                    user_input=row["question"],
                    retrieved_contexts=row["retrieved_contexts"],
                    reference=row["reference"],
                )
            ).value,
        }
        results.append({**row, "scores": scores})
        printable = ", ".join(f"{name}={scores[name]:.2f}" for name in METRIC_NAMES)
        print(f"  scores  [{row['id']}]: {printable}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m evals.harness",
        description="Evalúa el pipeline RAG contra un gold set con métricas de Ragas.",
    )
    parser.add_argument(
        "--goldset", type=Path, default=DEFAULT_GOLDSET,
        help=f"ruta al gold set JSON (default: {DEFAULT_GOLDSET})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="evalúa solo las primeras N preguntas (para smoke tests)",
    )
    args = parser.parse_args()

    client = get_openai_client()  # valida la key; AsyncOpenAI reutiliza el env
    questions = load_goldset(args.goldset)
    if args.limit:
        questions = questions[: args.limit]
    print(f"Evaluando {len(questions)} preguntas de {args.goldset.name} "
          f"(juez: {EVAL_JUDGE_MODEL})")

    vectors, chunks = store.load()
    rows = run_pipeline(questions, client, vectors, chunks)
    results = asyncio.run(score_rows(rows))

    means = {
        name: sum(r["scores"][name] for r in results) / len(results)
        for name in METRIC_NAMES
    }
    print("\n== promedios ==")
    for name in METRIC_NAMES:
        print(f"  {name}: {means[name]:.3f}")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "meta": {
                    "goldset": str(args.goldset),
                    "judge_model": EVAL_JUDGE_MODEL,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "n_questions": len(results),
                },
                "means": means,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResultados guardados en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
