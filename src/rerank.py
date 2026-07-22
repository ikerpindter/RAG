"""Reranking de candidatos con Cohere.

Interfaz agnóstica de proveedor: el pipeline solo llama a rerank(); todo lo
específico de Cohere vive aquí, así que cambiar de proveedor es reescribir
este módulo sin tocar el pipeline.

Sin fallbacks silenciosos: si Cohere falla (key faltante, 429 persistente), la
consulta se detiene con error claro. Degradar en silencio a no-rerankear
contaminaría cualquier experimento antes/después sin que nadie se entere.
"""

import os
import sys
import time

from dotenv import load_dotenv

from .config import PROJECT_ROOT, RERANKER_MODEL

_MAX_RETRIES = 2
# La trial key de Cohere permite ~10 llamadas/min; una pausa corta suele bastar.
_RETRY_WAIT_S = 12

_client = None


def _get_client():
    global _client
    if _client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.environ.get("COHERE_API_KEY", "").strip()
        if not api_key:
            sys.exit(
                "ERROR: falta COHERE_API_KEY.\n"
                "Abre el archivo .env en la raíz del proyecto y pega tu clave:\n"
                "    COHERE_API_KEY=...\n"
                "(O apaga el reranker con RERANKER_ENABLED = False en src/config.py.)"
            )
        import cohere

        _client = cohere.ClientV2(api_key=api_key)
    return _client


def _rerank_with_retries(client, **kwargs):
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.rerank(**kwargs)
        except Exception as error:
            # Solo el rate limit (429) se reintenta; cualquier otro error sube tal cual.
            if getattr(error, "status_code", None) != 429:
                raise
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Cohere sigue devolviendo 429 tras {_MAX_RETRIES} reintentos "
                    "(rate limit de la trial key); espera un minuto y reintenta."
                ) from error
            print(
                f"  (Cohere 429: rate limit; reintento {attempt + 1}/{_MAX_RETRIES} "
                f"en {_RETRY_WAIT_S}s)"
            )
            time.sleep(_RETRY_WAIT_S)


def rerank(
    question: str, candidate_ids: list[int], chunks: list[dict], top_k: int
) -> list[tuple[int, float]]:
    """Reordena candidatos por relevancia contra la pregunta.

    Devuelve [(chunk_id global, relevance_score)] con los top_k mejores.
    """
    response = _rerank_with_retries(
        _get_client(),
        model=RERANKER_MODEL,
        query=question,
        documents=[chunks[i]["text"] for i in candidate_ids],
        top_n=top_k,
    )
    return [
        (candidate_ids[result.index], float(result.relevance_score))
        for result in response.results
    ]
