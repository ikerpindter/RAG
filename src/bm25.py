"""Búsqueda por keyword con BM25 sobre los chunks del índice local.

El corpus son 124 chunks, así que el índice se construye al vuelo en
milisegundos al arrancar el query; no hay nada que persistir y la ingesta
(y sus embeddings ya pagados) no se toca.
"""

import re

from rank_bm25 import BM25Okapi

# Lowercase + secuencias alfanuméricas. "$35,441,452" queda como
# ["35", "441", "452"]: suficiente para matching literal de cifras y términos
# sin un tokenizador sofisticado.
_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[str]):
        self._bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (índice de chunk, score BM25), de mayor a menor.

        Se excluyen los chunks con score 0 (ningún término de la consulta):
        meterlos al ranking solo agregaría ruido en orden arbitrario a la fusión.
        """
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order[:k] if scores[i] > 0]
