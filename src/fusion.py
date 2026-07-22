"""Fusión de rankings con Reciprocal Rank Fusion (RRF)."""

# RRF: cada documento acumula score(d) = Σ 1/(K + rank_r(d)) sobre cada ranking
# r donde aparece (rank empieza en 1). Solo usa posiciones, así que combina
# rankings cuyos scores no son comparables entre sí (coseno vs BM25) sin
# normalizar nada. Un documento bien posicionado en AMBAS listas suma dos
# términos grandes y sube; K=60 (el valor del paper original de RRF) amortigua
# las diferencias entre posiciones cercanas y evita que el #1 de una sola lista
# domine la fusión.
RRF_K = 60


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fusiona listas de doc-ids ordenadas por relevancia.

    Devuelve [(doc_id, score_rrf)] ordenado de mayor a menor.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
