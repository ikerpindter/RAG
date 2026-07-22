from src.bm25 import BM25Index

CORPUS = [
    "total revenues increased to 100 in fiscal 2024",
    "revenues from home sales were strong this year",
    "the backlog dollar value was 5 billion at year end",
    "net revenues declined slightly compared to prior year",
]


def test_keyword_exacta_gana():
    results = BM25Index(CORPUS).search("backlog", k=3)
    assert results[0][0] == 2


def test_scores_cero_excluidos():
    # "backlog" solo existe en el doc 2: los demás no deben aparecer.
    results = BM25Index(CORPUS).search("backlog", k=10)
    assert [doc_id for doc_id, _ in results] == [2]
    assert all(score > 0 for _, score in results)


def test_limite_k():
    # "revenues" está en 3 de 4 docs; con k=2 devuelve exactamente 2.
    results = BM25Index(CORPUS).search("revenues", k=2)
    assert len(results) == 2
