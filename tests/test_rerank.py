from types import SimpleNamespace

import pytest

from src import rerank as rerank_module


class FakeCohere:
    """Simula cohere.ClientV2: resultados enlatados y fallas 429 opcionales."""

    def __init__(self, results, fail_429_times=0):
        self.calls = 0
        self.last_kwargs = None
        self._results = results
        self._fail_429_times = fail_429_times

    def rerank(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.calls <= self._fail_429_times:
            error = Exception("rate limited")
            error.status_code = 429
            raise error
        return SimpleNamespace(results=self._results)


@pytest.fixture
def sin_espera(monkeypatch):
    monkeypatch.setattr(rerank_module, "_RETRY_WAIT_S", 0)


def _use(monkeypatch, fake):
    monkeypatch.setattr(rerank_module, "_get_client", lambda: fake)


def test_reordena_y_mapea_a_ids_globales(monkeypatch, toy_chunks):
    # El reranker prefiere el documento en posición 1 del pool.
    fake = FakeCohere(
        results=[
            SimpleNamespace(index=1, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.2),
        ]
    )
    _use(monkeypatch, fake)
    result = rerank_module.rerank("pregunta", [0, 1], toy_chunks, top_k=2)
    assert result == [(1, 0.9), (0, 0.2)]


def test_manda_textos_y_top_n_correctos(monkeypatch, toy_chunks):
    fake = FakeCohere(results=[SimpleNamespace(index=0, relevance_score=0.5)])
    _use(monkeypatch, fake)
    rerank_module.rerank("pregunta", [1], toy_chunks, top_k=1)
    assert fake.last_kwargs["documents"] == [toy_chunks[1]["text"]]
    assert fake.last_kwargs["top_n"] == 1
    assert fake.last_kwargs["query"] == "pregunta"


def test_key_faltante_termina_con_mensaje(monkeypatch):
    monkeypatch.setattr(rerank_module, "_client", None)
    monkeypatch.setattr(rerank_module, "load_dotenv", lambda *_a, **_k: None)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="COHERE_API_KEY"):
        rerank_module._get_client()


def test_429_reintenta_y_recupera(monkeypatch, toy_chunks, sin_espera):
    fake = FakeCohere(
        results=[SimpleNamespace(index=0, relevance_score=0.7)], fail_429_times=1
    )
    _use(monkeypatch, fake)
    result = rerank_module.rerank("pregunta", [0], toy_chunks, top_k=1)
    assert result == [(0, 0.7)]
    assert fake.calls == 2


def test_429_persistente_da_error_claro(monkeypatch, toy_chunks, sin_espera):
    fake = FakeCohere(results=[], fail_429_times=99)
    _use(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="429"):
        rerank_module.rerank("pregunta", [0], toy_chunks, top_k=1)


def test_error_no_429_sube_tal_cual(monkeypatch, toy_chunks):
    class FakeRoto:
        def rerank(self, **kwargs):
            raise ValueError("otro error")

    _use(monkeypatch, FakeRoto())
    with pytest.raises(ValueError, match="otro error"):
        rerank_module.rerank("pregunta", [0], toy_chunks, top_k=1)
