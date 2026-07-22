import numpy as np
import pytest

from src import store

TOY_DOC = {
    "doc_id": "toy-fy2024",
    "company": "Toy",
    "fiscal_year": 2024,
    "url": "https://example.com/toy.htm",
    "filename": "toy.htm",
}


@pytest.fixture
def toy_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(store, "FILINGS", [TOY_DOC])


def test_round_trip_conserva_vectores_y_metadata(toy_store):
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    store.save_doc(TOY_DOC, vectors, ["texto a", "texto b"], meta={"m": 1})
    loaded_vectors, chunks = store.load()
    # Los vectores se guardan normalizados (coseno = producto punto).
    assert np.allclose(np.linalg.norm(loaded_vectors, axis=1), 1.0)
    assert np.allclose(loaded_vectors[0], [0.6, 0.8])
    assert chunks[0] == {
        "doc_id": "toy-fy2024",
        "company": "Toy",
        "fiscal_year": 2024,
        "chunk_no": 0,
        "text": "texto a",
    }
    assert chunks[1]["text"] == "texto b"


def test_doc_indexed(toy_store):
    assert not store.doc_indexed("toy-fy2024")
    store.save_doc(TOY_DOC, np.ones((1, 3), dtype=np.float32), ["t"], meta={})
    assert store.doc_indexed("toy-fy2024")


def test_indice_faltante_termina_con_mensaje(toy_store):
    with pytest.raises(SystemExit, match="ingesta"):
        store.load()
