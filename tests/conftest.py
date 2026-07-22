"""Fixtures de juguete: los tests corren sin data/, sin .env y sin red."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def toy_chunks():
    """Mini-corpus con la forma real de los chunks del índice (dicts + metadata)."""
    return [
        {
            "doc_id": "alfa-fy2024",
            "company": "Alfa",
            "fiscal_year": 2024,
            "chunk_no": 3,
            "text": "Alfa reportó ingresos totales de 100 millones en el año fiscal 2024. " * 5,
        },
        {
            "doc_id": "beta-fy2023",
            "company": "Beta",
            "fiscal_year": 2023,
            "chunk_no": 7,
            "text": "Beta cerró 500 casas en su año fiscal 2023.",
        },
    ]


class _FakeResponses:
    def __init__(self, output_text: str):
        self._output_text = output_text

    def create(self, **_kwargs):
        return SimpleNamespace(output_text=self._output_text)


@pytest.fixture
def fake_client():
    """Fábrica de clientes falsos: fake_client(texto) simula client.responses.create."""

    def _make(output_text: str):
        return SimpleNamespace(responses=_FakeResponses(output_text))

    return _make
