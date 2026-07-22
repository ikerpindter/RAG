import pytest

from src.fusion import RRF_K, rrf


def test_doc_en_ambas_listas_gana():
    fused = rrf([[1, 2, 3], [2, 4, 5]])
    assert fused[0][0] == 2


def test_orden_conocido_se_preserva():
    fused = rrf([[1, 2], [1, 2]])
    assert [doc_id for doc_id, _ in fused] == [1, 2]


def test_formula_con_k_60():
    fused = rrf([[7]])
    assert fused[0][1] == pytest.approx(1 / (RRF_K + 1))


def test_listas_vacias():
    assert rrf([[], []]) == []
