import pytest

from src.query_analysis import analyze_query

VALID = '{"companies": ["Lennar"], "fiscal_years": [2024], "comparative": false}'


def test_json_valido(fake_client):
    result = analyze_query("pregunta", fake_client(VALID))
    assert result == {"companies": ["Lennar"], "fiscal_years": [2024], "comparative": False}


def test_json_con_code_fence(fake_client):
    result = analyze_query("pregunta", fake_client(f"```json\n{VALID}\n```"))
    assert result["companies"] == ["Lennar"]


def test_json_malformado_error_claro(fake_client):
    with pytest.raises(RuntimeError, match="inválido"):
        analyze_query("pregunta", fake_client("esto no es json"))


def test_empresa_fuera_de_catalogo(fake_client):
    raw = '{"companies": ["Tesla"], "fiscal_years": [], "comparative": false}'
    with pytest.raises(RuntimeError, match="fuera del catálogo"):
        analyze_query("pregunta", fake_client(raw))


def test_anio_fuera_de_catalogo(fake_client):
    raw = '{"companies": [], "fiscal_years": [1999], "comparative": false}'
    with pytest.raises(RuntimeError, match="fuera del catálogo"):
        analyze_query("pregunta", fake_client(raw))


def test_listas_vacias_significan_sin_filtro(fake_client):
    raw = '{"companies": [], "fiscal_years": [], "comparative": false}'
    result = analyze_query("pregunta", fake_client(raw))
    assert result == {"companies": [], "fiscal_years": [], "comparative": False}
