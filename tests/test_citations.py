from src.citations import render_sources, source_label, strip_citations


def test_strip_marcador_simple():
    assert strip_citations("Las ventas subieron 10%. [1]") == "Las ventas subieron 10%."


def test_strip_marcadores_multiples():
    assert strip_citations("A [1] y B [2] [3].") == "A y B."


def test_strip_no_dana_texto_sin_marcadores():
    text = "Rango [a-z] y costos de $5.4 billion (2024)."
    assert strip_citations(text) == text


def test_source_label(toy_chunks):
    assert source_label(toy_chunks[0]) == "Alfa 10-K FY2024, chunk 3"


def test_render_sources(toy_chunks):
    output = render_sources([1, 0], toy_chunks)
    lines = output.splitlines()
    assert lines[0] == "Fuentes:"
    # El orden de los marcadores sigue el orden del top.
    assert "[1] [Beta 10-K FY2023, chunk 7]" in lines[1]
    assert "[2] [Alfa 10-K FY2024, chunk 3]" in lines[2]
    # El snippet largo se trunca (texto de Alfa ~350 chars, snippet ~200).
    assert len(lines[2]) < 300
