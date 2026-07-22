from src.chunking import chunk_text

TEXT = "\n".join(f"linea {i:03d} " + "x" * 80 for i in range(100))


def test_ningun_chunk_excede_el_limite():
    chunks = chunk_text(TEXT, chunk_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_solape_entre_chunks_consecutivos():
    chunks = chunk_text(TEXT, chunk_chars=500, overlap_chars=100)
    for previous, current in zip(chunks, chunks[1:]):
        # El chunk siguiente arranca con la cola del anterior.
        assert current.splitlines()[0] == previous.splitlines()[-1]


def test_determinismo():
    assert chunk_text(TEXT) == chunk_text(TEXT)


def test_linea_gigante_se_parte_en_seco():
    chunks = chunk_text("y" * 9000, chunk_chars=4000, overlap_chars=600)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 4000 for chunk in chunks)
