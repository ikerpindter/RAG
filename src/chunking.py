"""Parte el texto extraído en chunks con solape."""

# ¿Por qué ~4000 caracteres (~1000 tokens) con solape de ~600 (~15%)?
# - Un 10-K alterna prosa larga y tablas; ~1000 tokens alcanzan para que una idea
#   completa (un párrafo largo, o una tabla corta con su encabezado) quede entera
#   en un solo chunk, y así el dato se recupera junto con su contexto.
# - Chunks más grandes diluyen el embedding (mezclan temas) y encarecen cada
#   prompt; más chicos parten tablas y párrafos por la mitad. Para filings
#   financieros el rango típico que funciona bien es 800-1200 tokens.
# - El solape evita que un dato que cae justo en la frontera entre dos chunks
#   quede huérfano de su contexto en ambos lados.
CHUNK_CHARS = 4000
OVERLAP_CHARS = 600


def chunk_text(
    text: str, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[str]:
    # Unidad mínima: línea no vacía (la extracción deja un bloque/párrafo por línea).
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Una línea gigante (p. ej. una tabla enorme) se parte en seco.
        while len(line) > chunk_chars:
            units.append(line[:chunk_chars])
            line = line[chunk_chars - overlap_chars :]
        units.append(line)

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) + 1 > chunk_chars:
            chunks.append("\n".join(current))
            # El nuevo chunk arranca con la cola del anterior como solape.
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) + 1 > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            current = tail
            size = tail_size
        current.append(unit)
        size += len(unit) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks
