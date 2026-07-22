"""Citations: instrucción de citado, sección de fuentes y limpieza de marcadores.

Los extractos van numerados en el prompt ([Extracto 1], [Extracto 2], ...), así
que el marcador [n] de la respuesta mapea directo al extracto n del top-k. La
sección de fuentes la construye este código desde los metadatos del retrieval,
no el modelo: no puede citar fuentes inventadas.
"""

import re

# Se concatena a las INSTRUCTIONS base de la generación.
CITATION_INSTRUCTIONS = (
    "Cita tus fuentes: inmediatamente después de cada afirmación, agrega el "
    "marcador del extracto del que salió el dato con el formato [1], [2], etc., "
    "usando el número del extracto correspondiente. Toda afirmación con datos "
    "debe llevar al menos un marcador. No agregues una lista de fuentes al "
    "final; solo los marcadores en línea."
)

# Espacio opcional + [dígitos]: quita el marcador sin dejar dobles espacios.
_MARKER_RE = re.compile(r"\s?\[\d+\]")

SNIPPET_CHARS = 200


def strip_citations(text: str) -> str:
    """Quita los marcadores [n]; deja la respuesta limpia para el eval."""
    return _MARKER_RE.sub("", text)


def render_sources(top_ids: list[int], chunks: list[str]) -> str:
    """Sección de fuentes: número de marcador, fragmento corto y id del chunk."""
    lines = ["Fuentes:"]
    for rank, chunk_id in enumerate(top_ids, start=1):
        snippet = " ".join(chunks[chunk_id][:SNIPPET_CHARS].split())
        lines.append(f"  [{rank}] (chunk {chunk_id}) {snippet}…")
    return "\n".join(lines)
