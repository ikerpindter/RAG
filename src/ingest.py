"""Ingesta end-to-end: descarga el 10-K, extrae texto, trocea, embebe y guarda el índice."""

import re
import sys

import httpx
from bs4 import BeautifulSoup

from . import store
from .chunking import chunk_text
from .config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    FILING_URL,
    INDEX_DIR,
    RAW_HTML_PATH,
    SEC_USER_AGENT,
    get_openai_client,
)
from .embeddings import embed_texts

# Un 10-K real tiene cientos de páginas; menos texto que esto indica que la
# descarga o la extracción salió mal y es mejor pararse que indexar basura.
MIN_TEXT_CHARS = 50_000


def download() -> str:
    if RAW_HTML_PATH.exists():
        print(f"HTML ya descargado, se reutiliza: {RAW_HTML_PATH.name}")
        return RAW_HTML_PATH.read_text(encoding="utf-8", errors="ignore")
    print(f"Descargando {FILING_URL}")
    response = httpx.get(
        FILING_URL,
        headers={"User-Agent": SEC_USER_AGENT},
        timeout=120,
        follow_redirects=True,
    )
    response.raise_for_status()
    DATA_DIR.mkdir(exist_ok=True)
    RAW_HTML_PATH.write_bytes(response.content)
    print(f"Guardado en {RAW_HTML_PATH} ({len(response.content) / 1e6:.1f} MB)")
    return response.text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # El inline XBRL mete bloques ocultos de metadatos que no son parte del
    # documento legible.
    for hidden in soup.find_all(style=re.compile(r"display:\s*none")):
        hidden.decompose()
    # Tablas de forma pragmática: cada fila -> una línea "celda | celda | celda".
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")
    # Un salto de línea al final de cada bloque para conservar los párrafos;
    # sin esto, get_text pegaría párrafos vecinos en una sola línea.
    for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        block.append("\n")
    text = soup.get_text(" ")
    lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def main() -> None:
    client = get_openai_client()  # valida la API key antes de trabajar
    html = download()
    text = extract_text(html)
    print(f"Texto extraído: {len(text):,} caracteres")
    if len(text) < MIN_TEXT_CHARS:
        sys.exit(
            "ERROR: el texto extraído es sospechosamente corto para un 10-K; "
            "revisa el HTML en data/ antes de continuar."
        )
    chunks = chunk_text(text)
    print(f"Chunks: {len(chunks)}")
    vectors = embed_texts(client, chunks)
    store.save(
        vectors,
        chunks,
        meta={"source_url": FILING_URL, "embedding_model": EMBEDDING_MODEL},
    )
    print(
        f"Índice guardado en {INDEX_DIR}. Ya puedes preguntar con:\n"
        '    uv run python -m src.query "tu pregunta"'
    )


if __name__ == "__main__":
    main()
