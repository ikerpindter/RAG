"""Análisis de la consulta: qué empresas y años fiscales menciona, y si compara.

Una llamada corta a un modelo barato que devuelve JSON estricto, validado
contra el catálogo real del corpus (FILINGS). Sin fallbacks silenciosos: JSON
inválido o valores fuera de catálogo -> RuntimeError con el output crudo.
Listas vacías son un resultado VÁLIDO y significan "sin filtro" (la red de
seguridad legítima: retrieval normal sobre todo el corpus).
"""

import json

from .config import FILINGS, QUERY_ANALYZER_MODEL

KNOWN_COMPANIES = sorted({doc["company"] for doc in FILINGS})
KNOWN_YEARS = sorted({doc["fiscal_year"] for doc in FILINGS})

ANALYZER_INSTRUCTIONS = (
    "Analizas preguntas sobre reportes 10-K. Devuelve SOLO un objeto JSON, sin "
    "texto adicional ni markdown, con exactamente esta forma: "
    '{"companies": [...], "fiscal_years": [...], "comparative": false}. '
    f"companies: subconjunto de {KNOWN_COMPANIES} que la pregunta mencione o "
    "aluda (vacía si no menciona ninguna del catálogo). "
    f"fiscal_years: subconjunto de {KNOWN_YEARS} que la pregunta refiera, como "
    "números (vacía si no especifica año). "
    "comparative: true solo si la pregunta pide comparar entre empresas."
)


def _strip_code_fence(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def analyze_query(question: str, client) -> dict:
    """Devuelve {"companies": [...], "fiscal_years": [...], "comparative": bool}."""
    response = client.responses.create(
        model=QUERY_ANALYZER_MODEL,
        instructions=ANALYZER_INSTRUCTIONS,
        input=question,
    )
    raw = _strip_code_fence(response.output_text.strip())
    try:
        data = json.loads(raw)
        companies = data["companies"]
        years = [int(y) for y in data["fiscal_years"]]
        comparative = data["comparative"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"El analizador ({QUERY_ANALYZER_MODEL}) devolvió un resultado "
            f"inválido: {response.output_text!r}"
        ) from e
    if not isinstance(companies, list) or not isinstance(comparative, bool):
        raise RuntimeError(
            f"El analizador devolvió tipos inesperados: {response.output_text!r}"
        )
    unknown = [c for c in companies if c not in KNOWN_COMPANIES] + [
        y for y in years if y not in KNOWN_YEARS
    ]
    if unknown:
        raise RuntimeError(
            f"El analizador devolvió valores fuera del catálogo {unknown}: "
            f"{response.output_text!r}"
        )
    return {"companies": companies, "fiscal_years": years, "comparative": comparative}
