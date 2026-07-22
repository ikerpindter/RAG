"""Verificación de citas: ¿el chunk citado realmente respalda cada afirmación?

La respuesta se parte en afirmaciones de forma determinista (por oración); cada
afirmación hereda los marcadores [n] que contiene. Por cada afirmación citada,
un LLM barato da un veredicto binario contra el texto de los chunks citados.
Las oraciones sin marcador se reportan como "sin cita": visibles, no ignoradas.
"""

import re

from .citations import strip_citations
from .config import CITATION_VERIFIER_MODEL

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+")
_MARKER_NUM_RE = re.compile(r"\[(\d+)\]")
# Marcadores que quedaron DESPUÉS del punto (". [3]") se recolocan antes de él
# para que el separador de oraciones no los deje huérfanos de su afirmación.
_TRAILING_MARKER_RE = re.compile(r"([.!?:])((?:\s*\[\d+\])+)")
# Oraciones más cortas que esto son conectores/títulos, no afirmaciones.
MIN_CLAIM_CHARS = 25

VERIFIER_INSTRUCTIONS = (
    "Eres un verificador estricto de afirmaciones contra documentos. Se te da "
    "un EXTRACTO de un reporte 10-K y una AFIRMACIÓN. Responde exactamente "
    "'SI' si todos los datos de la afirmación aparecen en el extracto o se "
    "derivan directamente de él, o 'NO' en caso contrario. Responde únicamente "
    "SI o NO, nada más."
)


def extract_claims(answer: str) -> list[dict]:
    """Parte la respuesta en afirmaciones con sus marcadores: {text, markers}."""
    claims = []
    for line in answer.splitlines():
        line = line.strip().lstrip("•-* ").strip()
        if not line:
            continue
        line = _TRAILING_MARKER_RE.sub(lambda m: m.group(2) + m.group(1), line)
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            if len(sentence) < MIN_CLAIM_CHARS:
                continue
            markers = sorted({int(m) for m in _MARKER_NUM_RE.findall(sentence)})
            claims.append({"text": sentence, "markers": markers})
    return claims


def verify_answer(answer: str, top_ids: list[int], chunks: list[str], client) -> dict:
    """Verifica cada afirmación citada contra sus chunks. Devuelve veredictos y resumen."""
    results = []
    for claim in extract_claims(answer):
        markers = claim["markers"]
        if not markers:
            results.append({**claim, "verdict": "sin_cita"})
            continue
        invalid = [m for m in markers if not (1 <= m <= len(top_ids))]
        if invalid:
            # Marcador fuera de rango = el modelo citó un extracto inexistente.
            results.append({**claim, "verdict": "cita_invalida"})
            continue
        cited_text = "\n\n".join(chunks[top_ids[m - 1]] for m in markers)
        response = client.responses.create(
            model=CITATION_VERIFIER_MODEL,
            instructions=VERIFIER_INSTRUCTIONS,
            input=(
                f"EXTRACTO:\n{cited_text}\n\n"
                f"AFIRMACIÓN: {strip_citations(claim['text'])}"
            ),
        )
        raw = response.output_text.strip().upper().rstrip(".")
        if raw in ("SI", "SÍ"):
            verdict = "respaldada"
        elif raw == "NO":
            verdict = "no_respaldada"
        else:
            raise RuntimeError(
                f"Veredicto inesperado del verificador ({CITATION_VERIFIER_MODEL}): {raw!r}"
            )
        results.append({**claim, "verdict": verdict})

    cited = [r for r in results if r["verdict"] in ("respaldada", "no_respaldada")]
    supported = [r for r in cited if r["verdict"] == "respaldada"]
    return {"claims": results, "supported": len(supported), "cited": len(cited)}


def print_verification(verification: dict) -> None:
    symbols = {
        "respaldada": "✓",
        "no_respaldada": "✗",
        "sin_cita": "•",
        "cita_invalida": "!",
    }
    print("Verificación de citas:")
    for claim in verification["claims"]:
        markers = "".join(f"[{m}]" for m in claim["markers"]) or "[—]"
        label = {
            "respaldada": "respaldada",
            "no_respaldada": "NO respaldada",
            "sin_cita": "sin cita",
            "cita_invalida": "cita inválida",
        }[claim["verdict"]]
        text = strip_citations(claim["text"])
        if len(text) > 90:
            text = text[:87] + "..."
        print(f"  {symbols[claim['verdict']]} {markers} {label}: {text}")
    print(
        f"Resumen: {verification['supported']}/{verification['cited']} "
        "afirmaciones citadas verificadas como respaldadas"
    )
