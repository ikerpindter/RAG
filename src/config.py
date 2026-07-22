"""Configuración central: rutas, modelos y cliente de OpenAI."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_PATH = DATA_DIR / "len-20241130-10k.htm"
INDEX_DIR = DATA_DIR / "index"
VECTORS_PATH = INDEX_DIR / "vectors.npz"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

# 10-K de Lennar del año fiscal 2024 (cierre 2024-11-30, presentado 2025-01-23).
# URL verificada contra la API de submissions de EDGAR (CIK 0000920760).
FILING_URL = "https://www.sec.gov/Archives/edgar/data/920760/000162828025002404/len-20241130.htm"
# La SEC exige identificarse con un User-Agent de contacto para usar EDGAR.
SEC_USER_AGENT = "Iker Pindter iker.pindter@gmail.com"

EMBEDDING_MODEL = "text-embedding-3-small"
# El modelo de generación más económico vigente (verificado en la doc oficial de
# OpenAI, julio 2026: $0.20/$1.25 por 1M tokens). Si se queda corto, subir a gpt-5.4-mini.
GENERATION_MODEL = "gpt-5.4-nano"

TOP_K = 5

# Modelos para evals (swappables desde aquí). Nano mantiene todo en centavos
# mientras se itera con el borrador; para la corrida oficial del harness
# conviene subir el juez a un modelo más fino (p. ej. gpt-5.6-terra).
EVAL_JUDGE_MODEL = "gpt-5.4-nano"
TESTSET_GENERATOR_MODEL = "gpt-5.4-nano"

# Verificador de citas (swappable): un veredicto binario por afirmación citada;
# nano alcanza y cuesta fracciones de centavo por pregunta.
CITATION_VERIFIER_MODEL = "gpt-5.4-nano"


def get_openai_client():
    """Devuelve un cliente de OpenAI, o termina con un mensaje claro si falta la key."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "ERROR: falta OPENAI_API_KEY.\n"
            "Abre el archivo .env en la raíz del proyecto y pega tu clave:\n"
            "    OPENAI_API_KEY=sk-...\n"
            "(Si .env no existe, copia .env.example como .env.)"
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key)
