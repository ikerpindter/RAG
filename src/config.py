"""Configuración central: rutas, modelos y cliente de OpenAI."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

# La SEC exige identificarse con un User-Agent de contacto para usar EDGAR.
SEC_USER_AGENT = "Iker Pindter iker.pindter@gmail.com"

# Corpus: 10-Ks localizados vía la API de submissions de EDGAR (CIK Lennar
# 0000920760, D.R. Horton 0000882184; el de Horton resuelto desde el archivo
# oficial de tickers de la SEC) y verificados con HTTP 200. Ojo con los cierres
# fiscales distintos: Lennar 30-nov, D.R. Horton 30-sep.
FILINGS = [
    {
        "doc_id": "lennar-fy2024",
        "company": "Lennar",
        "fiscal_year": 2024,
        "url": "https://www.sec.gov/Archives/edgar/data/920760/000162828025002404/len-20241130.htm",
        "filename": "len-20241130-10k.htm",
    },
    {
        "doc_id": "lennar-fy2023",
        "company": "Lennar",
        "fiscal_year": 2023,
        "url": "https://www.sec.gov/Archives/edgar/data/920760/000162828024002371/len-20231130.htm",
        "filename": "len-20231130-10k.htm",
    },
    {
        "doc_id": "dhi-fy2024",
        "company": "D.R. Horton",
        "fiscal_year": 2024,
        "url": "https://www.sec.gov/Archives/edgar/data/882184/000088218424000057/dhi-20240930.htm",
        "filename": "dhi-20240930-10k.htm",
    },
    {
        "doc_id": "dhi-fy2023",
        "company": "D.R. Horton",
        "fiscal_year": 2023,
        "url": "https://www.sec.gov/Archives/edgar/data/882184/000088218423000115/dhi-20230930.htm",
        "filename": "dhi-20230930-10k.htm",
    },
]

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
