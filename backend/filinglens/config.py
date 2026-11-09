"""Central settings, read from the environment or backend/.env."""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

DATA_DIR = Path(os.getenv("FILINGLENS_DATA_DIR", PROJECT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "filinglens.db"

# The SEC asks automated clients to identify themselves. Put your own name and
# contact address in backend/.env if you run this regularly.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FilingLens student-research-project")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")

# USD per million tokens (input, output), used for the per-question cost readout.
MODEL_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
DAILY_QUESTION_LIMIT = int(os.getenv("DAILY_QUESTION_LIMIT", "0"))


# "local" uses any OpenAI-compatible server, such as Ollama. "anthropic" uses Claude.
LLM_PROVIDER = "anthropic" if os.getenv("LLM_PROVIDER") == "anthropic" else "local"
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "filinglens-llm")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "ollama")
LOCAL_LLM_THINKING = os.getenv("LOCAL_LLM_THINKING", "true").lower() == "true"


def using_local_llm() -> bool:
    return LLM_PROVIDER == "local"


def active_model() -> str:
    return LOCAL_LLM_MODEL if using_local_llm() else LLM_MODEL


def _local_llm_reachable() -> bool:
    try:
        response = requests.get(
            f"{LOCAL_LLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"},
            timeout=2,
        )
    except requests.RequestException:
        return False
    return response.ok


def llm_configured() -> bool:
    if using_local_llm():
        return _local_llm_reachable()
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
