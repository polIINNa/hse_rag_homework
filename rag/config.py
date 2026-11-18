"""Пути, модели и параметры RAG. Всё, что зависит от ключей, читается из окружения (.env)."""

import os
from pathlib import Path

try:  # .env в корне репозитория (необязательно)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
PAGES_PATH = DATA_DIR / "pages.jsonl.gz"          # результат парсинга: одна строка = одна страница PDF
INDEX_DIR = ROOT / "index"                        # эмбеддинги страниц (в .gitignore)

# --- Эмбеддинги deepvk/USER-bge-m3 (bge-m3, дообученная на русском, вектор 1024)
# course — веб-сервис курса (нужен EMB_KEY; выбирается сам, если ключ задан)
# local  — та же модель с Hugging Face через sentence-transformers (без ключа; лучше на GPU)
EMB_KEY = os.environ.get("EMB_KEY", "")
EMBEDDER = os.environ.get("EMBEDDER") or ("course" if EMB_KEY else "local")
EMB_MODEL = os.environ.get("EMB_MODEL", "deepvk/USER-bge-m3")
EMB_MAX_TOKENS = 4096                              # самая длинная страница ~8 000 символов — влезает
EMB_URL = os.environ.get("EMB_URL", "https://embeddings.parsers360.ru:10443")
EMB_MAX_CHARS = 20000                              # лимит сервиса; самая длинная страница ~8 000 — режется не будет

# --- LLM: любой OpenAI-совместимый API
# По умолчанию — бесплатная локальная модель через Ollama (http://localhost:11434), ключ не нужен.
# Если задан OPENROUTER_API_KEY — OpenRouter, как в шаблоне. Любой другой сервис: LLM_BASE_URL + LLM_API_KEY.
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OLLAMA_URL = "http://localhost:11434/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_BASE_URL = (os.environ.get("LLM_BASE_URL") or (OPENROUTER_URL if OPENROUTER_API_KEY else OLLAMA_URL)).rstrip("/")
LLM_ROOT_URL = LLM_BASE_URL[:-3] if LLM_BASE_URL.endswith("/v1") else LLM_BASE_URL
LLM_API_FORMAT = os.environ.get("LLM_API_FORMAT", "openai")            # openai (/v1/chat/completions) | anthropic (/v1/messages)
IS_OPENROUTER = "openrouter.ai" in LLM_BASE_URL
IS_LOCAL = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL
LLM_API_KEY = os.environ.get("LLM_API_KEY") or OPENROUTER_API_KEY or ("ollama" if IS_LOCAL else "")
OLLAMA_BASE_MODEL = os.environ.get("OLLAMA_BASE_MODEL", "qwen2.5:7b")   # скачивается один раз, ~4,7 ГБ
OLLAMA_CONTEXT = 24576                                                  # окно контекста: 8 страниц + доп. поиски
LLM_MODEL = os.environ.get("LLM_MODEL") or ("openai/gpt-5-mini" if IS_OPENROUTER else "rag-qwen" if IS_LOCAL else "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL") or ("google/gemini-3.8-flash" if IS_OPENROUTER else LLM_MODEL)
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")           # только для OpenRouter
LLM_TIMEOUT = 600 if IS_LOCAL else 120                                  # локальная модель на ноутбуке медленнее
# Лимит запросов в минуту к LLM (генерация + агент + судья вместе). У ключей модели курса — 10/мин;
# запросы идут не чаще лимита, поэтому агент не упирается в 429. 0 — без ограничения.
LLM_RPM = float(os.environ.get("LLM_RPM") or (10 if "gigaparsers" in LLM_BASE_URL else 0))

# --- Поиск
RAG_MODE = os.environ.get("RAG_MODE", "agentic")   # agentic | simple
TOP_K = int(os.environ.get("TOP_K", "8"))          # сколько страниц первичного поиска отдаём LLM
RRF_K = 60                                         # константа Reciprocal Rank Fusion
MAX_AGENT_SEARCHES = 3                             # лимит дополнительных поисков агента


def llm_api_key() -> str:
    if not LLM_API_KEY:
        raise RuntimeError("Нет ключа LLM: задайте LLM_API_KEY (или OPENROUTER_API_KEY) в .env, см. README.")
    return LLM_API_KEY


def require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Не задана переменная окружения {name} (см. README, раздел «Запуск»).")
    return value
