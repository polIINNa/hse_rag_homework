import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
load_dotenv(ROOT / ".env")

CHAT_URL = "https://claude.gigaparsers.ru/v1/messages"
EMBEDDING_URL = "https://embeddings.parsers360.ru:10443/embed"
EMBEDDING_MODEL = "parsers360/embed"
VECTORS_FILE = "vectors-parsers360.npz"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_CHARS = 20_000
EMBEDDING_INTERVAL = 1.05
ANSWER_MODEL = os.getenv("RAG_MODEL", "deepseek-flash")
JUDGE_MODEL = "deepseek-flash"
MAX_TOKENS = 1536
JUDGE_MAX_TOKENS = 512
TEMPERATURE = 0
CHAT_RPM = 10
CHAT_WINDOW = 61
HTTP_TIMEOUT = 120
RETRY_COUNT = 3
RETRY_BACKOFF = 2

CHUNK_CHARS = 6000
HEADER_CHARS = 400
OVERLAP_LINES = 4
BATCH_SIZE = 32
PDF_WORKERS = 4
OCR_LANGUAGE = "rus+eng"

TOP_K = 6
PERIOD_BONUS = 0.05
BANKS = {
    "sber": ("сбер",),
    "vtb": ("втб",),
    "domrf": ("дом.рф", "домрф"),
    "mkb": ("мкб", "московск"),
    "tbank": ("т-банк", "т-бан", "тинькофф"),
    "sovkom": ("совком",),
    "rshb": ("россельхоз", "рсхб"),
    "alfa": ("альфа",),
    "gpb": ("газпромбанк",),
}
