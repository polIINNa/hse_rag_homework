from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
INDEX = ROOT / "index"
CACHE = ROOT / ".cache"


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Заполните {name} в .env")
    return value

