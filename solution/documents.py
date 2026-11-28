"""Пути проекта и метаданные отчётов из data/documents.jsonl."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = ROOT / "outputs"


def load_documents() -> list[dict]:
    """doc_id устроен как <банк>_<год>_<annual|6m>, из него берутся банк, год и период."""
    documents = []
    with open(DATA_DIR / "documents.jsonl", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            bank, year, period = item["doc_id"].split("_")
            documents.append({
                "doc_id": item["doc_id"],
                "bank": bank,
                "organization": item["organization"],
                "year": int(year),
                "period": period,
                "path": REPORTS_DIR / item["path"],
                "pages": item["pages"],
            })
    return documents
