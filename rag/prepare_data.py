"""Скачать и распаковать корпус задания и открытые OCR-модели."""

import zipfile
from pathlib import Path

import requests

from rag.config import INDEX, ROOT, read_jsonl


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temporary.replace(target)


def prepare():
    archive = ROOT / "data/reports.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        download("https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip", archive)
    destination = ROOT / "data/reports"
    documents = read_jsonl(ROOT / "data/documents.jsonl")
    if any(not (destination / doc["path"]).is_file() for doc in documents):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                path = (destination / member.filename).resolve()
                if not path.is_relative_to(destination.resolve()):
                    raise ValueError("Недопустимый путь внутри ZIP")
            source.extractall(destination)
    for lang in ("rus", "eng"):
        target = INDEX / "tessdata" / f"{lang}.traineddata"
        if not target.exists():
            download(f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{lang}.traineddata", target)
    print(f"Корпус: {len(documents)} PDF; OCR: rus+eng")


if __name__ == "__main__":
    prepare()
