"""Скачать исходный архив, проверить SHA-256 и распаковать только ожидаемые PDF."""
import argparse
import hashlib
import zipfile
from pathlib import Path

import requests

from rag.common import ROOT, read_jsonl

URL = "https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip"


def prepare_ocr(root):
    # Только данные языков, не установщик и не внешний исполняемый файл.
    expected = {"eng": "bbef4675053b5b468cdb477053e28b1c698ba08e",
                "rus": "b146cb2263acbc6383f8e92ea0ce759537687bb8"}
    destination = Path(root) / "data/tessdata"
    destination.mkdir(parents=True, exist_ok=True)
    for language, sha in expected.items():
        path = destination / f"{language}.traineddata"
        if not path.exists():
            url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{language}.traineddata"
            response = requests.get(url, timeout=(20, 90))
            response.raise_for_status()
            path.write_bytes(response.content)
        b = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
        if actual != sha:
            raise ValueError(f"Контрольная сумма OCR-данных {language} не совпала")


def prepare(archive=None, root=ROOT):
    root = Path(root)
    archive = Path(archive) if archive else root / "data/reports.zip"
    pointer = (root / "data/reports.lfs").read_text(encoding="utf-8")
    digest = next(line.split("sha256:", 1)[1] for line in pointer.splitlines() if line.startswith("oid "))
    size = int(next(line.split()[1] for line in pointer.splitlines() if line.startswith("size ")))
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".part")
        print("Скачивание PDF-корпуса...", flush=True)
        with requests.get(URL, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as target:
                for block in response.iter_content(1024 * 1024):
                    target.write(block)
        temporary.replace(archive)
    h = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    if archive.stat().st_size != size or h.hexdigest() != digest:
        raise ValueError("Архив не совпал с исходным LFS-объектом преподавателя")
    destination = (root / "data/reports").resolve()
    documents = read_jsonl(root / "data/documents.jsonl")
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        for doc in documents:
            name = doc["path"]
            choices = [p for p in (name, "reports/" + name, "data/reports/" + name) if p in names]
            if len(choices) != 1:
                raise ValueError(f"В архиве не найден однозначный PDF: {name}")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Путь PDF выходит за папку корпуса")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(choices[0]))
    print(f"Готово: {len(documents)} PDF, SHA-256 подтверждён", flush=True)
    prepare_ocr(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    prepare(args.archive)
