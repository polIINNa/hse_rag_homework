"""Извлечение текста из PDF постранично.

Из корня репозитория: python -m rag.parse

Текстовый слой берём через `pdftotext -layout` (poppler): он сохраняет колонки таблиц.
Страницы без текстового слоя (сканы, картинки) распознаём tesseract (rus+eng).
Результат — data/parsed/pages.jsonl.gz: одна строка на страницу PDF с doc_id и номером страницы с 1.
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "data" / "reports"
DOCUMENTS = ROOT / "data" / "documents.jsonl"
PARSED = ROOT / "data" / "parsed" / "pages.jsonl.gz"
TESSDATA = ROOT / "tessdata"

MIN_TEXT_CHARS = 300  # меньше — считаем страницу сканом или картинкой


def load_documents():
    with DOCUMENTS.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def compact(text):
    """Схлопнуть длинные пробелы: выравнивание колонок остаётся, токенов меньше."""
    lines = [re.sub(r" {4,}", "   ", line.rstrip()) for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def text_layer(pdf_path):
    out = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True,
                         check=True).stdout
    return out.split("\f")


def ocr_page(task):
    pdf_path, page = task
    env = dict(os.environ)
    if TESSDATA.exists():
        env["TESSDATA_PREFIX"] = str(TESSDATA)
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "page"
        subprocess.run(["pdftoppm", "-r", "300", "-gray", "-png", "-f", str(page), "-l", str(page),
                        "-singlefile", str(pdf_path), str(image)], check=True)
        out = subprocess.run(["tesseract", str(image) + ".png", "-", "-l", "rus+eng", "--psm", "6",
                              "-c", "preserve_interword_spaces=1"],
                             capture_output=True, text=True, check=True, env=env).stdout
    return out


def parse(workers):
    documents = load_documents()
    pages = []
    for document in documents:
        layer = text_layer(REPORTS / document["path"])
        for number in range(1, document["pages"] + 1):
            text = layer[number - 1] if number <= len(layer) else ""
            pages.append({"doc_id": document["doc_id"], "page": number, "text": text, "source": "text",
                          "_path": str(REPORTS / document["path"])})

    to_ocr = [i for i, page in enumerate(pages) if len(page["text"].strip()) < MIN_TEXT_CHARS]
    print(f"Страниц: {len(pages)}, на OCR: {len(to_ocr)}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [(pages[i]["_path"], pages[i]["page"]) for i in to_ocr]
        for done, (i, text) in enumerate(zip(to_ocr, pool.map(ocr_page, tasks, chunksize=2)), 1):
            if len(text.strip()) > len(pages[i]["text"].strip()):
                pages[i]["text"], pages[i]["source"] = text, "ocr"
            if done % 50 == 0:
                print(f"  OCR {done}/{len(to_ocr)}")

    PARSED.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(PARSED, "wt", encoding="utf-8") as target:
        for page in pages:
            page.pop("_path")
            page["text"] = compact(page["text"])
            target.write(json.dumps(page, ensure_ascii=False) + "\n")
    print(f"Сохранено: {PARSED}")


def load_pages():
    with gzip.open(PARSED, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parse(parser.parse_args().workers)
