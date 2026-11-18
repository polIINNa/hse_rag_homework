"""Извлечение страниц. Строки таблиц собираются по координатам слов."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess

import pymupdf

from .config import CACHE, DATA, INDEX, ROOT


def layout_text(boxes):
    rows = []
    for box in sorted(boxes, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        center = (box[1] + box[3]) / 2
        if rows and abs(center - rows[-1][0]) <= max(2.5, (box[3] - box[1]) * .30):
            rows[-1][1].append(box)
        else:
            rows.append((center, [box]))
    result = []
    for _, words in rows:
        line, previous = "", None
        for x0, y0, x1, y1, text in sorted(words, key=lambda w: w[0]):
            if previous is not None:
                gap = x0 - previous
                line += " " if gap < 10 else " " * min(24, max(3, round(gap / 4)))
            line += text
            previous = x1
        if line.strip():
            result.append(line.rstrip())
    return "\n".join(result)


def ocr_page(pdf_path, number, doc_id, backend):
    cache = CACHE / "ocr" / doc_id
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{number:03d}.json"
    if not dest.exists():
        if backend == "vision":
            subprocess.run([str(ROOT / "tmp" / "vision_ocr"), str(pdf_path), str(number), str(dest)], check=True)
        else:
            with pymupdf.open(pdf_path) as pdf:
                page = pdf[number - 1]
                tp = page.get_textpage_ocr(language="rus+eng", dpi=216, full=True)
                words = page.get_text("words", textpage=tp)
                dest.write_text(json.dumps([dict(zip(("x0", "y0", "x1", "y1", "text"), w[:5])) for w in words], ensure_ascii=False))
    objects = json.loads(dest.read_text())
    return layout_text([(w["x0"], w["y0"], w["x1"], w["y1"], w["text"]) for w in objects])


def parse_document(document, backend):
    path = DATA / "reports" / document["path"]
    pages, scanned = [], 0
    with pymupdf.open(path) as pdf:
        if len(pdf) != document["pages"]:
            raise ValueError(f"Число страниц не совпало: {path}")
        for i, page in enumerate(pdf):
            words = page.get_text("words")
            raw = " ".join(w[4] for w in words)
            needs_ocr = len(raw.strip()) < 80 or raw.count("\ufffd") > len(raw) * .05
            if needs_ocr:
                text = ocr_page(path, i + 1, document["doc_id"], backend)
                scanned += 1
            else:
                text = layout_text([w[:5] for w in words])
            pages.append({"doc_id": document["doc_id"], "page": i + 1,
                          "organization": document["organization"], "path": document["path"],
                          "text": text, "extraction": backend if needs_ocr else "text"})
    output = INDEX / "documents" / (document["doc_id"] + ".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages))
    return document["doc_id"], len(pages), scanned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", choices=("vision", "tesseract"), default="vision" if os.uname().sysname == "Darwin" else "tesseract")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    INDEX.mkdir(exist_ok=True)
    if args.ocr == "vision":
        binary = ROOT / "tmp" / "vision_ocr"
        binary.parent.mkdir(exist_ok=True)
        source = ROOT / "scripts" / "vision_ocr.swift"
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(["swiftc", "-O", "-module-cache-path", str(CACHE / "swift"), str(source), "-o", str(binary)], check=True)
    documents = [json.loads(line) for line in (DATA / "documents.jsonl").read_text().splitlines()]
    pending = [d for d in documents if args.force or not (INDEX / "documents" / (d["doc_id"] + ".jsonl")).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(parse_document, d, args.ocr) for d in pending]
        for future in as_completed(tasks):
            doc, n, scanned = future.result()
            print(f"{doc}: {n} страниц, OCR {scanned}", flush=True)
    with (INDEX / "pages.jsonl").open("w") as out:
        for d in documents:
            out.write((INDEX / "documents" / (d["doc_id"] + ".jsonl")).read_text())
    print("Готово:", INDEX / "pages.jsonl", flush=True)


if __name__ == "__main__":
    main()
