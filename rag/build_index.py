"""python -m rag.build_index [--parse-only]"""

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf

from rag.config import CHUNK_SIZE, INDEX, MODEL, OVERLAP, ROOT, read_jsonl
from rag.ocr import recognize

PARSER_VERSION = f"{pymupdf.__version__}:grid-clean-1"


def chunks(text, size=CHUNK_SIZE, overlap=OVERLAP):
    if not 0 <= overlap < size:
        raise ValueError("Ожидается 0 <= overlap < size")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


def extract_page(page, tessdata):
    text = page.get_text("text", sort=True)
    ocr = sum(c.isalpha() for c in text) < 40
    if ocr:
        if not all((tessdata / f"{lang}.traineddata").exists() for lang in ("rus", "eng")):
            raise RuntimeError("Для сканов нужны index/tessdata/{rus,eng}.traineddata; см. README")
        text = recognize(page, tessdata)
    # Keep the spatial layout: spaces separate financial table columns.
    return text.replace("\x00", "").strip(), ocr


def parse_document(args):
    doc, reports, index = args
    path = reports / doc["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Per-document checkpoint avoids repeating expensive OCR after an interruption.
    checkpoint = index / "parsed" / f"{doc['doc_id']}.json"
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if saved and saved["sha256"] == digest and saved["parser"] == PARSER_VERSION:
        records = saved["pages"]
    else:
        records = []
        with pymupdf.open(path) as pdf:
            if len(pdf) != doc["pages"]:
                raise ValueError(f"Число страниц не совпадает: {doc['doc_id']}")
            for i, page in enumerate(pdf, 1):
                text, ocr = extract_page(page, index / "tessdata")
                records.append({**doc, "page": i, "text": text, "ocr": ocr})
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"sha256": digest, "parser": PARSER_VERSION, "pages": records}, ensure_ascii=False), encoding="utf-8")
    return records, {"doc_id": doc["doc_id"], "sha256": digest}

def parse(reports=ROOT / "data/reports", index=INDEX):
    index.mkdir(parents=True, exist_ok=True)
    pages = []
    manifest = []
    docs = read_jsonl(ROOT / "data/documents.jsonl")
    with ProcessPoolExecutor(max_workers=4) as pool:
        for number, (records, source) in enumerate(pool.map(parse_document, [(d, reports, index) for d in docs]), 1):
            pages.extend(records)
            manifest.append(source)
            print(f"PDF {number}/{len(docs)}: {source['doc_id']} ({sum(p['ocr'] for p in records)} OCR)", flush=True)
    (index / "pages.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages), encoding="utf-8")
    (index / "corpus.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pages


def build(parse_only=False):
    pages = parse()
    if parse_only:
        return
    from fastembed import TextEmbedding
    records = []
    for p in pages:
        for text in chunks(p["text"]):
            compact = re.sub(r"\s+", " ", text).strip()
            if compact:
                records.append({"doc_id": p["doc_id"], "page": p["page"], "text": compact})
    model = TextEmbedding(MODEL, cache_dir=str(INDEX / "models"), threads=4)
    previous = {}
    if all((INDEX / name).exists() for name in ("chunks.jsonl", "vectors.npy", "config.json")):
        config = json.loads((INDEX / "config.json").read_text())
        if config["model"] == MODEL:
            old = read_jsonl(INDEX / "chunks.jsonl")
            old_vectors = np.load(INDEX / "vectors.npy")
            if len(old) != len(old_vectors):
                raise ValueError("Неполный предыдущий индекс")
            previous = {record["text"]: vector for record, vector in zip(old, old_vectors)}
    missing = list(dict.fromkeys(r["text"] for r in records if r["text"] not in previous))
    for start in range(0, len(missing), 128):
        batch = missing[start:start + 128]
        previous.update(zip(batch, model.embed(batch, batch_size=32)))
        print(f"Новые эмбеддинги {min(start + 128, len(missing))}/{len(missing)}", flush=True)
    matrix = np.asarray([previous[r["text"]] for r in records], dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    np.save(INDEX / "vectors.npy", matrix)
    (INDEX / "chunks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    config = {"model": MODEL, "parser": PARSER_VERSION, "chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "documents": len({p['doc_id'] for p in pages}), "pages": len(pages), "chunks": len(records), "ocr_pages": sum(p['ocr'] for p in pages)}
    (INDEX / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parse-only", action="store_true")
    build(parser.parse_args().parse_only)
