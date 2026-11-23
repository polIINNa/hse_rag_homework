import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pymupdf

from rag import config
from rag.models import embed


def read_pages(path, metadata):
    with pymupdf.open(path) as pdf:
        pages = pdf.to_markdown(page_chunks=True, ocr_language=config.OCR_LANGUAGE)
    return [
        {
            "doc_id": metadata["doc_id"],
            "organization": metadata["organization"],
            "page": page["metadata"]["page_number"],
            "text": page["text"].strip(),
        }
        for page in pages
        if page["text"].strip()
    ]


def split_page(page):
    header = page["text"].split("\n\n", 1)[0][: config.HEADER_CHARS]
    chunks, lines, size = [], [], 0
    for line in page["text"].splitlines(keepends=True):
        if lines and size + len(line) > config.CHUNK_CHARS:
            prefix = header + "\n" if chunks else ""
            chunks.append({**page, "text": prefix + "".join(lines)})
            lines = lines[-config.OVERLAP_LINES :]
            size = sum(map(len, lines))
        lines.append(line)
        size += len(line)
    if lines:
        prefix = header + "\n" if chunks else ""
        chunks.append({**page, "text": prefix + "".join(lines)})
    return chunks


def parse_document(document, root=config.ROOT):
    pages = read_pages(root / "data/reports" / document["path"], document)
    print(f"PDF: {document['doc_id']}, {len(pages)} страниц", flush=True)
    return [chunk for page in pages for chunk in split_page(page)]


def save_atomic(path, save):
    with NamedTemporaryFile(dir=path.parent, delete=False) as pending:
        temporary = Path(pending.name)
    try:
        with temporary.open("wb") as target:
            save(target)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build(root=config.ROOT):
    index = root / "index"
    index.mkdir(exist_ok=True)
    chunks_path = index / "chunks.json"
    if not chunks_path.exists():
        documents = [
            json.loads(line)
            for line in (root / "data/documents.jsonl").read_text().splitlines()
        ]
        with ProcessPoolExecutor(
            max_workers=min(config.PDF_WORKERS, len(documents))
        ) as workers:
            chunks = [
                chunk
                for document in workers.map(
                    partial(parse_document, root=root), documents
                )
                for chunk in document
            ]
        data = json.dumps(chunks, ensure_ascii=False).encode()
        save_atomic(chunks_path, lambda target: target.write(data))
    chunks_data = chunks_path.read_bytes()
    chunks = json.loads(chunks_data)
    if not chunks:
        raise ValueError("В индексе нет ни одного фрагмента PDF.")
    cache = index / "embeddings-parsers360"
    cache.mkdir(exist_ok=True)
    vectors = []
    for start in range(0, len(chunks), config.BATCH_SIZE):
        batch = chunks[start : start + config.BATCH_SIZE]
        texts = [f"{c['organization']} {c['doc_id']}\n{c['text']}" for c in batch]
        key = hashlib.sha256(
            (config.EMBEDDING_MODEL + json.dumps(texts)).encode()
        ).hexdigest()
        batch_path = cache / f"{key}.npy"
        if not batch_path.exists():
            batch_vectors = embed(texts)
            save_atomic(batch_path, partial(np.save, arr=batch_vectors))
        vectors.append(np.load(batch_path))
        print(f"Эмбеддинги: {start + len(batch)}/{len(chunks)}", flush=True)
    save_atomic(
        index / config.VECTORS_FILE,
        partial(
            np.savez_compressed,
            vectors=np.concatenate(vectors),
            model=config.EMBEDDING_MODEL,
            chunks_hash=hashlib.sha256(chunks_data).hexdigest(),
        ),
    )


if __name__ == "__main__":
    build()
