import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "index" / "pages.jsonl"
EMBED_URL = "https://embeddings.parsers360.ru:10443/embed"
DIM = 1024
MAX_CHARS = 20_000


class Embedder:
    def __init__(self, key):
        self.key = key
        self.next_request = 0.0

    def embed(self, text):
        if not 0 < len(text) <= MAX_CHARS:
            raise ValueError(f"Expected 1–{MAX_CHARS} characters")
        request = Request(
            EMBED_URL,
            data=text.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
        for attempt in range(5):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + 1.05
            try:
                with urlopen(request, timeout=30) as response:
                    result = json.load(response)
                break
            except HTTPError as error:
                if error.code not in (429, 502, 503, 504) or attempt == 4:
                    raise
                try:
                    delay = max(2**attempt, float(error.headers.get("Retry-After", "0")))
                except ValueError:
                    delay = 2**attempt
            except (URLError, TimeoutError, ConnectionError) as error:
                reason = error.reason if isinstance(error, URLError) else error
                if not isinstance(reason, (TimeoutError, ConnectionError)):
                    raise
                if attempt == 4:
                    raise RuntimeError(f"Could not connect to the embedder at {EMBED_URL} after 5 attempts") from error
                delay = 2**attempt
            print(f"Embedder unavailable, retrying in {delay:g}s", flush=True)
            self.next_request = max(self.next_request, time.monotonic() + delay)

        vector = np.asarray(result["embedding"], dtype="<f4")
        if result["dim"] != DIM or vector.shape != (DIM,) or not np.isfinite(vector).all():
            raise ValueError("Unexpected embedding returned by the course service")
        return vector


def iter_chunks(data, doc_id, tessdata):
    import pymupdf

    documents = [json.loads(line) for line in (data / "documents.jsonl").read_text().splitlines()]
    selected = [doc for doc in documents if doc_id is None or doc["doc_id"] == doc_id]
    if not selected:
        raise ValueError(f"Unknown document: {doc_id}")

    for metadata in selected:
        name = metadata["doc_id"]
        prefix = f"{metadata['organization']} | {name}\n"
        chunk_chars = MAX_CHARS - len(prefix)
        with pymupdf.open(data / "reports" / metadata["path"]) as pdf:
            if len(pdf) != metadata["pages"]:
                raise ValueError(f"Page count differs from manifest for {name}")
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text(sort=True).strip()
                if not text and tessdata is not None:
                    recognized = page.get_textpage_ocr(
                        language="rus+eng", dpi=200, full=True, tessdata=str(tessdata)
                    )
                    text = page.get_text(textpage=recognized, sort=True).strip()
                for chunk, start in enumerate(range(0, len(text), chunk_chars)):
                    yield name, page_num, chunk, prefix + text[start : start + chunk_chars]


def read_records(stream, *, repair_tail=False):
    records = {}
    while True:
        offset = stream.tell()
        line = stream.readline()
        if line and not line.startswith(b"{"):
            raise ValueError("Expected a JSONL index. Choose a new path for older index formats.")
        if not line.endswith(b"\n"):
            if line and repair_tail:
                stream.truncate(offset)
            return records
        row = json.loads(line)
        key = row["doc_id"], row["page"], row["chunk"]
        records[key] = (row["text"], np.asarray(row["embedding"], dtype=np.float32))


def load_index(index):
    with index.open("rb") as stream:
        records = read_records(stream)
    if not records:
        raise ValueError(f"No embeddings in {index}")
    keys = sorted(records)
    chunks = [(name, page, records[name, page, chunk][0]) for name, page, chunk in keys]
    matrix = np.stack([records[key][1] for key in keys])
    return chunks, matrix


def build(index, data, doc_id, embed, tessdata=None):
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        cached = read_records(stream, repair_tail=True)
        added = reused = 0
        for name, page, chunk, text in iter_chunks(data, doc_id, tessdata):
            key = (name, page, chunk)
            if key in cached and cached[key][0] == text:
                reused += 1
                continue
            row = {
                "doc_id": name,
                "page": page,
                "chunk": chunk,
                "text": text,
                "embedding": embed(text).tolist(),
            }
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            stream.write((encoded + "\n").encode("utf-8"))
            stream.flush()
            added += 1
            if added % 10 == 0:
                print(f"Saved {added} new embeddings", flush=True)
    print(f"Saved {added} new embeddings, reused {reused}")


def search(index, question, top_k, embed):
    rows, matrix = load_index(index)
    scores = matrix @ embed(question)
    pages = {}
    for i in np.argsort(-scores, kind="stable"):
        name, page, _ = rows[i]
        pages.setdefault((name, page), float(scores[i]))
    return list(pages.items())[:top_k]


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Build and query a page index for the RAG homework.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("--doc-id")
    build_command.add_argument("--tessdata", type=Path, help="Enable OCR with rus and eng traineddata files")
    search_command = commands.add_parser("search")
    search_command.add_argument("question")
    search_command.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("EMB_KEY")
    if not api_key:
        parser.error("Set EMB_KEY to your course embedding key")
    embed = Embedder(api_key).embed
    if args.command == "build":
        if args.tessdata is not None:
            for language in ("rus", "eng"):
                if not (args.tessdata / f"{language}.traineddata").is_file():
                    parser.error(f"OCR requires {args.tessdata / (language + '.traineddata')}")
        build(args.index, ROOT / "data", args.doc_id, embed, args.tessdata)
    else:
        if args.top_k < 1:
            parser.error("--top-k must be positive")
        for (name, page), score in search(args.index, args.question, args.top_k, embed):
            print(f"{score:.4f}  {name}  p. {page}")


if __name__ == "__main__":
    main()
