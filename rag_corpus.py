import json
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import pymupdf


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INDEX = ROOT / "index"
TESSDATA = INDEX / "tessdata"
OCR_FILES = {
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    "rus": "e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225",
}


def prepare_ocr():
    TESSDATA.mkdir(exist_ok=True)
    for language, digest in OCR_FILES.items():
        path = TESSDATA / f"{language}.traineddata"
        if path.exists() and sha256(path.read_bytes()).hexdigest() == digest:
            continue
        url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/{path.name}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != digest:
            raise ValueError(f"Unexpected OCR language data for {language}.")
        path.write_bytes(data)


def ocr_batch(batch):
    source, pages = batch
    output = {}
    with pymupdf.open(source) as document:
        for number in pages:
            page = document[number - 1]
            if page.get_images() or page.get_drawings():
                textpage = page.get_textpage_ocr(language="rus+eng", dpi=180, full=True, tessdata=str(TESSDATA))
                output[number] = page_text(page, textpage)
    return output


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def page_text(page, textpage=None):
    text = page.get_text("text", sort=True, textpage=textpage)
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def extract_pages():
    for report in read_jsonl(DATA / "documents.jsonl"):
        source = DATA / "reports" / report["path"]
        with pymupdf.open(source) as document:
            if len(document) != report["pages"]:
                raise ValueError(f"Unexpected page count in {report['path']}")
            for number, page in enumerate(document, 1):
                yield {
                    "doc_id": report["doc_id"],
                    "page": number,
                    "text": page_text(page),
                }


def load_native_pages():
    INDEX.mkdir(exist_ok=True)
    path = INDEX / "pages-v1.jsonl"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as output:
            for page in extract_pages():
                output.write(json.dumps(page, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return read_jsonl(path)


def load_pages():
    path = INDEX / "pages-ocr-v1.jsonl"
    if path.exists():
        return read_jsonl(path)
    pages = load_native_pages()
    by_doc = {}
    for page in pages:
        if len(re.findall(r"[а-яёa-z]", page["text"], re.I)) < 60:
            by_doc.setdefault(page["doc_id"], {})[page["page"]] = page
    prepare_ocr()
    report_paths = {report["doc_id"]: DATA / "reports" / report["path"] for report in read_jsonl(DATA / "documents.jsonl")}
    batches = []
    docs = []
    for doc_id, sparse_pages in by_doc.items():
        numbers = list(sparse_pages)
        for offset in range(0, len(numbers), 8):
            batches.append((report_paths[doc_id], numbers[offset:offset + 8]))
            docs.append(doc_id)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for index, (doc_id, texts) in enumerate(zip(docs, pool.map(ocr_batch, batches)), 1):
            for number, text in texts.items():
                page = by_doc[doc_id][number]
                if len(text) > len(page["text"]):
                    page["text"] = text
            if index % 10 == 0 or index == len(batches):
                print(f"OCR completed for {index}/{len(batches)} batches", flush=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as output:
        for page in pages:
            output.write(json.dumps(page, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return pages


def passages(text, target=1400):
    lines = [re.sub(r" {2,}", " | ", line.strip()) for line in text.splitlines() if line.strip()]
    chunk = []
    size = 0
    for line in lines:
        if chunk and size + len(line) > target:
            yield "\n".join(chunk)
            chunk = chunk[-2:]
            size = sum(map(len, chunk))
        chunk.append(line)
        size += len(line) + 1
    if chunk:
        yield "\n".join(chunk)
