from dataclasses import dataclass
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "index" / "reports.npz"
MODEL_CACHE = ROOT / "index" / "hf"


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: str
    organization: str
    pages: int

    @property
    def label(self):
        _, year, period = self.doc_id.split("_", 2)
        period_name = "годовой отчёт" if period == "annual" else "отчёт за шесть месяцев"
        return f"{self.organization} | {period_name} | {year}"


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    page: int
    text: str

    @property
    def source(self):
        return self.doc_id, self.page


def read_documents(path):
    with Path(path).open(encoding="utf-8") as source:
        return [Document(**json.loads(line)) for line in source if line.strip()]


def laid_out_text(page, textpage=None):
    text = page.get_text("text", sort=True, textpage=textpage)
    return "\n".join(
        re.sub(r"[ \t]{3,}", " | ", line.strip())
        for line in text.splitlines()
        if line.strip()
    )


def split_page(text, size=1050, overlap=160):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for separator in ("\n", " "):
                boundary = text.rfind(separator, start + size // 2, end)
                if boundary != -1:
                    end = boundary + 1
                    break
        if chunk := text[start:end].strip():
            yield chunk
        if end == len(text):
            break
        boundary = text.find("\n", end - overlap, end - 1)
        if boundary == -1:
            boundary = text.find(" ", end - overlap, end - 1)
        start = boundary + 1 if boundary != -1 else end


def extract_chunks(documents, reports_dir, ocr_dir=None):
    import pymupdf

    if ocr_dir is not None:
        ocr_dir = Path(ocr_dir).resolve()
        for language in ("rus", "eng"):
            if not (ocr_dir / f"{language}.traineddata").is_file():
                raise FileNotFoundError(f"Missing {language}.traineddata in {ocr_dir}")

    chunks = []
    skipped = []
    ocr_pages = 0
    for document in documents:
        path = Path(reports_dir) / document.path
        with pymupdf.open(path) as pdf:
            if len(pdf) != document.pages:
                raise ValueError(f"{document.doc_id}: expected {document.pages} PDF pages, got {len(pdf)}")
            for page in pdf:
                text = laid_out_text(page)
                if sum(char.isalpha() for char in text) < 80 and page.get_image_info():
                    if ocr_dir is not None:
                        ocr = page.get_textpage_ocr(
                            language="rus+eng", dpi=200, full=True, tessdata=str(ocr_dir),
                        )
                        text = laid_out_text(page, ocr) or text
                        ocr_pages += 1
                    else:
                        skipped.append((document.doc_id, page.number + 1))
                if any(char.isalpha() for char in text):
                    chunks.extend(Chunk(document.doc_id, page.number + 1, part) for part in split_page(text))
    return chunks, skipped, ocr_pages
