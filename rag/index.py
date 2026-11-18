import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import fitz
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from rag.common import ROOT, read_jsonl, traced, write_json
from rag.ranker import POOL_SIZE, fit_ranker, rerank_with_model
from rag.retrieval import block_layout, compact_text, excerpt, report_hint, rerank_by_rows, tokens
from rag.scans import add_scan_evidence

INDEX_VERSION = "hybrid-v2.1-blocks-bm25"

ALIASES = {
    "alfa": ["альфа", "alfa"], "domrf": ["дом.рф", "дом рф", "domrf"],
    "gpb": ["газпромбанк", "газпромбанк-фонд", "gpb"],
    "mkb": ["московск", "мкб", "mkb"], "rshb": ["россельхоз", "рсхб", "rshb"],
    "sber": ["сбер", "sber"], "sovkom": ["совком", "sovkom"],
    "tbank": ["тинькофф", "т-банк", "т-банка", "т банк", "tbank", "t-bank"],
    "vtb": ["втб", "vtb"],
}


def norm(text):
    text = text.translate(str.maketrans({c: "-" for c in "‐‑‒–—"}))
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е").replace("\u00ad", "")).strip()


def organizations(question):
    text = norm(question)
    return [name for name, aliases in ALIASES.items() if any(alias in text for alias in aliases)]


def chunks(text, size=2200, overlap=300):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        yield text[start:end]
        if end == len(text):
            return
        start = max(start + 1, end - overlap)


PARSER_VERSION = "layout-v2-rotated-ocr"


def ocr_page(page, tessdata):
    """Повторить распознавание при повёрнутом скане, не меняя исходный PDF."""
    last_text = ""
    for angle in (0, 90, 270, 180):
        with fitz.open() as temporary:
            temporary.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
            target = temporary[0]
            target.set_rotation((target.rotation + angle) % 360)
            textpage = target.get_textpage_ocr(language="rus+eng", dpi=200, full=True, tessdata=str(tessdata))
            text = target.get_text("text", textpage=textpage, sort=True)
        if len(re.sub(r"\W", "", text)) > len(re.sub(r"\W", "", last_text)):
            last_text = text
        if len(re.sub(r"\W", "", text)) >= 40:
            return text, angle
    raise ValueError(f"OCR не прочитал страницу ни в одной ориентации; символов: {len(last_text)}")


def parse_corpus(reports, documents, tables=True, cache_dir=None):
    pages, diagnostics = [], []
    for doc in documents:
        path = Path(reports) / doc["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Нет {path}. Сначала python -m rag.prepare")
        cache = Path(cache_dir) / (doc["doc_id"] + ".json") if cache_dir else None
        signature = {"parser": PARSER_VERSION, "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "document": doc, "tables": tables}
        if cache and cache.exists():
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
                if saved["signature"] == signature:
                    pages.extend(saved["pages"])
                    diagnostics.extend(saved["diagnostics"])
                    print(f"Из кэша {doc['doc_id']}", flush=True)
                    continue
            except (ValueError, KeyError):
                pass
        first_page, first_diagnostic = len(pages), len(diagnostics)
        with fitz.open(path) as pdf:
            if len(pdf) != doc["pages"]:
                raise ValueError(f"{doc['doc_id']}: число страниц не совпало с documents.jsonl")
            for number, page in enumerate(pdf, 1):
                text = page.get_text("text", sort=True).replace("\u00ad", "")
                ocr = False
                if len(re.sub(r"\W", "", text)) < 40 and page.get_images():
                    try:
                        text, rotation = ocr_page(page, ROOT / "data/tessdata")
                        ocr = True
                        if rotation:
                            diagnostics.append({"doc_id": doc["doc_id"], "page": number, "ocr_rotation": rotation})
                    except Exception as error:
                        raise RuntimeError(
                            f"Нужен OCR для {doc['doc_id']}, PDF-страница {number}. "
                            "Проверьте данные rus+eng: сначала python -m rag.prepare."
                        ) from error
                table_count = 0
                # Сохраняем исходный текст и добавляем таблицы с явными границами столбцов.
                if tables and text and len(re.findall(r"\d", text)) >= 50:
                    try:
                        found = page.find_tables(strategy="lines_strict")
                        for table in found.tables:
                            if table.row_count >= 3 and table.col_count >= 2:
                                text += "\n\n[Таблица: столбцы разделены |]\n" + table.to_markdown()
                                table_count += 1
                    except Exception as error:
                        # Страница остаётся доступна в исходном извлечении; сбой виден в журнале.
                        diagnostics.append({"doc_id": doc["doc_id"], "page": number,
                                            "table_warning": type(error).__name__})
                if text.strip():
                    pages.append({"doc_id": doc["doc_id"], "page": number,
                                  "organization": doc["organization"], "text": text.strip(),
                                  "ocr": ocr, "tables": table_count})
                else:
                    diagnostics.append({"doc_id": doc["doc_id"], "page": number, "empty": True})
        print(f"Разобран {doc['doc_id']}", flush=True)
        if cache:
            write_json(cache, {"signature": signature, "pages": pages[first_page:],
                               "diagnostics": diagnostics[first_diagnostic:]})
    return pages, diagnostics


def corpus_fingerprint(reports, documents, use_tables):
    h = hashlib.sha256(f"{PARSER_VERSION};tables={use_tables}".encode())
    for doc in documents:
        h.update(doc["doc_id"].encode())
        with (Path(reports) / doc["path"]).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


class PageIndex:
    def __init__(self, pages):
        if not pages:
            raise ValueError("Корпус пуст")
        self.retriever_version = INDEX_VERSION
        self.pages = [{**p, "text": compact_text(p["text"])} for p in pages]
        self.pieces, self.excerpts, texts = [], [], []
        for page_index, page in enumerate(self.pages):
            for piece in chunks(page["text"]):
                self.pieces.append(page_index)
                self.excerpts.append(piece)
                texts.append(piece)
        self.lexical = CountVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False,
                                       ngram_range=(1, 2), max_features=140000)
        counts = self.lexical.fit_transform(texts).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        dfs = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((counts.shape[0] - dfs + 0.5) / (dfs + 0.5))
        coo = counts.tocoo()
        k1, b = 1.5, 0.65
        denominator = coo.data + k1 * (1 - b + b * lengths[coo.row] / max(float(lengths.mean()), 1))
        values = idf[coo.col] * coo.data * (k1 + 1) / denominator
        self.bm25 = csr_matrix((values, (coo.row, coo.col)), shape=counts.shape, dtype=np.float32)
        names = self.lexical.get_feature_names_out()
        self.query_weights = np.array([0.15 if "#" in name else 1.0 for name in names], dtype=np.float32)
        self.term_idf = {name: float(value) for name, value in zip(names, idf) if " " not in name}
        self.char = TfidfVectorizer(preprocessor=norm, analyzer="char_wb", ngram_range=(3, 5),
                                    max_features=100000, sublinear_tf=True)
        chars = self.char.fit_transform(texts).astype(np.float32)
        dimensions = min(192, chars.shape[0] - 1, chars.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=max(1, dimensions), random_state=42)
        self.vectors = normalize(self.svd.fit_transform(chars)).astype(np.float32)

    @classmethod
    def load(cls, path=None):
        path = Path(path or ROOT / "artifacts/index.joblib")
        if not path.is_file():
            raise FileNotFoundError(f"Нет индекса {path}. Сначала python -m rag.index")
        index = joblib.load(path)
        if getattr(index, "retriever_version", None) != INDEX_VERSION:
            raise ValueError("Версия индекса устарела: сначала python -m rag.index")
        index.file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return index

    @traced("hybrid_retrieval", "retriever")
    def search(self, question, top_k=4, organization_hint=None, rerank=True):
        if not question.strip():
            raise ValueError("Пустой поисковый запрос")
        query = self.lexical.transform([question]).astype(np.float32)
        query.data = np.minimum(query.data, 2)
        query = query.multiply(self.query_weights)
        word_score = (self.bm25 @ query.T).toarray().ravel()
        vector = normalize(self.svd.transform(self.char.transform([question])))
        dense_score = np.maximum((self.vectors @ vector.T).ravel(), 0)
        banks = organizations(question) or (organization_hint or [])
        eligible = np.array([not banks or self.pages[p]["doc_id"].split("_")[0] in banks for p in self.pieces])
        if not eligible.any():
            return []
        word_score /= max(float(word_score[eligible].max()), 1e-8)
        dense_score /= max(float(dense_score[eligible].max()), 1e-8)
        scores = 0.85 * word_score + 0.15 * dense_score
        primary_year, years, period = report_hint(question)
        page_scores = {}
        for i, score in enumerate(scores):
            p = self.pieces[i]
            page = self.pages[p]
            if banks and page["doc_id"].split("_")[0] not in banks:
                continue
            # Год — мягкая прибавка: сравнительные данные могут быть в более позднем отчёте.
            score = float(score)
            if primary_year and primary_year in page["doc_id"]:
                score += 0.22
            elif any(year in page["doc_id"] for year in years):
                score += 0.05
            score += 0.08 * bool(period and period in page["doc_id"])
            if score > page_scores.get(p, -1):
                page_scores[p] = score
        ranked = sorted(page_scores, key=lambda p: (-page_scores[p], self.pages[p]["doc_id"], self.pages[p]["page"]))
        rank_model = getattr(self, "rank_model", None)
        pool_size = POOL_SIZE if rank_model is not None else 12
        pool = []
        if len(banks) > 1:
            for bank in banks:
                candidate = next((p for p in ranked if self.pages[p]["doc_id"].startswith(bank + "_")), None)
                if candidate is not None:
                    pool.append(candidate)
        for p in ranked:
            if len(pool) >= max(pool_size if rerank else 0, top_k, len(banks)):
                break
            if p not in pool:
                pool.append(p)
        candidates = [{**self.pages[p], "score": page_scores[p]} for p in pool]
        if rerank and candidates:
            if rank_model is None:
                candidates = rerank_by_rows(candidates, question, self.term_idf)
            else:
                candidates = rerank_with_model(rank_model, candidates, question, self.term_idf)
        selected = []
        # Для сравнения нескольких организаций резервируем хотя бы по одному месту каждой.
        if len(banks) > 1:
            for bank in banks:
                best = next((p for p in candidates if p["doc_id"].startswith(bank + "_")), None)
                if best is not None:
                    selected.append(best)
        selected.extend(p for p in candidates if p not in selected)
        return [{**p, "excerpt": excerpt(p["text"], question, idf=self.term_idf),
                 "excerpt_idf": {t: self.term_idf.get(t, 1.0) for t in tokens(question)},
                 "queries": [question]} for p in selected[:max(top_k, len(banks))]]


def build(reports=None, output=None, tables=True):
    reports = Path(reports or ROOT / "data/reports")
    output = Path(output or ROOT / "artifacts/index.joblib")
    documents = read_jsonl(ROOT / "data/documents.jsonl")
    fingerprint = corpus_fingerprint(reports, documents, tables)
    pages, diagnostics = parse_corpus(reports, documents, tables=tables, cache_dir=output.parent / "parsed")
    # Исходный OCR-кэш остаётся неизменным. Новый этап читает нативные PDF-блоки,
    # чтобы не перемешивать две колонки в строках повествовательных страниц.
    page_lookup = {(p["doc_id"], p["page"]): p for p in pages}
    column_pages = 0
    for doc in documents:
        with fitz.open(reports / doc["path"]) as pdf:
            for number, native in enumerate(pdf, 1):
                p = page_lookup.get((doc["doc_id"], number))
                if p and not p["ocr"]:
                    p["text"], layout = block_layout(native, p["text"])
                    p["layout"] = layout
                    column_pages += layout == "columns"
    print(f"Нативные блоки подготовлены: {column_pages} страниц с двумя колонками", flush=True)
    index = PageIndex(pages)
    train = ROOT / "data/train.jsonl"
    index.rank_model, candidates, positives = fit_ranker(index, read_jsonl(train))
    scan_tables = add_scan_evidence(index, reports, documents, ROOT / "data/tessdata")
    index.fingerprint = hashlib.sha256((fingerprint + INDEX_VERSION).encode() + train.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, output, compress=3)
    write_json(output.with_suffix(".json"), {"documents": len(documents), "pages_indexed": len(pages),
               "chunks": len(index.pieces), "embedding": "char-TFIDF + SVD, 192 dimensions max",
               "corpus_sha256": fingerprint, "index_version": INDEX_VERSION,
               "retrieval": "BM25 + local LSA + logistic regression reranker",
               "ranker_training_candidates": candidates, "ranker_training_positives": positives,
               "column_pages": column_pages, "scan_tables": scan_tables,
               "diagnostics": diagnostics})
    print(f"Индекс готов: {len(pages)} страниц, {len(index.pieces)} фрагментов", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-tables", action="store_true")
    args = parser.parse_args()
    # При -m классы для pickle должны принадлежать модулю rag.index, а не __main__.
    from rag.index import build as canonical_build
    canonical_build(args.reports, args.output, tables=not args.no_tables)
