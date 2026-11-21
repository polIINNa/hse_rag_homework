"""Индекс: чанки страниц, BM25 и эмбеддинги, гибридный поиск по страницам.

Из корня репозитория: python -m rag.index  (пересчитать эмбеддинги в data/index/, если поменялись чанки)

Эмбеддинги корпуса в data/index/ посчитаны курсовым эмбеддером (bge-m3) — тем же, что кодирует запросы.
Через курсовой эмбеддер пересчёт идёт около 2.5 часа (лимит 60 запросов в минуту), локальным bge-m3 — минуты на GPU.

Каждая страница режется на чанки по строкам, у каждого чанка есть doc_id и номер страницы.
К тексту чанка добавляется шапка «организация, отчёт, страница» и первые строки страницы
(обычно название формы или примечания) — без них куски таблиц теряют смысл.
Поиск ранжирует страницы: max по чанкам страницы в BM25 и в векторном поиске, затем RRF.
"""

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import Stemmer
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from rag.parse import load_pages
from rag.router import DOCUMENTS

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
INDEX = ROOT / "data" / "index"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
CHUNK_CHARS = 1500
HEAD_LINES = 3
RRF_K = 20
DENSE_WEIGHT = 1.5  # подобрано на train: векторный поиск точнее BM25

_encode_lock = threading.Lock()
_stem_ru = Stemmer.Stemmer("russian")
_stem_en = Stemmer.Stemmer("english")


def tokenize(text):
    tokens = re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))
    return [_stem_en.stemWord(t) if t.isascii() else _stem_ru.stemWord(t) for t in tokens]


def doc_title(doc_id):
    org, year, kind = doc_id.split("_")
    period = f"годовая отчётность за {year} год" if kind == "annual" else f"промежуточная отчётность за 6 месяцев {year} года"
    return f"{DOCUMENTS[doc_id]['organization']}, {period}"


def split_page(text):
    lines = [line for line in text.splitlines() if line.strip()]
    head = "\n".join(lines[:HEAD_LINES])
    chunks, current = [], []
    size = 0
    for line in lines:
        if current and size + len(line) > CHUNK_CHARS:
            chunks.append("\n".join(current))
            current, size = current[-2:], sum(len(x) for x in current[-2:])  # перекрытие в две строки
        current.append(line)
        size += len(line)
    if current:
        chunks.append("\n".join(current))
    return head, chunks or [""]


def build_chunks(pages):
    chunks = []
    for page in pages:
        head, parts = split_page(page["text"])
        header = f"{doc_title(page['doc_id'])}, стр. {page['page']}"
        for i, part in enumerate(parts):
            prefix = header if i == 0 else f"{header}\n{head}"
            chunks.append({"doc_id": page["doc_id"], "page": page["page"], "text": f"{prefix}\n{part}"})
    return chunks


class RemoteEncoder:
    """Курсовой эмбеддер (bge-m3, 1024, нормированный вектор) по HTTP: один текст на запрос."""

    def __init__(self, url, key, max_concurrent=4):
        import requests

        self.session = requests.Session()
        self.url, self.headers = url, {"Authorization": f"Bearer {key}"}
        self.slots = threading.Semaphore(max_concurrent)

    def _embed(self, text):
        import requests

        for attempt in range(8):
            try:
                with self.slots:
                    response = self.session.post(self.url, headers=self.headers, json={"text": text[:20000]},
                                                 timeout=60)
            except requests.exceptions.RequestException:
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(float(response.headers.get("Retry-After", min(2 ** attempt, 30))))
                continue
            response.raise_for_status()
            return response.json()["embedding"]
        response.raise_for_status()

    def encode(self, texts, **_):
        return np.array([self._embed(text) for text in texts], dtype=np.float32)


def load_encoder():
    """Векторизация запросов: курсовой эмбеддер, если задан EMB_KEY, иначе локальный bge-m3."""
    if os.environ.get("EMB_KEY") and os.environ.get("EMBEDDER", "remote") == "remote":
        return RemoteEncoder(os.environ.get("EMB_URL", "https://embeddings.parsers360.ru:10443/embed"),
                             os.environ["EMB_KEY"])
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def chunks_fingerprint(chunks):
    return hashlib.sha256("\n".join(f"{c['doc_id']}|{c['page']}|{c['text']}" for c in chunks).encode()).hexdigest()


class Index:
    def __init__(self, encoder=None):
        self.pages = {(p["doc_id"], p["page"]): p for p in load_pages()}
        # Чанки детерминированно строятся из data/parsed; эмбеддинги к ним лежат в data/index.
        self.chunks = build_chunks(self.pages.values())
        meta = {"model": EMBEDDING_MODEL, "chunks": len(self.chunks), "sha256": chunks_fingerprint(self.chunks)}
        self.encoder = encoder
        meta_path, emb_path = INDEX / "meta.json", INDEX / "embeddings.npy"
        if meta_path.exists() and emb_path.exists() and json.loads(meta_path.read_text()) == meta:
            self.embeddings = np.load(emb_path).astype(np.float32)
        else:
            self.embeddings = self._encode([c["text"] for c in self.chunks], batch_size=16, progress=True)
            INDEX.mkdir(parents=True, exist_ok=True)
            np.save(emb_path, self.embeddings.astype(np.float16))
            meta_path.write_text(json.dumps(meta, indent=1))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.doc_of_chunk = np.array([c["doc_id"] for c in self.chunks])
        self.page_of_chunk = np.array([c["page"] for c in self.chunks])

    def _encode(self, texts, batch_size=32, progress=False):
        with _encode_lock:
            if self.encoder is None:
                self.encoder = load_encoder()
            return self.encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                       show_progress_bar=progress, convert_to_numpy=True).astype(np.float32)

    def _page_ranks(self, scores, mask):
        """Страницы по убыванию max-скора их чанков среди отфильтрованных."""
        best = {}
        for i in np.flatnonzero(mask):
            key = (self.doc_of_chunk[i], int(self.page_of_chunk[i]))
            if scores[i] > best.get(key, -np.inf):
                best[key] = scores[i]
        return sorted(best, key=best.get, reverse=True)

    def search(self, query, doc_ids=None, top_k=8):
        """Гибридный поиск страниц. Возвращает [{doc_id, page, score, text}]."""
        mask = np.isin(self.doc_of_chunk, list(doc_ids)) if doc_ids else np.ones(len(self.chunks), bool)
        if not mask.any():
            mask = np.ones(len(self.chunks), bool)
        dense = self.embeddings @ self._encode([query])[0]
        sparse = np.asarray(self.bm25.get_scores(tokenize(query)))
        fused = {}
        for ranking, weight in ((self._page_ranks(dense, mask), DENSE_WEIGHT), (self._page_ranks(sparse, mask), 1.0)):
            for rank, key in enumerate(ranking[:200]):
                fused[key] = fused.get(key, 0.0) + weight / (RRF_K + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [{"doc_id": d, "page": p, "score": round(fused[(d, p)], 5), "text": self.pages[(d, p)]["text"]}
                for d, p in top]


if __name__ == "__main__":
    index = Index()
    print(f"Чанков: {len(index.chunks)}, страниц: {len(index.pages)}, эмбеддинги: {index.embeddings.shape}")
