"""Гибридный поиск по страницам отчётов → RRF:
- BM25 по словам со стеммингом (точные термины: «средства клиентов», «ОКУ», коды и названия строк);
- TF-IDF по символьным n-граммам (морфология, слипшиеся слова после OCR);
- эмбеддинги deepvk/USER-bge-m3 (перефразировки).

Единица индекса — страница PDF: эталон размечен страницами, а страница отчёта — законченный кусок
(таблица с шапкой или примечание). Перед поиском база сужается фильтром по метаданным (банк, год,
период — см. query_parser), а априорная уместность документа взвешивает оценки страниц.
"""

import gzip
import json
from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.config import EMB_KEY, EMBEDDER, INDEX_DIR, PAGES_PATH, RRF_K
from rag.query_parser import search_terms
from rag.text import tokenize

BM25_K1, BM25_B = 1.2, 0.75
PRIOR_POWER = 0.5          # насколько сильно метаданные (год/период) сдвигают ранжирование
WEIGHTS = {"bm25": 1.0, "tfidf": 0.5, "dense": 0.5}   # веса в RRF: подобраны на train (eval/tune_retrieval.py)


def page_header(page):
    return f"{page['title']}. Страница {page['page']}."


def title_zone(text, lines=6):
    """Первые строки страницы: название отчёта/примечания («15. Средства клиентов (продолжение)»)."""
    return " ".join(text.split("\n")[:lines])


def indexed_text(page):
    # заголовочную зону повторяем — так совпадение с названием таблицы весит больше, чем случайное слово в тексте
    return page_header(page) + "\n" + (title_zone(page["text"]) + "\n") * 2 + page["text"]


def load_pages():
    with gzip.open(PAGES_PATH, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


class BM25:
    def __init__(self, texts):
        docs = [tokenize(t) for t in texts]
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, tokens in enumerate(docs):
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(c)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float32)
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        idf = np.log((len(docs) - df + 0.5) / (df + 0.5) + 1.0)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * lengths / lengths.mean())
        tf = tf.tocoo()
        weights = idf[tf.col] * tf.data * (BM25_K1 + 1) / (tf.data + norm[tf.row])
        self.matrix = sparse.csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape, dtype=np.float32)

    def scores(self, query):
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for t in tokenize(query):
            if t in self.vocab:
                q[self.vocab[t]] += 1
        return self.matrix @ q


class PageIndex:
    def __init__(self, use_dense=True):
        self.pages = load_pages()
        self.key_to_row = {(p["doc_id"], p["page"]): i for i, p in enumerate(self.pages)}
        doc_rows = {}
        for i, p in enumerate(self.pages):
            doc_rows.setdefault(p["doc_id"], []).append(i)
        self.doc_rows = {d: np.array(r) for d, r in doc_rows.items()}
        texts = [indexed_text(p).replace("ё", "е") for p in self.pages]
        self.bm25 = BM25(texts)
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                     min_df=2, max_df=0.5, dtype=np.float32)
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self.embeddings = None
        self.embedder = None
        if use_dense:
            path = INDEX_DIR / "page_embeddings.npy"
            if not path.exists():
                print(f"[retriever] нет {path} — ищу без эмбеддингов (соберите индекс: python -m rag.index)")
            elif EMBEDDER == "course" and not EMB_KEY:
                print("[retriever] EMBEDDER=course, но не задан EMB_KEY — ищу без эмбеддингов")
            else:
                from rag.embeddings import get_embedder

                self.embeddings = np.load(path)
                if len(self.embeddings) != len(self.pages):
                    raise RuntimeError("index/page_embeddings.npy не совпадает с data/pages.jsonl.gz — "
                                       "пересоберите индекс: python -m rag.index")
                self.embedder = get_embedder()

    def search(self, query, doc_priors: dict, k=6, exclude=()):
        """Страницы из документов doc_priors по убыванию релевантности: [{doc_id, page, text, score, ...}]."""
        docs = [d for d in doc_priors if d in self.doc_rows]
        if not docs:
            return []
        rows = np.concatenate([self.doc_rows[d] for d in docs])
        prior = np.concatenate([np.full(len(self.doc_rows[d]), doc_priors[d], dtype=np.float32) for d in docs])
        prior = prior ** PRIOR_POWER
        terms = search_terms(query) or query

        fused = WEIGHTS["bm25"] / (RRF_K + rank_of(self.bm25.scores(terms)[rows] * prior))
        tfidf = (self.tfidf_matrix[rows] @ self.tfidf.transform([terms.replace("ё", "е")]).T).toarray().ravel()
        fused += WEIGHTS["tfidf"] / (RRF_K + rank_of(tfidf * prior))
        if self.embeddings is not None:
            q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)   # эмбеддеру — полный вопрос
            q /= np.linalg.norm(q) + 1e-9
            dense = self.embeddings[rows] @ q
            fused += WEIGHTS["dense"] / (RRF_K + rank_of(dense * prior))
        order = np.argsort(-fused, kind="stable")
        hits = []
        for i in order:
            page = self.pages[rows[i]]
            if (page["doc_id"], page["page"]) in exclude:
                continue
            hits.append({**page, "score": float(fused[i])})
            if len(hits) == k:
                break
        return hits

    def get_page(self, doc_id, page):
        row = self.key_to_row.get((doc_id, int(page)))
        return None if row is None else self.pages[row]


def rank_of(scores):
    """Ранг каждого элемента (0 — лучший)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores))
    return ranks


@lru_cache(maxsize=1)
def get_index():
    return PageIndex()
