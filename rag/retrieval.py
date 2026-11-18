import re
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from langsmith import traceable
from rank_bm25 import BM25Okapi
import snowballstemmer

from rag.config import INDEX, MODEL, read_jsonl

BANKS = {
    "alfa": r"альфа|alfa",
    "domrf": r"дом\s*\.?\s*рф|domrf",
    "gpb": r"газпромбанк|газфонд|gpb",
    "mkb": r"московск\w*\s+кредитн|\bмкб\b",
    "rshb": r"россельхоз|\bрсхб\b",
    "sber": r"сбер",
    "sovkom": r"совком",
    "tbank": r"\bт[\s-]?банк|тинькофф|тиньков|\bт[\s-]?технолог",
    "vtb": r"\bвтб\b|vtb",
}

STEMMER = snowballstemmer.stemmer("russian")


@lru_cache(maxsize=100000)
def stem(word):
    return STEMMER.stemWord(word)


STOP = {stem(w) for w in "какой каков как какая какие каковы сколько чему составляет составил согласно соответствии состоянию отчет отчетность обобщенный консолидированный промежуточный финансовый год года годы месяц месяцев шесть три период закончившийся декабрь июнь январь группа банк пао ао зао оао пп п к на за и в во по из у с со от для ли его ее их об объеме конец начало рублей руб млн тыс млрд российский россия белорусский общий всего всего отношении отношении".split()}
STOP.update(stem(w) for w in "чей чьи больше меньше насколько объем величина размер сумма значение указанный мсфо или отношение доля часть процентах сравнению изменилось аудировано отчетного".split())


def search_terms(query):
    text = query.lower().replace("ё", "е")
    text = re.sub(r"совокупн\w*\s+актив\w*", "итого активы", text)
    for pattern in BANKS.values():
        text = re.sub(pattern, " ", text)
    text = re.sub(r"российск\w*|московск\w*|кредитный банк|\bнпф\b|\bфонд\b", " ", text)
    words = re.findall(r"[а-яa-z]+", text)
    meaningful = [w for w in words if len(w) > 2 and stem(w) not in STOP]
    return " ".join(meaningful) or query


def tokens(text):
    return [stem(w) for w in re.findall(r"[а-яa-z]+", text.lower().replace("ё", "е"))]


def context_of(pages):
    docs = {}
    for page in pages:
        docs.setdefault(page["doc_id"], set()).add(page["page"])
    return [{"doc_id": doc, "pages": sorted(numbers)} for doc, numbers in sorted(docs.items())]


class Retriever:
    def __init__(self, index=INDEX):
        self.chunks = read_jsonl(index / "chunks.jsonl")
        self.pages = {(p["doc_id"], p["page"]): p for p in read_jsonl(index / "pages.jsonl")}
        self.vectors = np.load(index / "vectors.npy")
        if len(self.vectors) != len(self.chunks):
            raise ValueError("Индекс повреждён: число векторов не совпадает с числом фрагментов")
        self.model = TextEmbedding(MODEL, cache_dir=str(index / "models"), threads=4)
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in self.chunks])
        # Warm-up belongs to preparing the RAG, before the evaluation timer.
        list(self.model.embed(["финансовая отчетность"]))

    @traceable(run_type="retriever", name="hybrid_search")
    def search(self, query, top_k=6, doc_ids=None):
        banks = [bank for bank, pattern in BANKS.items() if re.search(pattern, query, re.I)]
        years = set(re.findall(r"20\d{2}", query))
        documents = {c["doc_id"] for c in self.chunks}
        allowed = set(doc_ids) if doc_ids else {d for d in documents if not banks or d.split("_")[0] in banks}
        if not doc_ids and years:
            dated = {d for d in allowed if d.split("_")[1] in years}
            if dated:
                allowed = dated
            if len(years) == 1:
                half_year = bool(re.search(r"июн|полугод|шесть\s+месяц|6\s*месяц|30[./]0?6", query, re.I))
                period = "6m" if half_year else "annual"
                # Only constrain organizations for which this period is present.
                period_docs = {d for d in allowed if d.endswith("_" + period)}
                covered_banks = {d.split("_")[0] for d in period_docs}
                allowed = period_docs | {d for d in allowed if d.split("_")[0] not in covered_banks}
        eligible = np.array([i for i, c in enumerate(self.chunks) if c["doc_id"] in allowed])
        if not len(eligible):
            return []
        focused = search_terms(query)
        vector = next(iter(self.model.embed([focused]))).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        dense = self.vectors[eligible] @ vector
        sparse = self.bm25.get_scores(tokens(focused))[eligible]
        scores = {}
        for ranking in (np.argsort(-dense, kind="stable")[:80], np.argsort(-sparse, kind="stable")[:80]):
            for rank, pos in enumerate(ranking):
                i = int(eligible[pos])
                scores[i] = scores.get(i, 0) + 1 / (40 + rank)
        ranked_pages = {}
        for i, score in scores.items():
            c = self.chunks[i]
            # A soft prior retains comparative figures from later reports.
            if c["doc_id"].split("_")[1] in years:
                score *= 1.2
            key = (c["doc_id"], c["page"])
            ranked_pages[key] = max(ranked_pages.get(key, 0), score)
        ranking = sorted(ranked_pages, key=lambda k: (-ranked_pages[k], k))
        keys = []
        if len(banks) > 1:
            for bank in banks:
                candidate = next((k for k in ranking if k[0].split("_")[0] == bank), None)
                if candidate is not None:
                    keys.append(candidate)
        for key in ranking:
            if len(keys) >= top_k:
                break
            if key not in keys:
                keys.append(key)
        keys = sorted(keys, key=lambda k: (-ranked_pages[k], k))[:top_k]
        return [self.pages[k] for k in keys]

    @traceable(run_type="retriever", name="read_pdf_page")
    def get_page(self, doc_id, page):
        item = self.pages.get((doc_id, page))
        return [item] if item else []
