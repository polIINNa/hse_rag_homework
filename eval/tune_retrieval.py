"""Подбор весов гибридного поиска на train (без LLM, нужен только EMB_KEY).

Из корня репозитория:  python3 -m eval.tune_retrieval

Векторы вопросов train считаются один раз (308 запросов к эмбеддеру, ~5–6 минут при лимите 60/мин)
и кэшируются в outputs/train_query_embeddings.npz; дальше варианты весов перебираются за секунды.
Печатает recall@k первичного поиска для каждого варианта — пришлите таблицу целиком.
"""

import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np

import rag.retriever as retriever
from eval.eval_functions import score_context
from rag.pipeline import initial_retrieval

CACHE = Path("outputs/train_query_embeddings.npz")
KS = (1, 3, 5, 8)
GRID = [  # (bm25, tfidf, dense)
    (1.0, 0.5, 0.0),
    (1.0, 0.5, 0.5),   # выбрано
    (1.0, 0.5, 1.0),
    (1.0, 0.5, 2.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.0, 2.0),
    (0.5, 0.5, 1.0),
    (0.0, 0.0, 1.0),   # только векторный поиск
]


class CachedEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache = dict(np.load(CACHE)) if CACHE.exists() else {}

    def embed_query(self, text):
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key not in self.cache:
            self.cache[key] = np.asarray(self.embedder.embed_query(text), dtype=np.float32)
            if len(self.cache) % 25 == 0:
                print(f"  векторы вопросов: {len(self.cache)}", flush=True)
        return self.cache[key]

    def save(self):
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **self.cache)


def recall_at_k(items):
    recall = {k: [] for k in KS}
    for item in items:
        hits = initial_retrieval(item["question"], k=max(KS))
        for k in KS:
            found = {}
            for hit in hits[:k]:
                found.setdefault(hit["doc_id"], []).append(hit["page"])
            context = [{"doc_id": d, "pages": p} for d, p in found.items()]
            recall[k].append(score_context(item["gold_context"], context)["context_recall"])
    return {k: mean(v) for k, v in recall.items()}


def main():
    items = [json.loads(line) for line in open("data/train.jsonl", encoding="utf-8") if line.strip()]
    index = retriever.get_index()
    if index.embeddings is None:
        raise SystemExit("Нет векторного индекса или EMB_KEY — проверьте index/page_embeddings.npy и .env")
    cached = CachedEmbedder(index.embedder)
    index.embedder = cached
    print(f"train: {len(items)} вопросов. Считаю векторы вопросов (один раз)…")
    for item in items:
        cached.embed_query(item["question"])
    cached.save()

    print("\nbm25  tfidf  dense | " + "  ".join(f"recall@{k}" for k in KS))
    for bm25, tfidf, dense in GRID:
        retriever.WEIGHTS.update(bm25=bm25, tfidf=tfidf, dense=dense)
        scores = recall_at_k(items)
        mark = "  ← сейчас" if (bm25, tfidf, dense) == (1.0, 0.5, 0.5) else ""
        print(f"{bm25:4.1f}  {tfidf:5.1f}  {dense:5.1f} | " + "  ".join(f"{scores[k]:9.3f}" for k in KS) + mark,
              flush=True)


if __name__ == "__main__":
    main()
