"""Быстрое сравнение retrieval на выборке train: v1 (страницы) vs v2 (чанки).

Не вызывает LLM-судью, поэтому не расходует лимит API.
Запуск: python -m eval.compare_retrieval --n 50
"""

import argparse
import json
from statistics import mean

import numpy as np

from eval.eval_functions import score_context
from eval.rag_example import RAG


def load_items(n):
    items = []
    with open("data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if len(items) >= n:
                    break
    return items


def eval_v1(items, top_k, model):
    from pathlib import Path

    cache = Path("eval/cache")
    pages = [json.loads(l) for l in open(cache / "pages.jsonl", encoding="utf-8") if l.strip()]
    emb = np.load(cache / "embeddings.npy")
    rec, prec = [], []
    for it in items:
        q = model.encode([f"query: {it['question']}"], normalize_embeddings=True)[0]
        scores = emb @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        by_doc = {}
        for i in idx:
            p = pages[int(i)]
            by_doc.setdefault(p["doc_id"], set()).add(p["page"])
        context = [{"doc_id": d, "pages": sorted(ps)} for d, ps in by_doc.items()]
        s = score_context(it["gold_context"], context)
        rec.append(s["context_recall"])
        prec.append(s["context_precision"])
    return mean(rec), mean(prec)


def eval_v2(items, top_k, rag):
    rec, prec = [], []
    for it in items:
        hits = rag._search(it["question"], top_k_pages=top_k)
        context = rag._group_context(hits)
        s = score_context(it["gold_context"], context)
        rec.append(s["context_recall"])
        prec.append(s["context_precision"])
    return mean(rec), mean(prec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--top-k", type=int, nargs="+", default=[6, 8, 10, 12])
    args = parser.parse_args()

    items = load_items(args.n)
    print(f"items: {len(items)}")
    rag = RAG()
    model = rag._get_model()

    print("\nV1 (page-level, old cache)")
    for k in args.top_k:
        r, p = eval_v1(items, k, model)
        print(f"  top_k={k:2d}  recall={r:.3f}  precision={p:.3f}")

    print("\nV2 (chunks + metadata boost)")
    for k in args.top_k:
        r, p = eval_v2(items, k, rag)
        print(f"  top_k={k:2d}  recall={r:.3f}  precision={p:.3f}")
