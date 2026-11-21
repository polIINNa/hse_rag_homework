"""Оценка поиска без LLM: recall@K по страницам на train.

Из корня репозитория: python -m rag.eval_retrieval data/train.jsonl
"""

import argparse
import json
from statistics import mean

import numpy as np

from rag.agent import Agent
from rag.index import Index, tokenize
from rag.router import route


def ranked_pages(index, question, doc_ids, mode, top_k):
    mask = np.isin(index.doc_of_chunk, doc_ids) if doc_ids else np.ones(len(index.chunks), bool)
    if mode == "hybrid":
        return [(p["doc_id"], p["page"]) for p in index.search(question, doc_ids, top_k=top_k)]
    if mode == "dense":
        scores = index.embeddings @ index._encode([question])[0]
    else:
        scores = np.asarray(index.bm25.get_scores(tokenize(question)))
    return index._page_ranks(scores, mask)[:top_k]


def main(path, ks=(1, 3, 5, 10)):
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    index = Index()
    agent = Agent(index=index)
    results = {}
    for mode in ("bm25", "dense", "hybrid", "hybrid_norouter", "prefetch"):
        recalls = {k: [] for k in ks}
        for item in items:
            gold = {(c["doc_id"], p) for c in item["gold_context"] for p in c["pages"]}
            if mode == "prefetch":
                found = [(p["doc_id"], p["page"]) for p in agent.prefetch(item["question"])]
            else:
                doc_ids = None if mode == "hybrid_norouter" else route(item["question"])
                found = ranked_pages(index, item["question"], doc_ids, mode.split("_")[0], max(ks))
            for k in ks:
                recalls[k].append(len(gold & set(found[:k])) / len(gold))
        results[mode] = {f"R@{k}": round(mean(v), 3) for k, v in recalls.items()}
        print(mode, results[mode], flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    main(parser.parse_args().dataset_path)
