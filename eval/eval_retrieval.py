"""Быстрая оценка одного поиска (без LLM): context_recall страниц первичного поиска при разном k.

Из корня репозитория:  python -m eval.eval_retrieval data/train.jsonl
"""

import argparse
import json
from statistics import mean

from eval.eval_functions import score_context
from rag.pipeline import initial_retrieval


def main(dataset_path, ks=(1, 3, 5, 8)):
    items = [json.loads(line) for line in open(dataset_path, encoding="utf-8") if line.strip()]
    recall = {k: [] for k in ks}
    for item in items:
        hits = initial_retrieval(item["question"], k=max(ks))
        for k in ks:
            context = {}
            for hit in hits[:k]:
                context.setdefault(hit["doc_id"], []).append(hit["page"])
            found = [{"doc_id": d, "pages": p} for d, p in context.items()]
            recall[k].append(score_context(item["gold_context"], found)["context_recall"])
    for k in ks:
        print(f"recall@{k}: {mean(recall[k]):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    main(parser.parse_args().dataset_path)
