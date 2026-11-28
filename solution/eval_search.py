"""Качество поиска без LLM: средние context_recall и context_precision для top-1, top-3 и top-5.

Из корня репозитория: python -m solution.eval_search data/train.jsonl
Сравнение без фильтра по банку: добавьте --no-bank-filter
"""

import argparse
import json
from statistics import mean

from eval.eval_functions import score_context
from eval.rag_example import SimpleRAG, to_context

KS = (1, 3, 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path")
    parser.add_argument("--no-bank-filter", action="store_true")
    args = parser.parse_args()

    with open(args.dataset_path, encoding="utf-8") as source:
        items = [json.loads(line) for line in source if line.strip()]
    rag = SimpleRAG(use_bank_filter=not args.no_bank_filter)  # поиск тот же, что у агента
    scores = {k: [] for k in KS}
    for number, item in enumerate(items, start=1):
        documents = rag.search(item["question"], k=max(KS))  # один запрос, top-1 и top-3 — его начало
        for k in KS:
            scores[k].append(score_context(item["gold_context"], to_context(documents[:k])))
        if number % 50 == 0:
            print(f"{number} из {len(items)}", flush=True)

    for k in KS:
        recall = mean(score["context_recall"] for score in scores[k])
        precision = mean(score["context_precision"] for score in scores[k])
        print(f"top-{k}: context_recall {recall:.3f}, context_precision {precision:.3f}")


if __name__ == "__main__":
    main()
