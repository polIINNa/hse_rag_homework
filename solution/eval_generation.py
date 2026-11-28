"""Качество генерации без поиска: модели даются эталонные страницы вопроса (gold_context).

Из корня репозитория: python -m solution.eval_generation data/train.jsonl --limit 25
Это верхняя граница answer_correctness: каким был бы ответ при идеальном поиске.
"""

import argparse
import json
from statistics import mean

from eval.eval_functions import score_context
from eval.rag_example import to_context
from solution.build_index import load_pages, page_id
from solution.generate import generate_answer
from solution.judge import judge


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path")
    parser.add_argument("--limit", type=int, default=25, help="сколько вопросов взять, равномерно по файлу")
    args = parser.parse_args()

    with open(args.dataset_path, encoding="utf-8") as source:
        items = [json.loads(line) for line in source if line.strip()]
    items = items[::max(1, len(items) // args.limit)][:args.limit]  # равномерно, чтобы попали разные банки
    pages = {page_id(page): page for page in load_pages()}

    correct, cited = [], []
    for item in items:
        documents = [pages[f"{g['doc_id']}:{p}"] for g in item["gold_context"] for p in g["pages"]]
        answer, used = generate_answer(item["question"], documents)
        judged = judge(item["gold_answer"], answer)
        correct.append(judged["answer_correctness"])
        cited.append(score_context(item["gold_context"], to_context(used)))
        print(f"{judged['answer_correctness']} | {item['question_id']} | {answer[:120]}", flush=True)
        if not judged["answer_correctness"]:
            print(f"    эталон: {item['gold_answer'][:200]}\n    судья: {judged['reason'][:300]}", flush=True)

    print(f"\nanswer_correctness: {mean(correct):.3f} на {len(items)} вопросах")
    print(f"указанные моделью страницы: recall {mean(c['context_recall'] for c in cited):.3f}, "
          f"precision {mean(c['context_precision'] for c in cited):.3f}")


if __name__ == "__main__":
    main()
