"""Загрузить test.jsonl в LangSmith как датасет для eval/evaluator_langsmith.py.

Из корня репозитория:  python -m eval.langsmith_dataset data/test.jsonl hse-rag-test
"""

import argparse
import json
from pathlib import Path

from langsmith import Client

import rag.config  # noqa: F401 — загружает .env (LANGSMITH_API_KEY и др.)


def main(dataset_path, name):
    client = Client()
    if client.has_dataset(dataset_name=name):
        print(f"Датасет «{name}» уже есть в LangSmith — используем его.")
        return
    dataset = client.create_dataset(dataset_name=name, description="RAG по финансовой отчётности банков (HSE)")
    items = [json.loads(line) for line in Path(dataset_path).open(encoding="utf-8") if line.strip()]
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": item["question"]},
                "outputs": {"gold_answer": item["gold_answer"], "gold_context": item["gold_context"]},
                "metadata": {"question_id": item["question_id"]},
            }
            for item in items
        ],
    )
    print(f"Загружено {len(items)} вопросов в датасет «{name}».")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("name")
    args = parser.parse_args()
    main(args.dataset_path, args.name)
