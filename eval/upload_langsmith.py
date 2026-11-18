"""Из корня репозитория: python -m eval.upload_langsmith data/test.jsonl hw-rag-test"""

import argparse
import json

from langsmith import Client


def read_examples(dataset_path):
    """Примеры в формате, который ждёт eval/evaluator_langsmith.py."""
    examples = []
    with open(dataset_path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            examples.append({
                "inputs": {"question": item["question"]},
                "outputs": {"gold_answer": item["gold_answer"], "gold_context": item["gold_context"]},
                "metadata": {"question_id": item["question_id"], "split": item["split"]},
            })
    if not examples:
        raise ValueError("В датасете нет вопросов.")
    return examples


def upload(dataset_path, dataset_name):
    examples = read_examples(dataset_path)
    client = Client()
    if client.has_dataset(dataset_name=dataset_name):
        raise SystemExit(f"Датасет {dataset_name!r} уже есть в LangSmith: повторная загрузка создала бы дубли.")
    dataset = client.create_dataset(dataset_name, description=f"Вопросы из {dataset_path}")
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Загружено {len(examples)} вопросов в датасет {dataset_name!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("dataset_name")
    args = parser.parse_args()
    upload(args.dataset_path, args.dataset_name)
