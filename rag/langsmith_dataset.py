"""Загрузить датасет в LangSmith для eval/evaluator_langsmith.py.

Из корня репозитория: python -m rag.langsmith_dataset data/test.jsonl hse-rag-test
Затем: python -m eval.evaluator_langsmith hse-rag-test
"""

import argparse
import json

from dotenv import load_dotenv
from langsmith import Client

from rag.agent import ROOT

load_dotenv(ROOT / ".env")


def upload(path, name):
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    client = Client()
    if client.has_dataset(dataset_name=name):
        print(f"Датасет {name} уже есть")
        return
    dataset = client.create_dataset(name, description=f"RAG по финансовой отчётности: {path}")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": item["question"]} for item in items],
        outputs=[{"gold_answer": item["gold_answer"], "gold_context": item["gold_context"]} for item in items],
        metadata=[{"question_id": item["question_id"]} for item in items],
    )
    print(f"Загружено {len(items)} примеров в {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("dataset_name")
    args = parser.parse_args()
    upload(args.dataset_path, args.dataset_name)
