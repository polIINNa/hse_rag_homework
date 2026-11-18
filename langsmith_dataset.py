"""Необязательный шаг: создать датасет LangSmith для исходного evaluator_langsmith."""
import argparse

from rag.common import ROOT, load_env, read_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    load_env()
    from langsmith import Client
    client = Client()
    if client.has_dataset(dataset_name=args.name):
        print(f"Используется существующий датасет: {args.name}")
        raise SystemExit(0)
    dataset = client.create_dataset(dataset_name=args.name, description="Финансовый RAG, 141 test-вопрос")
    examples = [{"inputs": {"question": item["question"]},
                 "outputs": {"gold_answer": item["gold_answer"], "gold_context": item["gold_context"]},
                 "metadata": {"question_id": item["question_id"]}}
                for item in read_jsonl(ROOT / "data/test.jsonl")]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Датасет создан: {args.name}. Затем python -m eval.evaluator_langsmith {args.name}")
