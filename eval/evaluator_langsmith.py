"""Из корня репозитория: python -m eval.evaluator_langsmith ИМЯ_ДАТАСЕТА"""

import argparse
from pathlib import Path
from time import perf_counter

from langsmith import Client, tracing_context

from eval.evaluator_excel import read_items, result_row, save_report
from eval.eval_functions import score_answer, score_context
from eval.rag_example import RAG
from rag_agent import retry


def metrics(outputs, reference_outputs):
    context = score_context(reference_outputs["gold_context"], outputs["context"])
    answer = retry(lambda: score_answer(reference_outputs["gold_answer"], outputs["answer"]))
    return [
        {"key": "context_recall", "score": context["context_recall"]},
        {"key": "context_precision", "score": context["context_precision"]},
        {"key": "answer_correctness", "score": answer["answer_correctness"], "comment": answer["reason"]},
        {"key": "time", "score": outputs["time"]},
    ]


def evaluate(dataset_name, dataset_path=None, workers=1, output_path=None):
    system = RAG()
    client = Client()
    data = dataset_name
    if dataset_path is not None:
        items = list(read_items(dataset_path))
        examples = [
            {
                "inputs": {"question": item["question"]},
                "outputs": {"gold_answer": item["gold_answer"], "gold_context": item["gold_context"]},
                "metadata": {"question_id": item["question_id"]},
            }
            for item in items
        ]
        if not examples:
            raise ValueError("В датасете нет вопросов.")
        if not client.has_dataset(dataset_name=dataset_name):
            dataset = client.create_dataset(dataset_name)
            client.create_examples(dataset_id=dataset.id, examples=examples)
        stored = {
            (example.metadata or {}).get("question_id"): example
            for example in client.list_examples(dataset_name=dataset_name)
        }
        data = []
        for expected in examples:
            question_id = expected["metadata"]["question_id"]
            example = stored.get(question_id)
            if example is None or example.inputs != expected["inputs"] or example.outputs != expected["outputs"]:
                raise ValueError(f"LangSmith dataset differs from the local source at {question_id}")
            data.append(example)
        if len(stored) != len(data):
            raise ValueError("LangSmith dataset size differs from the local source.")

    def answer_question(inputs):
        start = perf_counter()
        result = system.rag(inputs["question"])
        return {**result, "time": perf_counter() - start}

    with tracing_context(enabled=True, client=client):
        experiment = client.evaluate(
            answer_question, data=data, evaluators=[metrics],
            experiment_prefix="financial-rag", max_concurrency=0 if workers == 1 else workers,
        )
    if output_path is not None:
        rows = []
        failed = []
        for item in experiment:
            example = item["example"]
            question_id = (example.metadata or {}).get("question_id") or str(example.id)
            scores = {result.key: result.score for result in item["evaluation_results"]["results"]}
            if not {"context_recall", "context_precision", "answer_correctness", "time"}.issubset(scores):
                failed.append(question_id)
                continue
            rows.append(result_row(
                {"question_id": question_id, **example.inputs, **example.outputs},
                item["run"].outputs,
                scores,
            ))
        if failed or not rows:
            raise RuntimeError(f"Incomplete LangSmith experiment. Failed questions: {failed}")
        save_report(rows, output_path)
    client.flush()
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output and args.dataset_path and args.output.resolve() == args.dataset_path.resolve():
        parser.error("Dataset and report must use separate paths")
    result = evaluate(args.dataset_name, args.dataset_path, args.workers, args.output)
    print(f"Experiment: {result.experiment_name}")
