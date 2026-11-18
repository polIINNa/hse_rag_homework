"""Запускает оценку агентного RAG на датасете LangSmith."""

import argparse
from functools import wraps
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid5

from dotenv import load_dotenv
from langsmith import Client, trace, tracing_context
from langsmith.schemas import ExampleCreate

from eval.eval_functions import score_context
from eval.rag_example import RAG
from rag.pipeline import score_answer_limited as score_answer

ROOT = Path(__file__).resolve().parents[1]
PILOT_IDS = ("q_009c97884b8dc010", "q_09a24831b90478e8")


def metrics(outputs, reference_outputs):
    context = score_context(reference_outputs["gold_context"], outputs["context"])
    answer = score_answer(reference_outputs["gold_answer"], outputs["answer"])
    return [
        {"key": "context_recall", "score": context["context_recall"]},
        {"key": "context_precision", "score": context["context_precision"]},
        {"key": "answer_correctness", "score": answer["answer_correctness"], "comment": answer["reason"]},
        {"key": "time", "score": outputs["time"]},
    ]


def source_refs(pages):
    return [
        {"doc_id": page["doc_id"], "pdf_page": page["pdf_page"], "role": page.get("context_role")}
        for page in pages
    ]


def trace_step(operation, name, run_type, inputs, outputs):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        with trace(name, run_type=run_type, inputs=inputs(*args, **kwargs)) as span:
            result = operation(*args, **kwargs)
            if span is not None:
                span.end(outputs=outputs(result))
            return result
    return wrapped


def answer_question(inputs):
    system = RAG(ROOT, agentic=True)
    try:
        system.retriever.retrieve = trace_step(
            system.retriever.retrieve, "Первичный гибридный поиск", "chain",
            lambda question, top_k: {"question": question, "top_k": top_k},
            lambda found: {"method": found["method"], "pages": source_refs(found["page_context"])},
        )
        system.generator.decide = trace_step(
            system.generator.decide, "Решение агента", "llm",
            lambda question, pages: {"question": question, "pages": source_refs(pages)},
            lambda decision: decision or {"action": "fallback"},
        )
        system.retriever.search_more = trace_step(
            system.retriever.search_more, "Дополнительный поиск в SQLite", "chain",
            lambda question, query, pages, doc_id: {"query": query, "requested_doc_id": doc_id},
            lambda page: {"page": source_refs([page])[0] if page is not None else None},
        )
        system.generator.generate = trace_step(
            system.generator.generate, "Итоговый ответ", "llm",
            lambda question, pages: {"question": question, "pages": source_refs(pages)},
            lambda answer: {"answer": answer},
        )
        start = perf_counter()
        result = system.rag(inputs["question"])
        return {**result, "time": perf_counter() - start}
    finally:
        system.close()


def pilot_examples():
    with (ROOT / "data/train.jsonl").open(encoding="utf-8") as file:
        rows = {row["question_id"]: row for row in map(json.loads, file)}
    return {
        qid: (
            {"question": rows[qid]["question"], "question_id": qid},
            {key: rows[qid][key] for key in ("gold_context", "gold_answer")},
        )
        for qid in PILOT_IDS
    }


def ensure_pilot_dataset(client, name):
    examples = pilot_examples()
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        existing = {example.id: example for example in client.list_examples(dataset_id=dataset.id, limit=3)}
    else:
        dataset = client.create_dataset(name, description="Два train-примера для проверки решений агента.")
        existing = {}
    ids = {uuid5(dataset.id, qid) for qid in examples}
    if not existing.keys() <= ids:
        raise ValueError("Датасет с этим именем содержит другие примеры. Укажи новое имя.")
    missing = []
    for qid, (inputs, outputs) in examples.items():
        example_id = uuid5(dataset.id, qid)
        if example_id in existing:
            if existing[example_id].inputs != inputs or existing[example_id].outputs != outputs:
                raise ValueError("Данные существующего пилота изменились. Укажи новое имя.")
        else:
            missing.append(ExampleCreate(id=example_id, inputs=inputs, reference_outputs=outputs, split="train"))
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=missing)
    return dataset


def evaluate(dataset_name, *, pilot=False):
    load_dotenv(ROOT / ".env")
    if not os.environ.get("LANGSMITH_API_KEY", "").strip():
        raise SystemExit("Добавь LANGSMITH_API_KEY в локальный .env.")
    client = Client()
    try:
        data = ensure_pilot_dataset(client, dataset_name).name if pilot else dataset_name
        with tracing_context(enabled=True, client=client):
            result = client.evaluate(
                answer_question, data=data, evaluators=[metrics],
                experiment_prefix="hse-rag-agent", max_concurrency=0,
                disable_evaluator_tracing=True,
            )
        return result.experiment_name
    finally:
        client.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name")
    parser.add_argument("--pilot", action="store_true", help="Создать или использовать датасет из двух train-вопросов")
    arguments = parser.parse_args()
    print(evaluate(arguments.dataset_name, pilot=arguments.pilot))
