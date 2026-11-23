"""Из корня репозитория: python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx"""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook, load_workbook

from eval.eval_functions import score_answer, score_context
from rag.models import wait_for_chat_capacity
from rag.pipeline import RAG


def read_items(dataset_path):
    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
        return

    source = load_workbook(dataset_path, read_only=True, data_only=True)
    try:
        rows = source.worksheets[0].values
        headers = next(rows, ())
        required = {"question_id", "question", "gold_context", "gold_answer"}
        if not required.issubset(headers):
            raise ValueError(
                f"В первом листе не хватает колонок: {required.difference(headers)}"
            )
        for values in rows:
            if all(value is None for value in values):
                continue
            item = dict(zip(headers, values))
            item["gold_context"] = json.loads(item["gold_context"])
            yield item
    finally:
        source.close()


def evaluate(dataset_path, output_path=None):
    dataset_path = Path(dataset_path)
    output_path = (
        Path(output_path)
        if output_path is not None
        else dataset_path.with_name(dataset_path.stem + "_results.xlsx")
    )
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("Выберите для отчёта отдельный файл.")
    system = RAG()
    results = []
    for item in read_items(dataset_path):
        print(f"[{len(results) + 1}] {item['question_id']}: ", end="", flush=True)
        wait_for_chat_capacity(2)
        start = perf_counter()
        prediction = system.rag(item["question"])
        elapsed = perf_counter() - start
        context_scores = score_context(item["gold_context"], prediction["context"])
        answer_score = score_answer(item["gold_answer"], prediction["answer"])
        results.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
                "gold_answer": item["gold_answer"],
                "context": json.dumps(prediction["context"], ensure_ascii=False),
                "answer": prediction["answer"],
                **context_scores,
                "answer_correctness": answer_score["answer_correctness"],
                "time": elapsed,
            }
        )
        print(f"{elapsed:.2f} с", flush=True)

    if not results:
        raise ValueError("В датасете нет вопросов.")
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    metrics = ("context_recall", "context_precision", "answer_correctness", "time")
    summary.append(metrics)
    summary.append([mean(row[key] for row in results) for key in metrics])
    details = report.create_sheet("По вопросам")
    columns = list(results[0])
    details.append(columns)
    for row in results:
        details.append([row[key] for key in columns])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                # Сохранить как текст, даже если ответ начинается с "=".
                cell.data_type = "s"

    report.save(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output))
