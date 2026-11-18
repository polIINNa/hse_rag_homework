"""Из корня репозитория: python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx"""

import argparse
from itertools import islice
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
from time import perf_counter

from openpyxl import Workbook, load_workbook
from requests import HTTPError

from eval.eval_functions import score_context
from eval.judge import score_answer
from eval.rag_example import RAG


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
            raise ValueError(f"В первом листе не хватает колонок: {required.difference(headers)}")
        for values in rows:
            if all(value is None for value in values):
                continue
            item = dict(zip(headers, values))
            item["gold_context"] = json.loads(item["gold_context"])
            yield item
    finally:
        source.close()


def evaluate(dataset_path, output_path=None, *, limit=None):
    dataset_path = Path(dataset_path)
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("Set the instructor-issued LLM_API_KEY to run the official answer judge.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output_path = Path(output_path) if output_path else dataset_path.with_name(dataset_path.stem + "_results.xlsx")
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("Выберите для отчёта отдельный файл.")
    items = list(islice(read_items(dataset_path), limit))
    if not items:
        raise ValueError("В датасете нет вопросов.")
    system = RAG()
    results = []
    for count, item in enumerate(items, 1):
        start = perf_counter()
        prediction = system.rag(item["question"])
        elapsed = perf_counter() - start
        context_scores = score_context(item["gold_context"], prediction["context"])
        answer_score = score_answer(item["gold_answer"], prediction["answer"])
        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
            "gold_answer": item["gold_answer"],
            "context": json.dumps(prediction["context"], ensure_ascii=False),
            "answer": prediction["answer"],
            **context_scores,
            "answer_correctness": answer_score["answer_correctness"],
            "time": elapsed,
        })
        if count == 1 or count % 10 == 0 or count == len(items):
            print(f"Scored {count}/{len(items)} questions", flush=True)

    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    metrics = ("context_recall", "context_precision", "answer_correctness", "time")
    averages = [mean(row[key] for row in results) for key in metrics]
    summary.append(metrics)
    summary.append(averages)
    details = report.create_sheet("По вопросам")
    columns = list(results[0])
    details.append(columns)
    for row in results:
        details.append([row[key] for key in columns])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"  # Сохранить как текст, даже если ответ начинается с "=".

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".xlsx", delete=False) as target:
        temporary_path = Path(target.name)
    try:
        report.save(temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    for metric, value in zip(metrics, averages):
        print(f"{metric}: {value:.6f}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    parser.add_argument("--limit", type=int, help="Evaluate a limited number of rows for development")
    args = parser.parse_args()
    try:
        report_path = evaluate(args.dataset_path, args.output, limit=args.limit)
    except HTTPError as error:
        if error.response is not None and error.response.status_code == 401:
            parser.exit(
                1,
                "HTTP 401: The answer judge rejected LLM_API_KEY. "
                "Set it to the instructor-issued judge key and retry.\n",
            )
        raise
    print(report_path)
