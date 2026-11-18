"""Из корня репозитория: python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from statistics import mean
from textwrap import wrap
from time import perf_counter

import requests
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from eval.eval_functions import score_answer, score_context
from eval.rag_example import RAG
from rag_agent import retry
from rag_corpus import ROOT, read_jsonl


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


def evaluate_item(system, item):
    start = perf_counter()
    prediction = system.rag(item["question"])
    elapsed = perf_counter() - start
    context_scores = score_context(item["gold_context"], prediction["context"])
    answer_score = retry(lambda: score_answer(item["gold_answer"], prediction["answer"]))
    return result_row(item, prediction, {**context_scores, **answer_score, "time": elapsed})


def result_row(item, prediction, scores):
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
        "gold_answer": item["gold_answer"],
        "context": json.dumps(prediction["context"], ensure_ascii=False),
        "answer": prediction["answer"],
        "context_recall": scores["context_recall"],
        "context_precision": scores["context_precision"],
        "answer_correctness": scores["answer_correctness"],
        "time": scores["time"],
    }


def save_report(results, output_path):
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    metrics = ("context_recall", "context_precision", "answer_correctness", "time")
    summary.append(metrics)
    summary.append([mean(row[key] for row in results) for key in metrics])
    for cell in summary[2]:
        cell.number_format = "0.0000"
        cell.font = Font(name="Arial", size=11)
    details = report.create_sheet("По вопросам")
    columns = list(results[0])
    widths = (22, 58, 44, 58, 44, 65, 19, 19, 22, 12)
    details.append(columns)
    for row in results:
        details.append([row[key] for key in columns])
        cells = details[details.max_row]
        lines = 1
        for cell, width in zip(cells, widths):
            if isinstance(cell.value, str):
                cell.data_type = "s"  # Сохранить как текст, даже если ответ начинается с "=".
                lines = max(lines, sum(max(1, len(wrap(line, width - 3))) for line in cell.value.splitlines()))
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        details.row_dimensions[details.max_row].height = min(360, max(32, 14 * lines + 6))
        for cell in (*cells[6:8], cells[9]):
            cell.number_format = "0.0000"
    for sheet, column_widths in (
        (summary, (24, 24, 24, 14)),
        (details, widths),
    ):
        sheet.freeze_panes = "A2"
        sheet.row_dimensions[1].height = 24
        for number, width in enumerate(column_widths, 1):
            sheet.column_dimensions[get_column_letter(number)].width = width
        for cell in sheet[1]:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="243D59")
            cell.alignment = Alignment(vertical="center")
    details.auto_filter.ref = details.dimensions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.xlsx")
    report.save(tmp)
    tmp.replace(output_path)


def checkpoint_results(checkpoint, dataset_path, workers):
    digest = sha256(dataset_path.read_bytes())
    for relative_path in (
        "rag_agent.py", "rag_corpus.py", "rag_retrieval.py", "eval/eval_functions.py",
        "eval/evaluator_excel.py", "data/documents.jsonl", "requirements.txt",
    ):
        digest.update((ROOT / relative_path).read_bytes())
    run = {"signature": digest.hexdigest(), "workers": workers}
    if checkpoint.exists():
        records = read_jsonl(checkpoint)
        if not records or records[0] != run:
            raise ValueError(f"Checkpoint belongs to a different evaluation: {checkpoint}")
        return {row["question_id"]: row for row in records[1:]}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(run) + "\n", encoding="utf-8")
    return {}


def evaluate(dataset_path, output_path=None, workers=1, checkpoint=None):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path else dataset_path.with_name(dataset_path.stem + "_results.xlsx")
    if len({path.resolve() for path in (dataset_path, output_path, checkpoint) if path is not None}) != (
        3 if checkpoint is not None else 2
    ):
        raise ValueError("Dataset, report, and checkpoint must use separate paths.")
    items = list(read_items(dataset_path))
    if not items:
        raise ValueError("В датасете нет вопросов.")
    if len({item["question_id"] for item in items}) != len(items):
        raise ValueError("Question IDs must be unique.")
    results = checkpoint_results(checkpoint, dataset_path, workers) if checkpoint else {}
    pending = [item for item in items if item["question_id"] not in results]
    failures = []
    if pending:
        system = RAG()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(evaluate_item, system, item): item["question_id"] for item in pending}
            for task in as_completed(tasks):
                question_id = tasks[task]
                try:
                    row = task.result()
                except (ValueError, requests.RequestException) as error:
                    print(f"{question_id}: {error}", flush=True)
                    failures.append(question_id)
                    continue
                results[question_id] = row
                if checkpoint:
                    with checkpoint.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(results) % 10 == 0 or len(results) == len(items):
                    print(f"Evaluated {len(results)}/{len(items)} questions", flush=True)
    if failures:
        raise RuntimeError(f"Evaluation failed for {len(failures)} questions: {', '.join(map(str, failures))}")
    save_report([results[item["question_id"]] for item in items], output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    print(evaluate(args.dataset_path, args.output, args.workers, args.checkpoint))
