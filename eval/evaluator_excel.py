"""Из корня репозитория: python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
from time import perf_counter

from dotenv import load_dotenv
from openpyxl import Workbook, load_workbook

from build_index import ROOT
from eval.eval_functions import score_answer, score_context
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


def save_checkpoint(path, dataset_hash, results):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump({"dataset_hash": dataset_hash, "results": results}, stream, ensure_ascii=False)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def evaluate(dataset_path, output_path=None, *, checkpoint_path=None):
    load_dotenv(ROOT / ".env")
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path is not None else dataset_path.with_name(
        dataset_path.stem + "_results.xlsx"
    )
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("Выберите для отчёта отдельный файл.")

    items = list(read_items(dataset_path))
    if not items:
        raise ValueError("В датасете нет вопросов.")

    results = []
    dataset_hash = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.resolve() in (dataset_path.resolve(), output_path.resolve()):
            raise ValueError("Выберите отдельный файл для сохранения прогресса.")
        dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint["dataset_hash"] != dataset_hash:
                raise ValueError("Сохранённый прогресс относится к другому датасету.")
            results = checkpoint["results"]
            if [row["question_id"] for row in results] != [
                item["question_id"] for item in items[:len(results)]
            ]:
                raise ValueError("Порядок вопросов в сохранённом прогрессе не совпадает с датасетом.")
            completed = sum(row["answer_correctness"] is not None for row in results)
            print(f"Восстановлено оценённых вопросов: {completed}", flush=True)

    system = None
    for number, item in enumerate(items, 1):
        if number <= len(results):
            row = results[number - 1]
            if row["answer_correctness"] is not None:
                continue
        else:
            if system is None:
                system = RAG()
            start = perf_counter()
            prediction = system.rag(item["question"])
            elapsed = perf_counter() - start
            row = {
                "question_id": item["question_id"],
                "question": item["question"],
                "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
                "gold_answer": item["gold_answer"],
                "context": json.dumps(prediction["context"], ensure_ascii=False),
                "answer": prediction["answer"],
                **score_context(item["gold_context"], prediction["context"]),
                "answer_correctness": None,
                "time": elapsed,
            }
            results.append(row)
            save_checkpoint(checkpoint_path, dataset_hash, results)
        answer_score = score_answer(row["gold_answer"], row["answer"])
        row["answer_correctness"] = answer_score["answer_correctness"]
        save_checkpoint(checkpoint_path, dataset_hash, results)
        if number == 1 or number % 10 == 0:
            print(f"Оценено вопросов: {number}", flush=True)

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
                cell.data_type = "s"  # Сохранить как текст, даже если ответ начинается с "=".

    report.save(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    parser.add_argument("--checkpoint", type=Path, help="Файл прогресса для продолжения после сбоя")
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output, checkpoint_path=args.checkpoint))
