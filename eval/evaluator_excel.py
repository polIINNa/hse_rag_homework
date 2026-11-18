"""Из корня репозитория: python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
import tempfile
from statistics import mean
from time import perf_counter, sleep

from filelock import FileLock
from openpyxl import Workbook, load_workbook
from requests import RequestException

from eval.eval_functions import score_context
from rag.pipeline import score_answer_limited as score_answer
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


METRICS = ("context_recall", "context_precision", "answer_correctness", "time")


def checkpoint_signature(dataset_path, agentic):
    root = Path(__file__).resolve().parents[1]
    files = [dataset_path, *(root / name for name in (
        "rag/pipeline.py", "rag/retrieval.py", "rag/context.py", "rag/embeddings.py",
        "eval/rag_example.py", "eval/eval_functions.py", "eval/evaluator_excel.py",
        "index/corpus.sqlite", "index/lexical_stems.sqlite",
    ))]
    digest = sha256(str(agentic).encode())
    for path in files:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, document):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(document, file, ensure_ascii=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def evaluate_rows(items, agentic, state, checkpoint_path):
    results = state["results"]
    if len(results) > len(items):
        raise ValueError("Checkpoint contains more results than the dataset.")
    for row, item in zip(results, items):
        if row["question_id"] != item["question_id"]:
            raise ValueError("Checkpoint question order differs from the dataset.")

    if len(results) == len(items):
        return results
    system = RAG(agentic=agentic)
    try:
        for index, item in enumerate(items[len(results):], start=len(results) + 1):
            row = state["pending"]
            if row is None:
                start = perf_counter()
                prediction = system.rag(item["question"])
                elapsed = perf_counter() - start
                context_scores = score_context(item["gold_context"], prediction["context"])
                row = {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
                    "gold_answer": item["gold_answer"],
                    "context": json.dumps(prediction["context"], ensure_ascii=False),
                    "answer": prediction["answer"],
                    **context_scores,
                    "time": elapsed,
                }
                state["pending"] = row
                if checkpoint_path is not None:
                    save_checkpoint(checkpoint_path, state)
            elif row["question_id"] != item["question_id"]:
                raise ValueError("Pending answer does not match the next dataset question.")

            for attempt in range(3):
                try:
                    score = score_answer(item["gold_answer"], row["answer"])
                    break
                except RequestException as error:
                    status = error.response.status_code if error.response is not None else None
                    if attempt == 2 or (status not in (None, 429) and status < 500):
                        raise RuntimeError(f"Judge request failed: {status or type(error).__name__}") from None
                except (ValueError, KeyError):
                    if attempt == 2:
                        raise RuntimeError("Judge returned invalid scores after three attempts.") from None
                sleep(2 ** attempt)
            results.append(dict(row, answer_correctness=score["answer_correctness"]))
            state["pending"] = None
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, state)
            print(json.dumps({"completed": index, "total": len(items), "question_id": row["question_id"],
                              "answer_correctness": score["answer_correctness"]}), flush=True)
    finally:
        system.close()
    return results


def evaluate(dataset_path, output_path=None, *, agentic=False, checkpoint_path=None):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path is not None else dataset_path.with_name(dataset_path.stem + "_results.xlsx")
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("Выберите для отчёта отдельный файл.")
    items = list(read_items(dataset_path))
    if not items:
        raise ValueError("В датасете нет вопросов.")
    if len({item["question_id"] for item in items}) != len(items):
        raise ValueError("В датасете есть повторяющиеся question_id.")

    state = {"signature": None, "results": [], "pending": None}
    if checkpoint_path is None:
        results = evaluate_rows(items, agentic, state, None)
    else:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.resolve() in (dataset_path.resolve(), output_path.resolve()):
            raise ValueError("Выберите для checkpoint отдельный файл.")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(checkpoint_path) + ".lock", timeout=0):
            signature = checkpoint_signature(dataset_path, agentic)
            if checkpoint_path.exists():
                state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if state["signature"] != signature:
                    raise ValueError("Checkpoint belongs to another dataset or version of the pipeline.")
            else:
                state["signature"] = signature
            results = evaluate_rows(items, agentic, state, checkpoint_path)
    return write_report(results, output_path)


def write_report(results, output_path):
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    summary.append(METRICS)
    summary.append([mean(row[key] for row in results) for key in METRICS])
    details = report.create_sheet("По вопросам")
    columns = ("question_id", "question", "gold_context", "gold_answer", "context", "answer", *METRICS)
    details.append(columns)
    for row in results:
        details.append([row[key] for key in columns])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"  # Сохранить как текст, даже если ответ начинается с "=".

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path)
    print(json.dumps({"results": len(results), **{key: mean(r[key] for r in results) for key in METRICS}}), flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    parser.add_argument("--agent", action="store_true", help="Разрешить модели один дополнительный поиск")
    parser.add_argument("--checkpoint", type=Path, help="Сохранять ответы и оценки для продолжения прогона")
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output, agentic=args.agent, checkpoint_path=args.checkpoint))
