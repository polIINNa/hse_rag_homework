"""Eval на test с продолжением после сбоя. Из корня репозитория:

    python -m eval.evaluator_resumable data/test.jsonl --output test_results.xlsx

Считает то же, что eval/evaluator_excel.py: те же score_context и score_answer (модель и промпт судьи не
менялись), тот же замер time вокруг ответа RAG, тот же порядок вызовов и формат Excel. Отличия только в
надёжности: каждый готовый вопрос сохраняется в outputs/<имя отчёта>.progress.jsonl, повторный запуск
продолжает с места; упавший вызов судьи (сеть, 429) повторяется.
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from time import perf_counter

import requests
from openpyxl import Workbook

from eval.eval_functions import score_answer, score_context
from eval.evaluator_excel import read_items
from eval.rag_example import RAG

METRICS = ("context_recall", "context_precision", "answer_correctness", "time")


def judge(system, gold_answer, answer):
    """Судья вызывается сразу, как в evaluator_excel.py: место под него RAG резервирует в лимите сам.
    Повтор — только если вызов упал (сеть или 429), чтобы время RAG считалось так же, как в штатных eval."""
    window = system.agent.llm.window
    for attempt in range(8):
        try:
            return score_answer(gold_answer, answer)
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 429 or attempt == 7:
                raise
            window.penalize()
            window.acquire()
        except requests.exceptions.RequestException:
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 60))


def write_report(results, output_path):
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    summary.append(METRICS)
    summary.append([mean(row[key] for row in results) for key in METRICS])
    details = report.create_sheet("По вопросам")
    columns = list(results[0])
    details.append(columns)
    for row in results:
        details.append([row[key] for key in columns])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    report.save(output_path)


def evaluate(dataset_path, output_path):
    dataset_path, output_path = Path(dataset_path), Path(output_path)
    progress = Path("outputs") / (output_path.stem + ".progress.jsonl")
    progress.parent.mkdir(exist_ok=True)
    done = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["question_id"]] = row
    items = list(read_items(dataset_path))
    print(f"Вопросов: {len(items)}, уже готово: {len(done)}", flush=True)
    system = RAG()
    for n, item in enumerate(items, 1):
        if item["question_id"] in done:
            continue
        start = perf_counter()
        prediction = system.agent.run(item["question"])
        elapsed = perf_counter() - start
        context_scores = score_context(item["gold_context"], prediction["context"])
        answer_score = judge(system, item["gold_answer"], prediction["answer"])
        row = {
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
        done[item["question_id"]] = row
        with progress.open("a", encoding="utf-8") as target:
            target.write(json.dumps({**row, "judge_reason": answer_score["reason"],
                                     "searches": prediction["searches"], "llm_wait": prediction["llm_wait"]},
                                    ensure_ascii=False) + "\n")
        print(f"{len(done)}/{len(items)} R={row['context_recall']:.2f} P={row['context_precision']:.2f} "
              f"AC={row['answer_correctness']} t={elapsed:.1f}s", flush=True)

    keep = ("question_id", "question", "gold_context", "gold_answer", "context", "answer", *METRICS)
    results = [{k: done[item["question_id"]][k] for k in keep} for item in items]
    write_report(results, output_path)
    print({k: round(mean(r[k] for r in results), 4) for k in METRICS})
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, default=Path("test_results.xlsx"))
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output))
