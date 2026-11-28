"""Итоговая оценка за один прогон: эксперимент в LangSmith и Excel из его результатов.

Из корня репозитория: python -m solution.evaluate_langsmith hw-rag-test --output test_results.xlsx
Датасет сначала загрузите: python -m eval.upload_langsmith data/test.jsonl hw-rag-test

Эксперимент — как eval/evaluator_langsmith.py: те же метрики (функция metrics), модель и промпт судьи не меняются,
но судья соблюдает лимит LLM и повторяет запрос, а у эксперимента есть название варианта и метаданные.
Excel — в формате eval/evaluator_excel.py: листы «Итоги» и «По вопросам» с теми же колонками.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from langsmith import Client
from openpyxl import Workbook

import eval.evaluator_langsmith as evaluator_langsmith
from eval.rag_example import CANDIDATES, DENSE_WEIGHT, MAX_SEARCHES, RAG, SimpleRAG
from solution.generate import LLM_MODEL, llm_tokens
from solution.judge import judge

METRICS = ("context_recall", "context_precision", "answer_correctness", "time")


def result_rows(result):
    """Строки для Excel из результатов эксперимента. Вопросы, где упал RAG или судья, — отдельным списком."""
    rows, failed = [], []
    for item in result:
        run, example = item["run"], item["example"]
        scores = {r.key: r.score for r in item["evaluation_results"]["results"] if r.score is not None}
        question_id = (example.metadata or {}).get("question_id")
        if run.error or not run.outputs or any(key not in scores for key in METRICS):
            failed.append(question_id)
            continue
        rows.append({
            "question_id": question_id,
            "question": example.inputs["question"],
            "gold_context": json.dumps(example.outputs["gold_context"], ensure_ascii=False),
            "gold_answer": example.outputs["gold_answer"],
            "context": json.dumps(run.outputs["context"], ensure_ascii=False),
            "answer": run.outputs["answer"],
            **{key: scores[key] for key in METRICS},
        })
    return rows, failed


def write_excel(rows, path):
    """Тот же формат, что пишет eval/evaluator_excel.py."""
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    summary.append(METRICS)
    summary.append([mean(row[key] for row in rows) for key in METRICS])
    details = report.create_sheet("По вопросам")
    details.append(list(rows[0]))
    for row in rows:
        details.append(list(row.values()))
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"  # сохранить как текст, даже если ответ начинается с "="
    report.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_name")
    parser.add_argument("--variant", choices=["agentic", "simple"], default="agentic")
    parser.add_argument("--output", type=Path, help="Excel с результатами, например test_results.xlsx")
    args = parser.parse_args()

    evaluator_langsmith.score_answer = judge  # metrics() вызывает score_answer из своего модуля
    system = RAG() if args.variant == "agentic" else SimpleRAG()

    def answer_question(inputs):
        start = perf_counter()
        result = system.rag(inputs["question"])
        return {**result, "time": perf_counter() - start}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = Client().evaluate(
        answer_question,
        data=args.dataset_name,
        evaluators=[evaluator_langsmith.metrics],
        experiment_prefix=f"{args.variant}-{stamp}",
        metadata={"dataset": args.dataset_name, "variant": args.variant, "k": system.k, "llm": LLM_MODEL,
                  "candidates": CANDIDATES, "dense_weight": DENSE_WEIGHT,
                  "max_searches": MAX_SEARCHES if args.variant == "agentic" else None},
        max_concurrency=0,  # вопросы по одному: у LLM лимит 10 запросов в минуту
    )
    print("Эксперимент:", result.experiment_name)

    rows, failed = result_rows(result)
    print(f"Вопросов: {len(rows) + len(failed)}, без ошибок: {len(rows)}. "
          f"Токены LLM без судьи: {llm_tokens['input'] + llm_tokens['output']}.")
    if failed:
        # Неполный Excel дал бы неверные средние: перезапустите прогон (например, после сброса дневного лимита)
        raise SystemExit(f"Ошибки в {len(failed)} вопросах, Excel не записан: {failed[:10]}")
    if args.output:
        write_excel(rows, args.output)
        print("Excel:", args.output)


if __name__ == "__main__":
    main()
