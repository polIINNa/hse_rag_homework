"""Запуск eval с сохранением прогресса, троттлингом, ретраями и LangSmith-трассировкой.

Использование (из корня репозитория):
  python -m eval.run_eval_resumable data/test.jsonl --output test_results.xlsx \
      --judge-rpm 10

Зачем: evaluator_excel.py не сохраняет промежуточный прогресс; при ошибке
(например, 429 от LLM-судьи) весь прогон теряется. Этот скрипт:
- ограничивает частоту вызовов судьи (по умолчанию 10 запросов/мин),
- повторяет запрос при 429/временных сетевых ошибках,
- сохраняет прогресс в JSONL и умеет продолжать с места остановки,
- при включённом LangSmith отправляет трейсы RAG и метрики (feedback) в проект.

LangSmith (доп. балл): установите переменные окружения и запустите eval как обычно:
  $env:LANGSMITH_TRACING="true"
  $env:LANGSMITH_API_KEY="<ключ>"
  $env:LANGSMITH_PROJECT="hse-rag-homework"   # опционально
Трейсы вызовов RAG и метрики context_recall/context_precision/answer_correctness/time
будут доступны в LangSmith.
"""

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook, load_workbook

from eval.eval_functions import score_answer, score_context
from eval.rag_example import RAG

# Опциональная интеграция с LangSmith: если не настроено/недоступно — просто no-op.
try:
    from langsmith import traceable  # type: ignore
    from langsmith.run_helpers import get_current_run_tree  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*dargs, **dkwargs):
        def deco(fn):
            return fn

        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]
        return deco

    def get_current_run_tree():
        return None


def _log_feedback(row):
    """Отправляет метрики в LangSmith как feedback текущего трейса (если включено)."""
    try:
        rt = get_current_run_tree()
        if not rt:
            return
        client = getattr(rt, "client", None)
        if client is None:
            from langsmith import Client  # type: ignore

            client = Client()
        run_id = getattr(rt, "id", None)
        if not run_id:
            return
        for key in ("context_recall", "context_precision", "answer_correctness", "time"):
            val = row.get(key)
            if val is None:
                continue
            try:
                client.create_feedback(run_id=run_id, key=key, score=float(val))
            except Exception:
                pass
    except Exception:
        pass


def read_items(dataset_path):
    dataset_path = Path(dataset_path)
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
        for values in rows:
            if all(value is None for value in values):
                continue
            item = dict(zip(headers, values))
            item["gold_context"] = json.loads(item["gold_context"])
            yield item
    finally:
        source.close()


def call_judge_with_retry(gold_answer, answer, retries=12, base_sleep=10, min_interval=6.5):
    """Вызов судьи с троттлингом и ретраями.

    min_interval: минимальная пауза между вызовами (сек). Для лимита 10 RPM безопасно >= 6с.
    При 429 ждём дольше (экспоненциально), при сетевых ошибках тоже повторяем.
    """
    last_err = None
    for attempt in range(retries):
        try:
            if min_interval > 0:
                time.sleep(min_interval)
            return score_answer(gold_answer, answer)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            is_rate = "429" in msg or "Too Many Requests" in msg
            is_temp = is_rate or "timed out" in msg.lower() or "timeout" in msg.lower() or "Connection" in msg
            if is_temp and attempt < retries - 1:
                sleep_s = max(base_sleep * (2 ** min(attempt, 4)), 60 if is_rate else base_sleep)
                print(f"[judge retry {attempt + 1}/{retries}] {msg}")
                print(f"  wait {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            # даже при фатальной ошибке не теряем прогресс — пробрасываем наверх,
            # но всё, что уже записано в progress JSONL, сохранено
            raise
    raise last_err


def load_done(progress_path):
    done = {}
    if not progress_path.exists():
        return done
    with progress_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["question_id"]] = row
    return done


def write_excel(results, output_path):
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
                cell.data_type = "s"
    report.save(output_path)


@traceable(run_type="chain", name="evaluate_question")
def _evaluate_one(system, item, min_interval):
    """Оценивает один вопрос: RAG + метрики контекста + судья.
    Обёрнуто в traceable, чтобы в LangSmith был трейс и можно было прикрепить метрики."""
    start = perf_counter()
    prediction = system.rag(item["question"])
    elapsed = perf_counter() - start
    context_scores = score_context(item["gold_context"], prediction["context"])
    answer_score = call_judge_with_retry(
        item["gold_answer"], prediction["answer"], min_interval=min_interval
    )

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
    _log_feedback(row)
    return row


def evaluate(dataset_path, output_path=None, progress_path=None, judge_rpm=10):
    dataset_path = Path(dataset_path)
    if output_path is None:
        output_path = dataset_path.with_name(dataset_path.stem + "_results.xlsx")
    output_path = Path(output_path)
    if progress_path is None:
        progress_path = output_path.with_suffix(".progress.jsonl")
    progress_path = Path(progress_path)

    # 10 RPM => >= 6s между запросами; берём небольшой запас
    min_interval = 60.0 / max(judge_rpm, 1) + 0.5 if judge_rpm else 0

    done = load_done(progress_path)
    system = RAG()
    items = list(read_items(dataset_path))

    try:
        with progress_path.open("a", encoding="utf-8") as prog:
            for i, item in enumerate(items, 1):
                qid = item["question_id"]
                if qid in done:
                    print(f"[{i}/{len(items)}] skip {qid} (cached)")
                    continue

                row = _evaluate_one(system, item, min_interval)
                prog.write(json.dumps(row, ensure_ascii=False) + "\n")
                prog.flush()
                done[qid] = row
                print(f"[{i}/{len(items)}] done {qid}")
    except Exception:
        print(f"Прогресс сохранён в {progress_path}. Повторите запуск — он продолжится с места остановки.")
        raise

    if len(done) != len(items):
        raise ValueError("Не все вопросы обработаны.")

    # preserve dataset order
    results = [done[item["question_id"]] for item in items]
    write_excel(results, output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument(
        "--judge-rpm",
        type=int,
        default=10,
        help="Лимит запросов к судье в минуту (по умолчанию 10). 0 — без троттлинга.",
    )
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output, args.progress, args.judge_rpm))
