"""Eval нашего RAG с записью Excel (те же колонки и листы, что у eval/evaluator_excel.py).

Из корня репозитория:
    python -m eval.run_eval data/test.jsonl --output results/test_results.xlsx

Отличия от шаблонного evaluator_excel: печатает прогресс, повторяет запрос к судье при сбое сети и
сохраняет каждый ответ в outputs/<имя>.jsonl — если прогон оборвался, повторный запуск продолжит с места обрыва
(флаг --fresh начинает заново).
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook

from eval.eval_functions import score_answer, score_context
from eval.evaluator_excel import read_items
from eval.rag_example import RAG

COLUMNS = ["question_id", "question", "gold_context", "gold_answer", "context", "answer",
           "context_recall", "context_precision", "answer_correctness", "time"]
METRICS = ("context_recall", "context_precision", "answer_correctness", "time")


def judge(gold_answer, answer, attempts=4):
    for attempt in range(attempts):
        try:
            return score_answer(gold_answer, answer)
        except Exception as error:  # сеть/лимиты/невалидный JSON судьи — пробуем ещё раз
            if attempt == attempts - 1:
                raise
            print(f"   судья: {error!r}, повтор через {2 ** attempt} c")
            time.sleep(2 ** attempt)


def write_excel(results, output_path):
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    summary.append(METRICS)
    summary.append([mean(row[key] for row in results) for key in METRICS])
    details = report.create_sheet("По вопросам")
    details.append(COLUMNS)
    for row in results:
        details.append([row[key] for key in COLUMNS])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path)


def update_readme(results, mode, output_path):
    """Вписать четыре средних по test в README между маркерами METRICS."""
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- METRICS:START -->", "<!-- METRICS:END -->"
    if start not in text or end not in text:
        return
    means = {key: mean(row[key] for row in results) for key in METRICS}
    table = (f"{start}\n| context_recall | context_precision | answer_correctness | time, с |\n"
             f"|---:|---:|---:|---:|\n"
             f"| {means['context_recall']:.4f} | {means['context_precision']:.4f} | "
             f"{means['answer_correctness']:.4f} | {means['time']:.2f} |\n\n"
             f"Режим `{mode}`, {len(results)} вопросов test, отчёт: [`{output_path.as_posix()}`]({output_path.as_posix()}).\n{end}")
    before, rest = text.split(start, 1)
    readme.write_text(before + table + rest.split(end, 1)[1], encoding="utf-8")


def evaluate(dataset_path, output_path, mode=None, limit=None, fresh=False):
    dataset_path, output_path = Path(dataset_path), Path(output_path)
    items = list(read_items(dataset_path))[:limit]
    checkpoint = Path("outputs") / f"{output_path.stem}.jsonl"
    checkpoint.parent.mkdir(exist_ok=True)
    if fresh and checkpoint.exists():
        checkpoint.unlink()
    done = {}
    if checkpoint.exists():
        for line in checkpoint.open(encoding="utf-8"):
            row = json.loads(line)
            done[row["question_id"]] = row
        print(f"Продолжаю: уже посчитано {len(done)} из {len(items)}")

    system = RAG(mode)
    print(f"Режим RAG: {system.mode}, вопросов: {len(items)}")
    with checkpoint.open("a", encoding="utf-8") as out:
        for n, item in enumerate(items, 1):
            if item["question_id"] in done:
                continue
            start = perf_counter()
            prediction = system.rag(item["question"])
            elapsed = perf_counter() - start
            context_scores = score_context(item["gold_context"], prediction["context"])
            answer_score = judge(item["gold_answer"], prediction["answer"])
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
                "judge_reason": answer_score["reason"],
            }
            done[row["question_id"]] = row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{n}/{len(items)}] AC={row['answer_correctness']} R={row['context_recall']:.2f} "
                  f"P={row['context_precision']:.2f} t={elapsed:.1f}s")

    results = [done[item["question_id"]] for item in items]
    write_excel(results, output_path)
    print("\nСредние по датасету:")
    for key in METRICS:
        print(f"  {key:<20} {mean(row[key] for row in results):.4f}")
    print(f"Excel: {output_path}")
    if dataset_path.name == "test.jsonl" and limit is None:
        update_readme(results, system.mode, output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, default=Path("results/test_results.xlsx"))
    parser.add_argument("--mode", choices=["agentic", "simple"], help="по умолчанию RAG_MODE из .env (agentic)")
    parser.add_argument("--limit", type=int, help="только первые N вопросов (для отладки)")
    parser.add_argument("--fresh", action="store_true", help="не продолжать прошлый прогон, начать заново")
    args = parser.parse_args()
    evaluate(args.dataset_path, args.output, args.mode, args.limit, args.fresh)
