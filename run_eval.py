import argparse
import json
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from eval.eval_functions import score_answer, score_context


ROOT = Path(__file__).resolve().parent
COLUMNS = [
    "question_id", "question", "gold_context", "gold_answer", "context", "answer",
    "context_recall", "context_precision", "answer_correctness", "time",
]


def write_excel(rows, path):
    book = Workbook()
    summary = book.active
    summary.title = "Итоги"
    metrics = COLUMNS[-4:]
    summary.append(metrics)
    summary.append([
        mean(row[key] for row in rows) if all(row[key] is not None for row in rows) else None
        for key in metrics
    ])
    if rows[0]["answer_correctness"] is None:
        summary.append(["answer_correctness не оценена без LLM_API_KEY"])
    details = book.create_sheet("По вопросам")
    details.append(COLUMNS)
    for row in rows:
        details.append([
            json.dumps(row[key], ensure_ascii=False) if key in ("gold_context", "context") else row[key]
            for key in COLUMNS
        ])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    for sheet in book:
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="264653")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for column in "ABCDEFGHIJ":
            sheet.column_dimensions[column].width = 22
        sheet.row_dimensions[1].height = 32
    for cell in summary[2]:
        cell.number_format = "0.0000"
    if rows[0]["answer_correctness"] is None:
        summary.merge_cells("A3:D3")
        summary["A3"].alignment = Alignment(wrap_text=True, vertical="top")
        summary.row_dimensions[3].height = 32
    details.freeze_panes = "C2"
    details.auto_filter.ref = details.dimensions
    for column, width in {"B": 60, "D": 80, "F": 100}.items():
        details.column_dimensions[column].width = width
    for row in details.iter_rows(min_row=2):
        lines = max(
            sum(1 + len(line) // int(details.column_dimensions[cell.column_letter].width)
                for line in str(cell.value or "").split("\n"))
            for cell in row
        )
        details.row_dimensions[row[0].row].height = min(409, 14 * lines + 8)
        for cell in row:
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in details.iter_rows(min_row=2, min_col=7, max_col=10):
        for cell in row:
            cell.number_format = "0.0000"
    book.save(path)


def evaluate(no_judge=False, resume=False, limit=None, judge_only=False):
    folder = ROOT / "results"
    folder.mkdir(exist_ok=True)
    log = folder / "test_predictions.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if (resume or judge_only) and log.exists() else []
    if judge_only:
        for row in rows:
            if row["answer_correctness"] is None:
                row["answer_correctness"] = score_answer(row["gold_answer"], row["answer"])["answer_correctness"]
                log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    else:
        from rag import RAG
        system = RAG()
        items = [json.loads(line) for line in (ROOT / "data/test.jsonl").read_text().splitlines()]
        done = {row["question_id"] for row in rows}
        with log.open("a" if resume else "w") as file, (folder / "agent_trace.jsonl").open("a" if resume else "w") as traces:
            for item in items[:limit]:
                if item["question_id"] in done:
                    continue
                start = perf_counter()
                prediction = system.rag(item["question"])
                elapsed = perf_counter() - start
                scores = score_context(item["gold_context"], prediction["context"])
                correctness = None if no_judge else score_answer(item["gold_answer"], prediction["answer"])["answer_correctness"]
                row = {
                    **{key: item[key] for key in COLUMNS[:4]}, **prediction, **scores,
                    "answer_correctness": correctness, "time": elapsed,
                }
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                traces.write(json.dumps({"question_id": item["question_id"], **system.trace}, ensure_ascii=False) + "\n")
                traces.flush()
                if len(rows) % 5 == 0:
                    write_excel(rows, folder / "test_results.xlsx")
                    print(len(rows), "/", len(items), "recall", round(mean(r["context_recall"] for r in rows), 4),
                          "precision", round(mean(r["context_precision"] for r in rows), 4), flush=True)
    write_excel(rows, folder / "test_results.xlsx")
    if not limit:
        readme = (ROOT / "README.md").read_text()
        for key in COLUMNS[-4:]:
            value = f"{mean(row[key] for row in rows):.4f}" if all(row[key] is not None for row in rows) else "не оценена без ключа преподавателя"
            readme = re.sub(rf"^{key} - .*$", f"{key} - {value}", readme, flags=re.M)
        (ROOT / "README.md").write_text(readme)
    print({key: mean(row[key] for row in rows) if all(row[key] is not None for row in rows) else None for key in COLUMNS[-4:]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluate(args.no_judge, args.resume, args.limit, args.judge_only)
