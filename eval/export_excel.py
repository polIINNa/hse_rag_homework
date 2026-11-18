"""Экспорт завершённого прогона в Excel в формате задания."""
import argparse
import json
from pathlib import Path
from statistics import mean

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo

from eval.run import METRICS

COLUMNS = ["question_id", "question", "gold_context", "gold_answer", "context", "answer", *METRICS]


def export(source, output):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    summary = json.loads(source.with_suffix(".summary.json").read_text())
    if len(rows) != summary["questions"] or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Прогон неполный или содержит дубликаты")
    if summary["split"] == "test" and len(rows) != 141:
        raise ValueError("В итоговом Excel должно быть 141 вопрос")
    for metric in METRICS:
        if abs(mean(r[metric] for r in rows) - summary[metric]) > 1e-10:
            raise ValueError(f"Не совпала сводка: {metric}")
    wb = Workbook()
    totals = wb.active
    totals.title = "Итоги"
    totals.append(["RAG по финансовой отчётности"])
    totals.append(["Иванов Степан Андреевич"])
    totals.append(["Вопросов", len(rows)])
    totals.append(["Метрика", "Среднее"])
    for metric in METRICS:
        totals.append([metric, summary[metric]])
    details = wb.create_sheet("По вопросам")
    details.append(COLUMNS)
    for row in rows:
        values = [json.dumps(row[key], ensure_ascii=False) if key in {"context", "gold_context"} else row[key] for key in COLUMNS]
        details.append(values)
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    header_fill = PatternFill("solid", fgColor="203864")
    for sheet, header in [(totals, 4), (details, 1)]:
        sheet.sheet_view.showGridLines = False
        for row in sheet:
            for cell in row:
                cell.font = Font(name="Arial", size=10)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for cell in sheet[header]:
            cell.fill = header_fill
            cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    totals["A1"].font = Font(name="Arial", size=14, bold=True)
    totals.column_dimensions["A"].width = 42
    totals.column_dimensions["B"].width = 20
    for i in range(5, 9):
        totals.cell(i, 2).number_format = "0.0000"
    widths = dict(zip("ABCDEFGHIJ", [25, 65, 42, 65, 42, 85, 22, 22, 24, 16]))
    for column, width in widths.items():
        details.column_dimensions[column].width = width
    details.freeze_panes = "C2"
    details.row_dimensions[1].height = 30
    for i in range(2, len(rows) + 2):
        details.row_dimensions[i].height = min(300, max(60, 15 * max(len(str(details.cell(i, c).value)) // 60 + 1 for c in (2, 4, 6))))
        for j in range(7, 9):
            details.cell(i, j).number_format = "0.0000"
        details.cell(i, 9).number_format = "0"
        details.cell(i, 10).number_format = "0.00"
    table = Table(displayName="TestResults", ref=f"A1:J{len(rows)+1}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    details.add_table(table)
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test_results.xlsx"))
    args = parser.parse_args()
    export(args.results, args.output)

