"""Проверить допуск ±5% между двумя независимыми полными прогонами."""
import argparse
from pathlib import Path

from openpyxl import load_workbook

from evaluate import METRICS


def read(path):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book["По вопросам"].values)
        ids = [row[0] for row in rows[1:]]
        if len(ids) != 141 or len(set(ids)) != 141:
            raise ValueError(f"{path}: ожидаются все 141 вопрос")
        columns = {name: i for i, name in enumerate(rows[0])}
        return ids, {name: sum(float(r[columns[name]]) for r in rows[1:]) / 141 for name in METRICS}
    finally:
        book.close()


def compare(first, second):
    ids, a = read(first)
    ids2, b = read(second)
    if ids != ids2:
        raise ValueError("Прогоны относятся к разным вопросам или разному порядку")
    passed = True
    for name in METRICS:
        tolerance = abs(a[name]) * 0.05
        ok = abs(a[name] - b[name]) <= tolerance + 1e-12
        print(f"{name}: {a[name]:.6f} -> {b[name]:.6f}; {'PASS' if ok else 'FAIL'}")
        passed &= ok
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if compare(args.first, args.second) else 1)
