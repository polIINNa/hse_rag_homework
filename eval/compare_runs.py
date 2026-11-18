"""Сравнить четыре средних двух полных прогонов test с допуском 5%."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from eval.run import METRICS
from rag.config import DATA


def load_run(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    expected = {item["question_id"]: item for item in map(json.loads, (DATA / "test.jsonl").read_text().splitlines())}
    if len(rows) != len(expected) or {row["question_id"] for row in rows} != set(expected):
        raise ValueError(f"{path}: нужны все вопросы test без дубликатов")
    if summary["split"] != "test" or summary["questions"] != len(expected):
        raise ValueError(f"{path}: сводка не соответствует полному test")
    if summary["fingerprint"] != manifest["fingerprint"]:
        raise ValueError(f"{path}: расходятся fingerprint сводки и манифеста")
    for row in rows:
        original = expected[row["question_id"]]
        if any(row[key] != original[key] for key in ("question", "gold_context", "gold_answer")):
            raise ValueError(f"{path}: изменён вопрос или эталон")
        if any(not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) for key in METRICS):
            raise ValueError(f"{path}: отсутствует числовая метрика")
        if type(row["answer_correctness"]) is not int or row["answer_correctness"] not in (0, 1) or row["time"] <= 0:
            raise ValueError(f"{path}: неверная оценка или время")
        if any(not 0 <= row[key] <= 1 for key in ("context_recall", "context_precision")):
            raise ValueError(f"{path}: метрика контекста вне диапазона [0, 1]")
    averages = {key: mean(row[key] for row in rows) for key in METRICS}
    if any(not math.isclose(averages[key], summary[key], rel_tol=0, abs_tol=1e-10) for key in METRICS):
        raise ValueError(f"{path}: средние не совпадают со строками JSONL")
    return summary["fingerprint"], averages, len(rows)


def compare(reference, candidate, *, allow_fingerprint_change=False):
    if reference.resolve() == candidate.resolve():
        raise ValueError("Для проверки нужны два отдельных прогона")
    ref_hash, ref_mean, count = load_run(reference)
    cand_hash, cand_mean, _ = load_run(candidate)
    same_fingerprint = ref_hash == cand_hash
    if not same_fingerprint and not allow_fingerprint_change:
        raise ValueError("Код, индекс или параметры прогонов различаются")
    metrics = {}
    for name in METRICS:
        baseline, repeated = ref_mean[name], cand_mean[name]
        lower, upper = baseline * 0.95, baseline * 1.05
        metrics[name] = {
            "reference": baseline, "repeat": repeated,
            "relative_change_pct": 100 * (repeated - baseline) / baseline if baseline else None,
            "allowed_min": lower, "allowed_max": upper,
            "passed": lower <= repeated <= upper,
        }
    return {"reference": str(reference), "repeat": str(candidate), "questions": count,
            "reference_fingerprint": ref_hash, "repeat_fingerprint": cand_hash,
            "same_fingerprint": same_fingerprint, "tolerance_pct": 5,
            "comparison": "same_version" if same_fingerprint else "changed_version",
            "passed": all(value["passed"] for value in metrics.values()), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-fingerprint-change", action="store_true",
                        help="Сравнить метрики после изменения кода или настроек; отметить различие в отчёте")
    args = parser.parse_args()
    report = compare(args.reference, args.repeat, allow_fingerprint_change=args.allow_fingerprint_change)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
