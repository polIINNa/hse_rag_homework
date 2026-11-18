"""Полный test: python -m evaluate --split test --output test_results.xlsx"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from eval.eval_functions import score_answer, score_context
from rag.common import ROOT, load_env, read_jsonl, write_json
from rag.budget import call_official_judge
from rag.pipeline import RAG

METRICS = ("context_recall", "context_precision", "answer_correctness", "time")
COLUMNS = ("question_id", "question", "gold_context", "gold_answer", "context", "answer",
           "context_recall", "context_precision", "answer_correctness", "time")


def save_excel(results, path):
    if not results:
        raise ValueError("Нет оценённых вопросов")
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    summary.append(METRICS)
    summary.append([mean(row[key] for row in results) for key in METRICS])
    details = report.create_sheet("По вопросам")
    details.append(COLUMNS)
    for row in results:
        values = [json.dumps(row[key], ensure_ascii=False) if key in ("gold_context", "context")
                  else row[key] for key in COLUMNS]
        details.append(values)
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        details.row_dimensions[details.max_row].height = 75
    reasons = report.create_sheet("Судья")
    reasons.append(["question_id", "reason"])
    for row in results:
        reasons.append([row["question_id"], row["judge_reason"]])
        for cell in reasons[reasons.max_row]:
            cell.data_type = "s"
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for sheet in report:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24466B")
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = 24
    for col in ("B", "D", "F"):
        details.column_dimensions[col].width = 65
    for col in ("C", "E"):
        details.column_dimensions[col].width = 32
    reasons.column_dimensions["B"].width = 110
    for row in details.iter_rows(min_row=2, min_col=7, max_col=10):
        for cell in row:
            cell.number_format = "0.000"
    for cell in summary[2]:
        cell.number_format = "0.000"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.xlsx")
    report.save(temporary)
    temporary.replace(path)


def evaluate(split="test", output=None, limit=None, system=None, judge=score_answer, resume=False,
             update_readme=True, train_cache=None):
    load_env()
    real_run = system is None and judge is score_answer
    if split == "test" and not real_run:
        raise ValueError("Итоговый test разрешён только с реальным RAG и исходным судьёй")
    if train_cache is not None and (split != "train" or not real_run):
        raise ValueError("Кэш сравнительного эксперимента доступен только для реального train")
    dataset = ROOT / f"data/{split}.jsonl"
    items = read_jsonl(dataset)
    if split == "test" and (limit is not None or len(items) != 141):
        raise ValueError("Итоговая оценка требует все 141 вопрос; limit разрешён только на train")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit должен быть положительным")
        items = items[:limit]
    if len({item["question_id"] for item in items}) != len(items):
        raise ValueError("В датасете повторяются question_id")
    cached = {}
    if train_cache is not None:
        prior = read_jsonl(train_cache)
        cached = {row["question_id"]: row for row in prior}
        expected = {item["question_id"]: item for item in items}
        if len(cached) != len(prior) or not cached.keys() <= expected.keys():
            raise ValueError("Кэш содержит лишние или повторные train-вопросы")
        for key, row in cached.items():
            if any(row.get(field) != expected[key][field] for field in ("question", "gold_answer", "gold_context")):
                raise ValueError(f"Вопрос или разметка в кэше изменились: {key}")
    output = Path(output or ROOT / f"{split}_results.xlsx")
    checkpoint = output.with_suffix(".jsonl")
    pending_path = output.with_suffix(".pending.json")
    if output.exists() or ((checkpoint.exists() or pending_path.exists()) and not resume):
        raise FileExistsError(f"Результаты уже существуют: {output}. Для нового прогона задайте новый --output")
    system = system if system is not None else RAG()
    code_files = sorted(list((ROOT / "rag").glob("*.py")) + [ROOT / "evaluate.py", ROOT / "eval/eval_functions.py"])
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in code_files if p.exists())).hexdigest()
    identity = {"dataset": hashlib.sha256(dataset.read_bytes()).hexdigest(), "code": code_hash,
                "index": getattr(system.index, "fingerprint", "unknown"),
                "index_file": getattr(system.index, "file_sha256", "unknown"),
                "model": getattr(system.llm, "model", "unknown"),
                "top_k": system.top_k, "agentic": system.agentic,
                "context_chars": getattr(system, "context_chars", None), "limit": limit, "split": split}
    if train_cache is not None:
        identity["train_cache_sha256"] = hashlib.sha256(train_cache.read_bytes()).hexdigest()
    run_path = output.with_suffix(".run.json")
    results = read_jsonl(checkpoint) if resume and checkpoint.exists() else []
    if results or (resume and pending_path.exists()):
        if not run_path.exists() or json.loads(run_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Нельзя продолжить: код, индекс, параметры или датасет изменились")
        if [r["question_id"] for r in results] != [i["question_id"] for i in items[:len(results)]]:
            raise ValueError("Контрольные строки не совпадают с порядком датасета")
    # Восстановление после прерывания между append завершённой строки и удалением pending.
    if resume and results and pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        last = results[-1]
        if pending["question_id"] == last["question_id"]:
            same = pending["time"] == last["time"] and all(
                pending["prediction"].get(k, []) == last.get(k, []) for k in ("answer", "context", "steps"))
            if not same:
                raise ValueError("Завершённая строка расходится с pending")
            pending_path.unlink()
    write_json(run_path, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(items[len(results):], len(results) + 1):
        if item["question_id"] in cached:
            if pending_path.exists():
                raise ValueError("Для готового ответа в кэше уже есть незавершённый запрос")
            row = cached[item["question_id"]]
            with checkpoint.open("a", encoding="utf-8") as target:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
            print(f"{i}/{len(items)} {item['question_id']} взят из предыдущего прогона", flush=True)
            continue
        if resume and pending_path.exists():
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            if pending["question_id"] != item["question_id"]:
                raise ValueError("Несовпадение незавершённого вопроса при resume")
            prediction, elapsed = pending["prediction"], pending["time"]
        else:
            start = perf_counter()
            prediction = system.rag(item["question"])
            elapsed = perf_counter() - start
            write_json(pending_path, {"question_id": item["question_id"], "prediction": prediction, "time": elapsed})
        if not isinstance(prediction.get("answer"), str) or not prediction["answer"].strip():
            raise ValueError("Пустой ответ RAG")
        context_scores = score_context(item["gold_context"], prediction["context"])
        answer_score = call_official_judge(judge, item["gold_answer"], prediction["answer"]) if judge is score_answer else judge(item["gold_answer"], prediction["answer"])
        if type(answer_score["answer_correctness"]) is not int or answer_score["answer_correctness"] not in (0, 1):
            raise ValueError("Судья вернул некорректную оценку")
        row = {**item, "context": prediction["context"], "answer": prediction["answer"],
               **context_scores, "answer_correctness": answer_score["answer_correctness"],
               "time": elapsed, "judge_reason": answer_score["reason"], "steps": prediction.get("steps", [])}
        with checkpoint.open("a", encoding="utf-8") as target:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
        pending_path.unlink(missing_ok=True)
        results.append(row)
        print(f"{i}/{len(items)} {item['question_id']} recall={row['context_recall']:.3f} "
              f"precision={row['context_precision']:.3f} answer={row['answer_correctness']} "
              f"time={elapsed:.2f}s", flush=True)
    save_excel(results, output)
    summary = {key: mean(row[key] for row in results) for key in METRICS}
    manifest = {"status": "complete", "split": split, "questions": len(results),
                "utc_finished": datetime.now(timezone.utc).isoformat(), "means": summary,
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "judge_sha256": hashlib.sha256((ROOT / "eval/eval_functions.py").read_bytes()).hexdigest(),
                "corpus_sha256": getattr(system.index, "fingerprint", "unknown"),
                "rag_model": getattr(system.llm, "model", "unknown"),
                "top_k": system.top_k, "agentic": system.agentic, "python": platform.python_version(),
                "code_sha256": code_hash, "context_chars": getattr(system, "context_chars", None),
                "index_file_sha256": getattr(system.index, "file_sha256", "unknown"),
                "source": "live_gateway" if real_run else "injected_fixture",
                "packages": {name: importlib.metadata.version(name) for name in
                             ["PyMuPDF", "numpy", "scipy", "scikit-learn", "openpyxl", "requests"]},
                "time_definition": "search + agent decisions + generation; excludes index load and judge"}
    if train_cache is not None:
        manifest["reused_train_rows"] = len(cached)
        manifest["train_cache_sha256"] = identity["train_cache_sha256"]
    write_json(output.with_suffix(".summary.json"), manifest)
    # README получает числа только после полного реального test, без инъекции fake system/judge.
    if split == "test" and real_run and update_readme:
        path = ROOT / "README.md"
        text = path.read_text(encoding="utf-8")
        begin, end = "<!-- RESULTS -->", "<!-- /RESULTS -->"
        table = "\n\n| Метрика | Среднее по 141 вопросу |\n| --- | ---: |\n"
        table += "\n".join(f"| {key} | {summary[key]:.6f} |" for key in METRICS) + "\n\n"
        if begin not in text or end not in text:
            raise ValueError("В README нет маркеров для итоговых средних")
        text = text.split(begin)[0] + begin + table + end + text.split(end, 1)[1]
        path.write_text(text, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Продолжить ту же серию без смены кода и настроек")
    parser.add_argument("--no-readme", action="store_true", help="Повторный прогон: не менять средние основной серии в README")
    parser.add_argument("--train-cache", type=Path, help="Готовые неизменившиеся ответы сравнительного train")
    args = parser.parse_args()
    print(evaluate(args.split, args.output, args.limit, resume=args.resume, update_readme=not args.no_readme,
                   train_cache=args.train_cache))
