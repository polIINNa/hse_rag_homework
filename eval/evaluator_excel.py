"""Из корня репозитория: python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx"""

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
from rag.rate_limit import wait_for_slot

METRICS = ("context_recall", "context_precision", "answer_correctness", "time")


def write_excel(results, output_path):
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
                cell.data_type = "s"  # Сохранить как текст, даже если ответ начинается с "=".
    report.save(output_path)


def judge(gold_answer, answer):
    """Повторы не меняют модель, промпт или параметры преподавательского судьи."""
    import requests
    for attempt in range(4):
        try:
            wait_for_slot()
            return score_answer(gold_answer, answer)
        except requests.HTTPError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(6 * (attempt + 1))
        except (requests.Timeout, requests.ConnectionError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


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


def evaluate(dataset_path, output_path=None, langsmith=False, share=False, resume_experiment=None):
    dataset_path = Path(dataset_path)
    items = list(read_items(dataset_path))
    if not items:
        raise ValueError("В датасете нет вопросов.")
    if len({i["question_id"] for i in items}) != len(items):
        raise ValueError("Повторяющиеся question_id")
    if langsmith and not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("Для --langsmith заполните LANGSMITH_API_KEY в .env")
    if share and not langsmith:
        raise ValueError("--share используется вместе с --langsmith")
    if resume_experiment and not langsmith:
        raise ValueError("--resume-experiment используется вместе с --langsmith")
    if not os.getenv("LLM_API_KEY"):
        raise RuntimeError("Заполните LLM_API_KEY в .env ключом преподавателя")
    os.environ["LANGSMITH_TRACING"] = "true" if langsmith else "false"
    if output_path is None:
        output_path = dataset_path.with_name(dataset_path.stem + "_results.xlsx")
    output_path = Path(output_path)
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("Выберите для отчёта отдельный файл.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    system = RAG()
    results = []
    checkpoint = output_path.with_suffix(".jsonl")
    if resume_experiment:
        by_id = {item["question_id"]: item for item in items}
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            saved.pop("judge_reason", None)
            item = by_id[saved["question_id"]]
            if (saved["question"] != item["question"] or saved["gold_answer"] != item["gold_answer"]
                    or json.loads(saved["gold_context"]) != item["gold_context"]):
                raise ValueError("Checkpoint относится к другому датасету")
            results.append(saved)
        if len({r["question_id"] for r in results}) != len(results):
            raise ValueError("В checkpoint повторяются question_id")
    else:
        checkpoint.write_text("", encoding="utf-8")

    def predict(inputs):
        start = perf_counter()
        prediction = system.rag(inputs["question"])
        elapsed = perf_counter() - start
        return {**prediction, "time": elapsed}

    def assess(item, prediction):
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
            "time": prediction["time"],
        }
        results.append(row)
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**row, "judge_reason": answer_score["reason"]}, ensure_ascii=False) + "\n")
        print(f"{len(results)}/{len(items)} {item['question_id']}: correctness={answer_score['answer_correctness']}, time={prediction['time']:.2f}s", flush=True)
        return [{"key": key, "score": row[key], **({"comment": answer_score["reason"]} if key == "answer_correctness" else {})} for key in METRICS]

    experiment_url = None
    if langsmith:
        from langsmith import Client
        client = Client()
        project = None
        if resume_experiment:
            project = client.read_project(project_name=resume_experiment)
            dataset = client.read_dataset(dataset_id=project.reference_dataset_id)
        else:
            dataset = client.create_dataset(dataset_name=f"financial-rag-{dataset_path.stem}-{time.time_ns()}")
            client.create_examples(dataset_id=dataset.id, examples=[{
                "inputs": {"question": i["question"], "question_id": i["question_id"]},
                "outputs": {"gold_context": i["gold_context"], "gold_answer": i["gold_answer"]},
            } for i in items])
        examples_by_id = {example.inputs["question_id"]: example
                          for example in client.list_examples(dataset_id=dataset.id)}
        for item in items:
            example = examples_by_id[item["question_id"]]
            if (example.inputs["question"] != item["question"]
                    or example.outputs != {k: item[k] for k in ("gold_context", "gold_answer")}):
                raise ValueError("Эксперимент относится к другому датасету")

        if resume_experiment:
            # Если упал только судья, используем уже измеренный ответ того же запуска.
            # Обычный запуск eval никогда не читает сохранённые ответы.
            completed = {r["question_id"] for r in results}
            saved_by_id = {r["question_id"]: r for r in results}
            pending_by_example = {str(examples_by_id[i["question_id"]].id): i
                                  for i in items if i["question_id"] not in completed}
            for run in client.list_runs(project_id=project.id, is_root=True):
                saved = saved_by_id.get(run.inputs.get("question_id"))
                if saved and not run.error and run.outputs:
                    prediction = run.outputs
                    if (prediction.get("answer") == saved["answer"]
                            and prediction.get("context") == json.loads(saved["context"])
                            and prediction.get("time") == saved["time"]):
                        for key in METRICS:
                            if key not in (run.feedback_stats or {}):
                                client.create_feedback(run.id, key=key, score=saved[key])
                item = pending_by_example.get(str(run.reference_example_id))
                if item and item["question_id"] not in completed and not run.error and run.outputs:
                    for feedback in assess(item, run.outputs):
                        client.create_feedback(run.id, **feedback)
                    completed.add(item["question_id"])
        completed = {r["question_id"] for r in results}
        ordered_examples = [examples_by_id[item["question_id"]] for item in items
                            if item["question_id"] not in completed]

        def evaluator(run, example):
            if run.error or not run.outputs:
                raise RuntimeError(f"RAG не ответил на {example.inputs['question_id']}")
            return {"results": assess({**example.inputs, **example.outputs}, run.outputs)}

        if ordered_examples:
            experiment = client.evaluate(predict, data=ordered_examples, evaluators=[evaluator],
                                         experiment=project,
                                         experiment_prefix=None if project else "financial-agentic-rag",
                                         max_concurrency=0,
                                         metadata={"rag_model": system.llm.model}, blocking=True)
            # Iteration waits for all runs and evaluators before validating completeness.
            list(experiment)
            project = client.read_project(project_name=experiment.experiment_name)
        experiment_url = project.url
    else:
        for item in items:
            assess(item, predict({"question": item["question"]}))
    if len(results) != len(items):
        raise RuntimeError(f"Оценено {len(results)}/{len(items)} вопросов: полный Excel не создан; см. {checkpoint}")
    by_id = {r["question_id"]: r for r in results}
    results = [by_id[i["question_id"]] for i in items]

    write_excel(results, output_path)
    if share:
        experiment_url = client.share_dataset(dataset_id=dataset.id)["url"]
    stats = {key: mean(r[key] for r in results) for key in METRICS}
    output_path.with_suffix(".summary.json").write_text(json.dumps({"questions": len(results), **stats, "experiment_url": experiment_url}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if experiment_url:
        print(f"LangSmith: {experiment_url}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    parser.add_argument("--langsmith", action="store_true", help="Тот же прогон как эксперимент LangSmith с трассами и четырьмя метриками")
    parser.add_argument("--share", action="store_true", help="Открыть доступ по ссылке к датасету, эксперименту, трассам и метрикам")
    parser.add_argument("--resume-experiment", help="Продолжить прерванный эксперимент LangSmith из JSONL рядом с --output")
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output, args.langsmith, args.share, args.resume_experiment))
