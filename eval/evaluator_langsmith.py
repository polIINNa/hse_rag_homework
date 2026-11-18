"""Из корня репозитория: python -m eval.evaluator_langsmith ИМЯ_ДАТАСЕТА"""

import argparse
from pathlib import Path
from time import perf_counter

from eval.eval_functions import score_answer, score_context
from eval.rag_example import RAG
from evaluate import METRICS, save_excel
from rag.budget import call_official_judge
from rag.common import ROOT, read_jsonl


def metrics(outputs, reference_outputs):
    context = score_context(reference_outputs["gold_context"], outputs["context"])
    answer = call_official_judge(score_answer, reference_outputs["gold_answer"], outputs["answer"])
    return {"results": [
        {"key": "context_recall", "score": context["context_recall"]},
        {"key": "context_precision", "score": context["context_precision"]},
        {"key": "answer_correctness", "score": answer["answer_correctness"], "comment": answer["reason"]},
        {"key": "time", "score": outputs["time"]},
    ]}


def collect_rows(results, items):
    expected = {item["question_id"]: item for item in items}
    rows = {}
    for result in results:
        example = result["example"]
        question_id = (example.metadata or {}).get("question_id")
        if question_id not in expected or question_id in rows:
            raise ValueError(f"Неизвестный или повторный вопрос в LangSmith: {question_id}")
        item = expected[question_id]
        if example.inputs["question"] != item["question"] or any(
            example.outputs[key] != item[key] for key in ("gold_answer", "gold_context")
        ):
            raise ValueError(f"Пример в LangSmith не совпал с test: {question_id}")
        prediction = result["run"].outputs
        if not prediction or not prediction.get("answer", "").strip():
            raise ValueError(f"Отсутствует ответ по вопросу {question_id}")
        evaluations = {metric.key: metric for metric in result["evaluation_results"]["results"]}
        if any(name not in evaluations or evaluations[name].score is None for name in METRICS):
            raise ValueError(f"Отсутствуют оценки по вопросу {question_id}")
        rows[question_id] = {**item, **prediction,
                            **{name: evaluations[name].score for name in METRICS},
                            "judge_reason": evaluations["answer_correctness"].comment or ""}
    if rows.keys() != expected.keys():
        raise ValueError("LangSmith не вернул результаты для всех вопросов test")
    return [rows[item["question_id"]] for item in items]


def evaluate(dataset_name, output=None):
    from langsmith import Client

    if output and output.exists():
        raise FileExistsError(f"Файл уже существует: {output}")
    client = Client()
    items = read_jsonl(ROOT / "data/test.jsonl") if output else []
    system = RAG()

    def answer_question(inputs):
        start = perf_counter()
        result = system.rag(inputs["question"])
        return {**result, "time": perf_counter() - start}

    results = client.evaluate(
        answer_question, data=dataset_name, evaluators=[metrics], experiment_prefix="rag", max_concurrency=0,
    )
    if output:
        save_excel(collect_rows(results, items), output)
        print(f"Excel: {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name")
    parser.add_argument("--output", type=Path, help="Сохранить также Excel для сравнения с результатами сдачи")
    parser.add_argument("--url-output", type=Path, help="Сохранить ссылку на завершённый эксперимент")
    args = parser.parse_args()
    results = evaluate(args.dataset_name, args.output)
    url = getattr(results, "url", None)
    if args.url_output:
        if not url:
            raise RuntimeError("LangSmith не вернул ссылку на завершённый эксперимент")
        args.url_output.write_text(url + "\n", encoding="utf-8")
    print(url or results.experiment_name)

