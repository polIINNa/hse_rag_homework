"""Excel в формате eval/evaluator_excel.py из эксперимента, прогнанного eval/evaluator_langsmith.py.

Из корня репозитория: python -m eval.langsmith_to_excel rag-564492c5 data/test.jsonl --output test_results.xlsx

Ответ, context и time берутся из прогонов эксперимента, метрики — из оценок (feedback), которые
поставили функции из eval/eval_functions.py. Порядок строк — как в датасете.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langsmith import Client

from eval.evaluator_excel import read_items
from eval.evaluator_resumable import METRICS, write_report

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def export(experiment, dataset_path, output_path):
    client = Client()
    runs = {r.inputs["question"]: r for r in client.list_runs(project_name=experiment, is_root=True)}
    scores = {}
    for feedback in sorted(client.list_feedback(run_ids=[r.id for r in runs.values()]), key=lambda f: f.created_at):
        if feedback.score is not None:
            scores[(feedback.run_id, feedback.key)] = feedback.score
    results = []
    for item in read_items(Path(dataset_path)):
        run = runs[item["question"]]
        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
            "gold_answer": item["gold_answer"],
            "context": json.dumps(run.outputs["context"], ensure_ascii=False),
            "answer": run.outputs["answer"],
            **{key: scores[(run.id, key)] for key in METRICS},
        })
    write_report(results, output_path)
    print({key: round(mean(row[key] for row in results), 4) for key in METRICS})
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, default=Path("test_results.xlsx"))
    args = parser.parse_args()
    print(export(args.experiment, args.dataset_path, args.output))
