"""Загрузить реальные сохранённые трассы и оценки в эксперимент LangSmith."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

from rag.config import required
from eval.run import METRICS


def dotted(timestamp, run_id):
    return datetime.fromisoformat(timestamp).strftime("%Y%m%dT%H%M%S%fZ") + str(run_id)


def upload(path, name, share=False, workers=3):
    if not 1 <= workers <= 4:
        raise ValueError("workers должен быть от 1 до 4")
    required("LANGSMITH_API_KEY")
    client = Client(timeout_ms=60_000, auto_batch_tracing=False)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("answer_correctness") is None for row in rows):
        raise ValueError("Нужен завершённый прогон с оценками судьи")
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    if len(rows) != summary["questions"]:
        raise ValueError("Число строк не совпадает со сводкой")
    dataset_name = "hse-financial-reports-" + summary["split"] + "-stepan-ivanov"
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(dataset_name, description="Вопросы из открытого репозитория polIINNa/hse_rag_homework. Эталоны используются только оценщиком.")
    existing = {example.metadata.get("question_id"): example for example in client.list_examples(dataset_id=dataset.id)}
    new = []
    for row in rows:
        if row["question_id"] not in existing:
            new.append({"id": str(uuid.uuid5(dataset.id, row["question_id"])),
                        "inputs": {"question": row["question"]},
                        "reference_outputs": {"gold_context": row["gold_context"], "gold_answer": row["gold_answer"]},
                        "metadata": {"question_id": row["question_id"]}})
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    project = client.create_project(name, reference_dataset_id=dataset.id, upsert=True,
                                    num_examples=len(rows), num_repetitions=1, evaluator_keys=METRICS,
                                    description="Агентный RAG: гибридный поиск, выбор страниц, дополнительный поиск, калькулятор.",
                                    metadata={"fingerprint": summary["fingerprint"], "source": "recorded_local_run", "split": summary["split"]})
    checkpoint = path.with_suffix(".langsmith.json")
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    if saved and saved.get("project_id") != str(project.id):
        raise ValueError("Checkpoint относится к другому эксперименту")
    done = set(saved.get("uploaded", []))
    def upload_one(row):
        root_id = uuid.UUID(row["run_id"])
        example_id = existing[row["question_id"]].id if row["question_id"] in existing else uuid.uuid5(dataset.id, row["question_id"])
        root_order = dotted(row["start_time"], root_id)
        client.create_run("financial_rag", {"question": row["question"]}, "chain", id=root_id,
                          project_name=name, reference_example_id=example_id,
                          start_time=row["start_time"], end_time=row["end_time"],
                          outputs={"answer": row["answer"], "context": row["context"], "time": row["time"]},
                          trace_id=root_id, dotted_order=root_order,
                          extra={"metadata": {"question_id": row["question_id"], "fingerprint": summary["fingerprint"]}})
        stages = json.loads((path.parent / (path.stem + "_traces") / (row["question_id"] + ".json")).read_text())
        for stage in stages:
            run_type = "retriever" if stage["name"].endswith("search") else "tool" if stage["name"] == "calculator" else "chain"
            client.create_run(stage["name"], stage["inputs"], run_type, id=stage["id"], project_name=name,
                              start_time=stage["start_time"], end_time=stage["end_time"],
                              outputs={"result": stage["outputs"]}, parent_run_id=root_id, trace_id=root_id,
                              dotted_order=root_order + "." + dotted(stage["start_time"], stage["id"]))
        for metric in METRICS:
            client.create_feedback(root_id, metric, score=row[metric],
                                   session_id=project.id,
                                   feedback_id=uuid.uuid5(root_id, metric),
                                   comment=row["judge_reason"] if metric == "answer_correctness" else None)
        return row["question_id"]

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(upload_one, row): row["question_id"] for row in rows if row["question_id"] not in done}
        for future in as_completed(pending):
            try:
                done.add(future.result())
            except Exception as exc:
                failures.append(pending[future])
                print(f"LangSmith: ошибка {pending[future]} ({type(exc).__name__}), можно повторить загрузку", flush=True)
                continue
            saved = {"dataset_id": str(dataset.id), "project_id": str(project.id), "project_name": name, "uploaded": sorted(done)}
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
            print(f"LangSmith: {len(done)}/{len(rows)}", flush=True)
    if failures:
        raise RuntimeError(f"Не загружено вопросов: {len(failures)}. Повторите ту же команду для продолжения.")
    client.update_project(project.id, end_time=datetime.now(timezone.utc))
    if share:
        shared = client.share_dataset(dataset_id=dataset.id)
        saved["public_url"] = shared["url"]
        checkpoint.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
        print("Открытая ссылка:", saved["public_url"], flush=True)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--name", default="stepan-ivanov-agent-rag-test")
    parser.add_argument("--share", action="store_true", help="Открыть преподавателю по публичной ссылке")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    upload(args.results, args.name, args.share, args.workers)
