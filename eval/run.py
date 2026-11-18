"""Воспроизводимый прогон с сохранением каждого завершённого вопроса."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import time
import uuid

import requests

from eval.eval_functions import score_answer, score_context
from rag.clients import retry_delay
from rag.config import ROOT
from rag.pipeline import RAG

METRICS = ("context_recall", "context_precision", "answer_correctness", "time")


def judge(gold, answer, limiter):
    for attempt in range(6):
        limiter.wait()
        try:
            return score_answer(gold, answer)
        except requests.HTTPError as exc:
            if exc.response.status_code not in {408, 429, 500, 502, 503, 504, 529} or attempt == 5:
                raise
            delay = retry_delay(exc.response.headers.get("Retry-After"), attempt)
            if delay > 300:
                raise RuntimeError(f"Судья требует ожидания {delay:.0f}s. Продолжите позже с --resume.") from exc
            time.sleep(delay)
        except (requests.Timeout, requests.ConnectionError, ValueError):
            if attempt == 5:
                raise
            time.sleep(retry_delay(None, attempt))


def fingerprint(args):
    digest = hashlib.sha256()
    for path in sorted((ROOT / "rag").glob("*.py")) + [Path(__file__), ROOT / "eval" / "eval_functions.py", ROOT / "data" / f"{args.split}.jsonl"]:
        digest.update(path.read_bytes())
    for name in ("metadata.json", "pages.jsonl", "vectors.npy"):
        path = ROOT / "index" / name
        if path.exists():
            digest.update(path.read_bytes())
    digest.update(json.dumps({"split": args.split, "limit": args.limit, "stride": args.stride,
                              "mode": args.mode, "dense": not args.no_dense, "judge": not args.no_judge}, sort_keys=True).encode())
    digest.update(json.dumps({k: os.environ.get(k, "") for k in ("LLM_MODEL", "LLM_BASE_URL", "LLM_RPM", "EMB_BASE_URL", "EMB_RPM")}, sort_keys=True).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("results/test.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--mode", choices=("agent", "baseline"), default="agent")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and (args.no_dense or args.no_judge or args.stride != 1):
        parser.error("Итоговый test требует dense-поиск, судью и все вопросы без прореживания")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".manifest.json")
    run_hash = fingerprint(args)
    if output.exists() and not args.resume:
        parser.error("Файл уже существует. Выберите другой --output или --resume")
    completed = []
    if args.resume and output.exists():
        if json.loads(manifest.read_text())["fingerprint"] != run_hash:
            parser.error("Код, данные или параметры изменились. Нужен отдельный прогон")
        completed = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    else:
        manifest.write_text(json.dumps({"fingerprint": run_hash, "split": args.split, "mode": args.mode,
                                       "started_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    items = [json.loads(x) for x in (ROOT / "data" / f"{args.split}.jsonl").read_text().splitlines()][::args.stride]
    if args.limit:
        items = items[:args.limit]
    done_ids = {row["question_id"] for row in completed}
    if len(done_ids) != len(completed) or not done_ids.issubset({q["question_id"] for q in items}):
        raise ValueError("Несогласованный checkpoint")
    system = RAG(mode=args.mode, dense=not args.no_dense)
    if args.split == "test":
        # Одинаковый протокол первого и повторного прогона: один вопрос train
        # прогревает соединения и проверяет готовность RAG до начала измерений.
        warmup = json.loads((ROOT / "data" / "train.jsonl").read_text().splitlines()[0])["question"]
        print("Прогрев на одном вопросе train; в метрики test не входит", flush=True)
        system.rag(warmup)
    trace_dir = output.parent / (output.stem + "_traces")
    trace_dir.mkdir(exist_ok=True)
    for number, item in enumerate(items, 1):
        if item["question_id"] in done_ids:
            continue
        started_at = datetime.now(timezone.utc).isoformat()
        before = time.perf_counter()
        # RAG получает только текст вопроса, без id, gold_context и gold_answer.
        try:
            prediction = system.rag(item["question"])
        except Exception:
            (trace_dir / (item["question_id"] + ".failed.json")).write_text(json.dumps(system.last_trace, ensure_ascii=False))
            raise
        elapsed = time.perf_counter() - before
        ended_at = datetime.now(timezone.utc).isoformat()
        context_scores = score_context(item["gold_context"], prediction["context"])
        answer_score = {"answer_correctness": None, "reason": "Судья отключён для разработки"} if args.no_judge else judge(item["gold_answer"], prediction["answer"], system.llm.limiter)
        row = {"question_id": item["question_id"], "question": item["question"],
               "gold_context": item["gold_context"], "gold_answer": item["gold_answer"],
               **prediction, **context_scores, "answer_correctness": answer_score["answer_correctness"],
               "time": elapsed, "judge_reason": answer_score["reason"], "run_id": str(uuid.uuid4()),
               "start_time": started_at, "end_time": ended_at}
        (trace_dir / (item["question_id"] + ".json")).write_text(json.dumps(system.last_trace, ensure_ascii=False))
        with output.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        completed.append(row)
        print(f"{number}/{len(items)} {item['question_id']} recall={row['context_recall']:.3f} precision={row['context_precision']:.3f} correct={row['answer_correctness']} time={elapsed:.1f}s", flush=True)
    if len(completed) != len(items):
        raise ValueError("Прогон неполный")
    summary = {key: mean(r[key] for r in completed) if all(r[key] is not None for r in completed) else None for key in METRICS}
    summary.update({"questions": len(completed), "split": args.split, "fingerprint": run_hash})
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

