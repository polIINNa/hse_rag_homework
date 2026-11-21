"""Прогон на train для отладки (финальная оценка — eval/evaluator_excel.py).

Из корня репозитория: python -m rag.dev_eval data/train.jsonl --limit 40 --tag baseline
Строки пишутся в outputs/<tag>.jsonl по мере готовности.
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from time import perf_counter

import requests

from eval.eval_functions import score_answer, score_context
from rag.agent import ROOT, Agent
from rag.index import Index


def judge(agent, gold_answer, answer):
    """Судья через общий лимитер шлюза, с повтором при 429."""
    for attempt in range(5):
        agent.llm.window.acquire()
        try:
            return score_answer(gold_answer, answer)
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 429 or attempt == 4:
                raise
            agent.llm.window.penalize()


def run_one(agent, item):
    start = perf_counter()
    result = agent.run(item["question"], reserve_judge=False)
    elapsed = perf_counter() - start
    verdict = judge(agent, item["gold_answer"], result["answer"])
    return {
        "question_id": item["question_id"], "question": item["question"],
        "gold_answer": item["gold_answer"], "answer": result["answer"],
        "gold_context": item["gold_context"], "context": result["context"],
        "retrieved": result["retrieved"], "searches": result["searches"], "reason": verdict["reason"],
        **score_context(item["gold_context"], result["context"]),
        "answer_correctness": verdict["answer_correctness"], "time": elapsed, "llm_wait": result["llm_wait"],
    }


def summarize(rows):
    metrics = {k: round(mean(r[k] for r in rows), 3)
               for k in ("context_recall", "context_precision", "answer_correctness", "time")}
    metrics["searches"] = round(mean(len(r["searches"]) for r in rows), 2)
    if all("llm_wait" in r for r in rows):
        metrics["time_without_rate_wait"] = round(mean(r["time"] - r["llm_wait"] for r in rows), 3)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag", default="dev")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.dataset_path, encoding="utf-8") if line.strip()]
    if args.limit:
        items = random.Random(args.seed).sample(items, min(args.limit, len(items)))
    agent = Agent(index=Index())
    agent.llm.cooldown()

    out = ROOT / "outputs" / f"{args.tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("")
    lock, rows = threading.Lock(), []

    def work(item):
        row = run_one(agent, item)
        with lock:
            rows.append(row)
            with out.open("a", encoding="utf-8") as target:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{len(rows)}/{len(items)} R={row['context_recall']:.2f} P={row['context_precision']:.2f} "
                  f"AC={row['answer_correctness']} t={row['time']:.1f}s", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, items))
    print(agent.llm.model, len(rows), summarize(rows))
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
