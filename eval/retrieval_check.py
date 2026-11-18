"""Проверка поиска на train без вызовов LLM и без настройки на test."""
import json
from pathlib import Path
from statistics import mean

from rag.config import DATA
from rag.retrieval import Retriever


if __name__ == "__main__":
    system = Retriever(dense=False)
    results = []
    for row in map(json.loads, (DATA / "train.jsonl").read_text().splitlines()):
        candidates = system.search(row["question"], k=18, method="bm25")
        gold = {(doc["doc_id"], page) for doc in row["gold_context"] for page in doc["pages"]}
        scores = {}
        for k in (3, 10, 18):
            hits = gold & {(p["doc_id"], p["page"]) for p in candidates[:k]}
            scores[f"recall@{k}"] = len(hits) / len(gold)
        results.append({"question_id": row["question_id"], **scores})
    output = Path("results/train_retrieval.json")
    output.parent.mkdir(exist_ok=True)
    summary = {key: mean(row[key] for row in results) for key in ("recall@3", "recall@10", "recall@18")}
    output.write_text(json.dumps({"questions": len(results), "method": "BM25 с фильтрацией организации и приоритетом периода", "summary": summary, "rows": results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
