"""Сравнить кандидатов BM25 и гибридного поиска на каждом десятом вопросе train."""
import json
from pathlib import Path
from statistics import mean

from rag.config import DATA
from rag.retrieval import Retriever

if __name__ == "__main__":
    retriever = Retriever()
    questions = [json.loads(s) for s in (DATA / "train.jsonl").read_text().splitlines()][::10]
    rows = []
    for number, item in enumerate(questions, 1):
        gold = {(g["doc_id"], p) for g in item["gold_context"] for p in g["pages"]}
        result = {"question_id": item["question_id"]}
        for method in ("bm25", "hybrid"):
            found = retriever.search(item["question"], method=method)
            for k in (3, 18):
                hits = gold & {(p["doc_id"], p["page"]) for p in found[:k]}
                result[f"{method}@{k}"] = len(hits) / len(gold)
        rows.append(result)
        print(f"{number}/{len(questions)}", flush=True)
    summary = {key: mean(r[key] for r in rows) for key in ("bm25@3", "bm25@18", "hybrid@3", "hybrid@18")}
    Path("results").mkdir(exist_ok=True)
    Path("results/train_hybrid_check.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(summary)
