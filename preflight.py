"""Подготовительная проверка. Не делает запросов к LLM или судье."""
import argparse
import hashlib
import importlib.metadata
import json
import os

from rag.common import ROOT, load_env, read_jsonl
from rag.index import PageIndex


def check(require_index=False):
    load_env()
    info = {"inference_requests": 0, "key_present": bool(os.getenv("LLM_API_KEY")),
            "documents": len(read_jsonl(ROOT / "data/documents.jsonl")),
            "test_questions": len(read_jsonl(ROOT / "data/test.jsonl")),
            "train_questions": len(read_jsonl(ROOT / "data/train.jsonl")),
            "pdf_present": sum((ROOT / "data/reports" / d["path"]).is_file()
                               for d in read_jsonl(ROOT / "data/documents.jsonl")),
            "packages": {name: importlib.metadata.version(name) for name in
                         ["numpy", "scipy", "scikit-learn", "PyMuPDF", "requests", "openpyxl"]}}
    b = (ROOT / "eval/eval_functions.py").read_bytes()
    sha = hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
    if sha != "841185ed54cd1db2a6ebe72fd7e624edcb9586d9":
        raise RuntimeError("Исходный файл судьи изменён")
    info["judge_unchanged"] = True
    if require_index:
        index = PageIndex.load()
        items = read_jsonl(ROOT / "data/train.jsonl")[:24]
        hits, found, gold = 0, 0, 0
        for item in items:
            pages = index.search(item["question"], top_k=4)
            retrieved = {(p["doc_id"], p["page"]) for p in pages}
            expected = {(c["doc_id"], p) for c in item["gold_context"] for p in c["pages"]}
            hits += len(retrieved & expected); found += len(retrieved); gold += len(expected)
        info["retrieval_train24_only"] = {"micro_recall": hits / gold, "micro_precision": hits / found,
                                           "note": "Только диагностика поиска на train, не итоговые test-метрики"}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="store_true")
    check(parser.parse_args().index)
