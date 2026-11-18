"""Векторный индекс групп соседних страниц. Поиск затем уточняется до страницы."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .clients import Embedder
from .config import DATA, INDEX


def make_groups(pages, max_pages=4, max_chars=18_000):
    groups, current, parts = [], [], []
    def finish():
        if current:
            text = f"{current[0]['organization']}. Отчёт {current[0]['doc_id']}.\n" + "\n".join(parts)
            groups.append({"doc_id": current[0]["doc_id"], "pages": [p["page"] for p in current], "text": text})
    for page in pages:
        compact = re.sub(r"[ \t]+", " ", page["text"]).strip()
        # У страницы с очень плотной таблицей оставляем несколько отдельных фрагментов.
        if len(compact) > max_chars - 300:
            finish(); current, parts = [], []
            for start in range(0, len(compact), max_chars - 500):
                groups.append({"doc_id": page["doc_id"], "pages": [page["page"]],
                               "text": f"{page['organization']}. {page['doc_id']}. Страница {page['page']}.\n" + compact[start:start + max_chars - 300]})
            continue
        fragment = f"Страница {page['page']}:\n{compact}"
        if current and (len(current) >= max_pages or sum(map(len, parts)) + len(fragment) > max_chars - 300):
            finish(); current, parts = [], []
        current.append(page); parts.append(fragment)
    finish()
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-parse", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers: от 1 до 4")
    embedder = Embedder()
    documents = [json.loads(x) for x in (DATA / "documents.jsonl").read_text().splitlines()]
    pending = {d["doc_id"] for d in documents}
    records, vectors = [], []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending:
            available = [doc for doc in sorted(pending) if (INDEX / "documents" / (doc + ".jsonl")).exists()]
            if not available:
                if not args.wait_for_parse:
                    raise FileNotFoundError("Сначала python -m rag.parse")
                time.sleep(5)
                continue
            tasks = {}
            for doc in available:
                path = INDEX / "documents" / (doc + ".jsonl")
                pages = [json.loads(x) for x in path.read_text().splitlines()]
                for group in make_groups(pages):
                    tasks[pool.submit(embedder.embed, group["text"])] = group
                pending.remove(doc)
            for future in as_completed(tasks):
                group = tasks[future]
                vector = future.result()
                records.append(group); vectors.append(vector)
                if len(records) % 25 == 0:
                    print(f"Эмбеддинги: {len(records)}, прошло {(time.monotonic()-started)/60:.1f} мин.", flush=True)
    order = sorted(range(len(records)), key=lambda i: (records[i]["doc_id"], records[i]["pages"][0], records[i]["text"]))
    np.save(INDEX / "vectors.npy", np.asarray([vectors[i] for i in order], dtype=np.float32), allow_pickle=False)
    (INDEX / "groups.jsonl").write_text("".join(json.dumps(records[i], ensure_ascii=False) + "\n" for i in order))
    meta = {"dimension": 1024, "groups": len(records), "documents": len(documents),
            "max_pages_per_group": 4, "max_chars": 18000,
            "groups_sha256": hashlib.sha256((INDEX / "groups.jsonl").read_bytes()).hexdigest()}
    (INDEX / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Готово:", meta, flush=True)


if __name__ == "__main__":
    main()

