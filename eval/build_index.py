"""Сборка v2-индекса (чанки + эмбеддинги) с прогрессом и возобновлением.

Зачем: SentenceTransformer на CPU считает эмбеддинги медленно (несколько чанков/сек),
а старая сборка писала файлы только в самом конце — казалось, что процесс "завис".
Этот скрипт:
1) сразу извлекает чанки и сохраняет eval/cache/chunks_v2.jsonl;
2) считает эмбеддинги батчами и периодически сохраняет прогресс
   (eval/cache/embeddings_v2.partial.npy + eval/cache/embeddings_v2.progress.json);
3) при повторном запуске продолжает с последнего батча;
4) в конце сохраняет eval/cache/embeddings_v2.npy.

Запуск (из корня репозитория):
  python -m eval.build_index --batch-size 32
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from eval.rag_example import RAG


def main(batch_size=32, cache_dir=None):
    sys_rag = RAG.__new__(RAG)  # не вызываем __init__, чтобы не запустить авто-сборку
    root = Path(__file__).resolve().parents[1]
    sys_rag.root = root
    sys_rag.data_dir = root / "data"
    sys_rag.reports_dir = sys_rag.data_dir / "reports"
    sys_rag.docs_path = sys_rag.data_dir / "documents.jsonl"
    sys_rag.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent / "cache"
    sys_rag.cache_dir.mkdir(parents=True, exist_ok=True)
    sys_rag.chunks_path = sys_rag.cache_dir / "chunks_v2.jsonl"
    sys_rag.emb_path = sys_rag.cache_dir / "embeddings_v2.npy"
    sys_rag.model = None
    sys_rag.chunks = []

    # 1) Чанки
    if sys_rag.chunks_path.exists():
        sys_rag.chunks = [json.loads(l) for l in sys_rag.chunks_path.open(encoding="utf-8") if l.strip()]
        print(f"[chunks] loaded {len(sys_rag.chunks)} from {sys_rag.chunks_path}", flush=True)
    else:
        print("[chunks] extracting from PDFs...", flush=True)
        t0 = time.time()
        sys_rag.chunks = sys_rag._extract_chunks()
        with sys_rag.chunks_path.open("w", encoding="utf-8") as f:
            for c in sys_rag.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[chunks] saved {len(sys_rag.chunks)} chunks in {time.time() - t0:.1f}s", flush=True)

    if not sys_rag.chunks:
        print("[emb] no chunks, nothing to embed")
        return

    partial_path = sys_rag.cache_dir / "embeddings_v2.partial.npy"
    progress_path = sys_rag.cache_dir / "embeddings_v2.progress.json"

    model = sys_rag._get_model()
    dim = model.get_sentence_embedding_dimension()
    n = len(sys_rag.chunks)

    start_idx = 0
    if partial_path.exists() and progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            start_idx = int(progress.get("done", 0))
            partial = np.load(partial_path)
            if partial.ndim == 2 and partial.shape[0] >= start_idx and partial.shape[1] == dim:
                print(f"[emb] resuming from {start_idx}/{n}", flush=True)
            else:
                start_idx = 0
        except Exception:
            start_idx = 0

    if start_idx == 0:
        emb = np.zeros((n, dim), dtype=np.float32)
    else:
        emb = np.load(partial_path)[:n]
        if emb.shape[0] < n:
            grew = np.zeros((n, dim), dtype=np.float32)
            grew[: emb.shape[0]] = emb
            emb = grew

    print(f"[emb] model={os.environ.get('EMB_MODEL', 'intfloat/multilingual-e5-small')} dim={dim} total={n}", flush=True)
    t0 = time.time()
    done = start_idx
    for begin in range(start_idx, n, batch_size):
        end = min(n, begin + batch_size)
        texts = [f"passage: {c['text'][:2000]}" for c in sys_rag.chunks[begin:end]]
        vecs = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        emb[begin:end] = np.asarray(vecs, dtype=np.float32)
        done = end
        np.save(partial_path, emb[:done])
        progress_path.write_text(json.dumps({"done": done, "total": n}), encoding="utf-8")
        elapsed = time.time() - t0
        rate = (done - start_idx) / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"[emb] {done}/{n} ({done / n * 100:.1f}%), {rate:.1f} chunks/s, ETA {eta / 60:.1f} min", flush=True)

    np.save(sys_rag.emb_path, emb)
    try:
        partial_path.unlink()
        progress_path.unlink()
    except OSError:
        pass
    print(f"[emb] saved {sys_rag.emb_path} shape={emb.shape} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    main(batch_size=args.batch_size, cache_dir=args.cache_dir)
