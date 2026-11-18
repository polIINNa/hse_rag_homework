"""Индексация: эмбеддинги всех страниц → index/page_embeddings.npy (BM25 и TF-IDF строятся в памяти при запуске RAG).

Из корня репозитория:  python -m rag.index
По умолчанию модель deepvk/USER-bge-m3 считается локально (на GPU — минуты). Прогресс кэшируется
в index/embedding_cache.npz — при обрыве просто запустите ещё раз.
"""

import hashlib

import numpy as np

from rag.config import EMB_MODEL, EMBEDDER, INDEX_DIR
from rag.embeddings import get_embedder
from rag.retriever import load_pages, page_header

BATCH = 256


def page_text_for_embedding(page):
    # «хлебные крошки»: банк, отчёт и номер страницы в тексте чанка
    return page_header(page) + "\n" + page["text"]


def main(rebuild=False):
    INDEX_DIR.mkdir(exist_ok=True)
    pages = load_pages()
    ready = INDEX_DIR / "page_embeddings.npy"
    if ready.exists() and not rebuild and len(np.load(ready, mmap_mode="r")) == len(pages):
        print(f"Индекс уже есть: {ready} — пропускаю (пересчитать: python -m rag.index --rebuild)")
        return
    cache_path = INDEX_DIR / "embedding_cache.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    texts = [page_text_for_embedding(p) for p in pages]
    keys = [hashlib.sha1(f"{EMBEDDER}|{EMB_MODEL}|{t}".encode("utf-8")).hexdigest() for t in texts]
    todo = [i for i, key in enumerate(keys) if key not in cache]
    print(f"страниц: {len(pages)}, уже в кэше: {len(pages) - len(todo)}, считаем: {len(todo)}")

    if todo:
        embedder = get_embedder()
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            for i, vector in zip(batch, embedder.embed_documents([texts[i] for i in batch])):
                cache[keys[i]] = vector
            np.savez(cache_path, **cache)
            print(f"  {min(start + BATCH, len(todo))}/{len(todo)}", flush=True)

    matrix = np.stack([cache[key] for key in keys]).astype(np.float32)
    np.save(INDEX_DIR / "page_embeddings.npy", matrix)
    print(f"Готово: {matrix.shape} → {INDEX_DIR / 'page_embeddings.npy'}")


if __name__ == "__main__":
    import sys

    main(rebuild="--rebuild" in sys.argv)
