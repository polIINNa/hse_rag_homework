"""Эмбеддер deepvk/USER-bge-m3: локально (sentence-transformers) или веб-сервис курса (как на семинаре 1)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from rag.config import EMB_MAX_CHARS, EMB_MAX_TOKENS, EMB_MODEL, EMB_URL, EMBEDDER, require


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-9)


class LocalEmbeddings:
    """Модель с Hugging Face. Ключ не нужен; при первом запуске скачивается ~2,3 ГБ. На GPU — в fp16."""

    def __init__(self, model_name=EMB_MODEL):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = EMB_MAX_TOKENS
        if device == "cuda":
            self.model.half()
        self.batch_size = 16 if device == "cuda" else 4
        print(f"[embeddings] {model_name} на {device}")

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=True, convert_to_numpy=True)
        return [v.astype(np.float32) for v in vectors]


class CourseEmbeddings:
    """Веб-сервис курса: один текст (до 20 000 символов) — один запрос. Ключ EMB_KEY выдаёт преподаватель.

    Лимиты ключа: 60 запросов в минуту, не больше 4 одновременно; сверх лимита сервис отвечает 429
    с Retry-After. Поэтому запросы идут не чаще одного в секунду, а на 429 ждём столько, сколько просит сервис.
    """

    def __init__(self, url=EMB_URL, workers=4, per_minute=60):
        self.url = url
        self.headers = {"Authorization": f"Bearer {require('EMB_KEY')}"}
        self.workers = workers
        self.interval = 60.0 / per_minute
        self.next_slot = 0.0
        self.lock = threading.Lock()
        self.session = requests.Session()

    def _wait_slot(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - now))

    def _embed_one(self, text):
        last_error = None
        for attempt in range(8):
            self._wait_slot()
            try:
                response = self.session.post(f"{self.url}/embed", headers=self.headers,
                                             json={"text": text[:EMB_MAX_CHARS]}, timeout=60)
            except requests.RequestException as error:
                last_error = error
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 5)))
                continue
            if response.status_code in (500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                time.sleep(min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            return np.asarray(response.json()["embedding"], dtype=np.float32)
        raise RuntimeError(f"Эмбеддер курса не ответил: {last_error}")

    def embed_query(self, text):
        return _normalize(self._embed_one(text))

    def embed_documents(self, texts):
        with ThreadPoolExecutor(self.workers) as pool:
            return [_normalize(v) for v in pool.map(self._embed_one, texts)]


_embedder = None


def get_embedder():
    """Один эмбеддер на процесс: модель грузится один раз."""
    global _embedder
    if _embedder is None:
        _embedder = CourseEmbeddings() if EMBEDDER == "course" else LocalEmbeddings()
    return _embedder
