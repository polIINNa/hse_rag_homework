"""Клиенты сервисов курса: лимиты, Retry-After и кэш эмбеддингов."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import sys

import httpx
import numpy as np

from .config import CACHE, required


class RateLimiter:
    def __init__(self, rpm: float):
        if rpm <= 0:
            raise ValueError("rpm должен быть положительным")
        self.interval = 60 / rpm
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        time.sleep(delay)


def retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    return min(2 ** attempt, 60)


def request(client, limiter, method, url, **kwargs):
    for attempt in range(8):
        limiter.wait()
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            if attempt == 7:
                raise
            print(f"Сеть: {type(exc).__name__}, повтор {attempt+1}/8", file=sys.stderr, flush=True)
            time.sleep(retry_delay(None, attempt))
            continue
        if response.status_code not in {408, 429, 500, 502, 503, 504, 529}:
            response.raise_for_status()
            return response.json()
        if attempt == 7:
            response.raise_for_status()
        delay = retry_delay(response.headers.get("Retry-After"), attempt)
        print(f"API: HTTP {response.status_code}, повтор {attempt+1}/8 через {delay:.1f}s", file=sys.stderr, flush=True)
        if delay > 300:
            raise RuntimeError(f"Сервис требует ожидания {delay:.0f}s. Прогон можно продолжить позже.")
        time.sleep(delay)
    raise RuntimeError("Недостижимая ветка")


class Embedder:
    def __init__(self):
        self.base = os.getenv("EMB_BASE_URL", "https://embeddings.parsers360.ru:10443").rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(90, connect=15),
                                   limits=httpx.Limits(max_connections=4, max_keepalive_connections=4, keepalive_expiry=300),
                                   headers={"Authorization": "Bearer " + required("EMB_KEY")})
        self.limiter = RateLimiter(float(os.getenv("EMB_RPM", "55")))
        self.cache = CACHE / "embeddings"
        self.cache.mkdir(parents=True, exist_ok=True)

    def embed(self, text: str, *, use_cache: bool = True) -> np.ndarray:
        if not text.strip() or len(text) > 20_000:
            raise ValueError("Эмбеддер принимает от 1 до 20 000 символов")
        digest = hashlib.sha256((self.base + "\n" + text).encode()).hexdigest()
        path = self.cache / (digest + ".npy")
        if use_cache and path.exists():
            return np.load(path, allow_pickle=False)
        result = request(self.client, self.limiter, "POST", self.base + "/embed", json={"text": text})
        vector = np.asarray(result["embedding"], dtype=np.float32)
        if result.get("dim") != 1024 or vector.shape != (1024,) or not np.isfinite(vector).all():
            raise ValueError("Эмбеддер вернул некорректный вектор")
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            raise ValueError("Эмбеддер вернул нулевой вектор")
        vector /= norm
        if use_cache:
            temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, vector, allow_pickle=False)
            temporary.replace(path)
        return vector

    def embed_many(self, texts, workers=3):
        if not 1 <= workers <= 4:
            raise ValueError("Допустимо не больше 4 параллельных запросов")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return np.asarray(list(pool.map(self.embed, texts)))


class LLM:
    def __init__(self):
        self.base = os.getenv("LLM_BASE_URL", "https://claude.gigaparsers.ru").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "deepseek-flash")
        self.limiter = RateLimiter(float(os.getenv("LLM_RPM", "9")))
        self.client = httpx.Client(timeout=httpx.Timeout(120, connect=15),
                                  limits=httpx.Limits(max_connections=4, max_keepalive_connections=4, keepalive_expiry=300), headers={
            "x-api-key": required("LLM_API_KEY"), "anthropic-version": "2023-06-01",
        })

    def complete(self, system, content, max_tokens=2000):
        result = request(self.client, self.limiter, "POST", self.base + "/v1/messages", json={
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "thinking": {"type": "disabled"}, "system": system,
            "messages": [{"role": "user", "content": content}],
        })
        if result.get("stop_reason") == "max_tokens":
            raise ValueError("Ответ модели обрезан по max_tokens")
        text = "".join(block["text"] for block in result["content"] if block["type"] == "text").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return text

    def json(self, system, content, max_tokens=2000):
        return json.loads(self.complete(system, content, max_tokens))

