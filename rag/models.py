import os
from collections import deque
from functools import cache
from threading import Lock
from time import monotonic, sleep

import numpy as np
import requests
import truststore
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from rag import config

_embedding_lock = Lock()
_next_embedding_request = 0.0
_chat_lock = Lock()
_chat_starts = deque()


@cache
def get_session():
    truststore.inject_into_ssl()
    session = requests.Session()
    retries = Retry(
        total=config.RETRY_COUNT,
        connect=0,
        backoff_factor=config.RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504, 529],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _key(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Задайте {name} в окружении или в .env.")
    return value


def _post(url, headers, payload):
    response = get_session().post(
        url,
        headers=headers,
        json=payload,
        timeout=config.HTTP_TIMEOUT,
        allow_redirects=False,
    )
    response.raise_for_status()
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"Неожиданный HTTP {response.status_code} от {url}.")
    return response.json()


def embed(texts):
    global _next_embedding_request
    if not texts:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)
    if any(
        not text.strip() or len(text) > config.EMBEDDING_MAX_CHARS for text in texts
    ):
        raise ValueError("Для эмбеддинга нужен непустой текст до 20 000 символов.")
    headers = {"Authorization": f"Bearer {_key('EMB_KEY')}"}
    vectors = []
    for text in texts:
        with _embedding_lock:
            sleep(max(0, _next_embedding_request - monotonic()))
            _next_embedding_request = monotonic() + config.EMBEDDING_INTERVAL
            result = _post(config.EMBEDDING_URL, headers, {"text": text})
        vector = np.asarray(result["embedding"], dtype=np.float32)
        if result["dim"] != config.EMBEDDING_DIM or vector.shape != (
            config.EMBEDDING_DIM,
        ):
            raise ValueError(f"Ожидался эмбеддинг размерности {config.EMBEDDING_DIM}.")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Сервис вернул невалидный эмбеддинг.")
        vectors.append(vector / norm)
    return np.stack(vectors)


def _wait_for_chat_capacity(count):
    while True:
        now = monotonic()
        while _chat_starts and now - _chat_starts[0] >= config.CHAT_WINDOW:
            _chat_starts.popleft()
        overflow = len(_chat_starts) + count - config.CHAT_RPM
        if overflow <= 0:
            return
        sleep(_chat_starts[overflow - 1] + config.CHAT_WINDOW - now)


def wait_for_chat_capacity(count):
    """Allow calls for a full question before starting its latency timer."""
    if not 1 <= count <= config.CHAT_RPM:
        raise ValueError("Число вызовов должно укладываться в лимит модели.")
    with _chat_lock:
        _wait_for_chat_capacity(count)


def chat(system, message, model=config.ANSWER_MODEL, max_tokens=config.MAX_TOKENS):
    headers = {"x-api-key": _key("LLM_API_KEY"), "anthropic-version": "2023-06-01"}
    with _chat_lock:
        _wait_for_chat_capacity(1)
        _chat_starts.append(monotonic())
    result = _post(
        config.CHAT_URL,
        headers,
        {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": message}],
            "temperature": config.TEMPERATURE,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        },
    )
    text = "".join(
        block["text"] for block in result["content"] if block["type"] == "text"
    )
    if not text.strip():
        raise ValueError(f"Модель {model} не вернула текст ответа.")
    return text
