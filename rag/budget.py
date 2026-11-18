"""Общий лимит для генератора и неизменённого судьи преподавателя."""
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import requests

from rag.common import ROOT, load_env


class ApiBudget:
    def __init__(self, directory=None, limit=None):
        load_env()
        key_id = hashlib.sha256(os.environ.get("LLM_API_KEY", "unset").encode()).hexdigest()[:12]
        self.directory = directory or Path(os.getenv("RAG_BUDGET_DIR", ROOT / ".private"))
        self.limit = int(limit or os.getenv("RAG_DAILY_BUDGET", "1700000"))
        self.key_id = key_id
        self.lock = threading.RLock()

    def _path(self):
        return self.directory / f"usage_{self.key_id}.json"

    @contextmanager
    def _process_lock(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"usage_{self.key_id}.lock"
        with path.open("a+b") as lock:
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0"); lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def request(self, send, url, **kwargs):
        if url != "https://claude.gigaparsers.ru/v1/messages":
            raise ValueError("Бюджетный клиент предназначен только для курсового шлюза")
        body = kwargs["json"]
        # Число UTF-8 байтов — консервативная граница входных токенов byte-level tokenizer.
        upper = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + body.get("max_tokens", 0)
        with self.lock, self._process_lock():
            path = self._path()
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "charged_tokens": 0, "requests": [], "unmeasured_reservations": 0,
            }
            recent_day = [r for r in state["requests"] if time.time() - r["at"] < 86400]
            state["charged_tokens"] = sum(r["tokens"] for r in recent_day)
            state["unmeasured_reservations"] = sum(not r["measured"] for r in recent_day)
            state["window"] = "rolling_24_hours"
            if state["charged_tokens"] + upper > self.limit:
                raise RuntimeError("Достигнут защитный дневной бюджет. Готовые строки сохранены; продолжайте на следующий день с --resume")
            recent = [r["at"] for r in state["requests"] if time.time() - r["at"] < 60.2]
            if len(recent) >= 10:
                time.sleep(max(0, recent[-10] + 60.2 - time.time()))
            # Равномерный темп без всплесков, общий для генератора и судьи.
            if state["requests"]:
                delay = state["requests"][-1]["at"] + 6.2 - time.time()
                if delay > 0:
                    time.sleep(delay)
            record = {"at": time.time(), "model": body.get("model"), "tokens": upper, "measured": False}
            state["requests"].append(record)
            state["charged_tokens"] += upper
            state["unmeasured_reservations"] += 1
            self._save(path, state)
            response = send(url, **kwargs)
            try:
                usage = response.json().get("usage")
                keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
                if isinstance(usage, dict) and all(k in usage for k in keys[:2]) and all(
                    type(usage.get(k, 0)) is int and usage.get(k, 0) >= 0 for k in keys
                ):
                    used = sum(usage.get(k, 0) for k in keys)
                    state["charged_tokens"] += used - upper
                    state["unmeasured_reservations"] -= 1
                    record.update(tokens=used, measured=True)
                    self._save(path, state)
            except (ValueError, TypeError):
                pass  # При неизвестном расходе консервативная резервация остаётся.
            return response

    @staticmethod
    def _save(path, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


_BUDGET = None


def budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = ApiBudget()
    return _BUDGET


def call_official_judge(function, gold_answer, answer):
    # Меняется только HTTP-транспорт на время синхронного вызова. Аргументы запроса,
    # модель и JUDGE_PROMPT остаются побайтно теми, что задаёт eval_functions.py.
    original_post = requests.post

    def measured_post(url, **kwargs):
        if url == "https://claude.gigaparsers.ru/v1/messages":
            return budget().request(original_post, url, **kwargs)
        return original_post(url, **kwargs)

    with patch("requests.post", measured_post):
        return function(gold_answer, answer)
