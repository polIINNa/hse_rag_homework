"""Клиент курсового LLM-шлюза (Anthropic Messages API, модель deepseek-flash) с учётом лимита запросов.

Шлюз пропускает LLM_RPM запросов в минуту на ключ (сверх — 429) и считает вместе запросы RAG и судьи
из eval/eval_functions.py, который при 429 падает. Поэтому клиент держит скользящее окно на LLM_RPM - 1
запросов и после каждого ответа RAG резервирует в окне место под вызов судьи.
"""

import os
import threading
import time
from collections import deque

import requests
from langsmith import traceable

WINDOW = 60.0


class RateWindow:
    def __init__(self, limit):
        self.limit = max(1, limit)
        self.events = deque()
        self.lock = threading.Lock()
        self.waited = threading.local()  # сколько секунд текущий поток простоял в ожидании лимита

    def _expire(self, now):
        while self.events and now - self.events[0] >= WINDOW:
            self.events.popleft()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self._expire(now)
                if len(self.events) < self.limit:
                    self.events.append(now)
                    return
                wait = WINDOW - (now - self.events[0]) + 0.05
            self.waited.total = getattr(self.waited, "total", 0.0) + wait
            time.sleep(wait)

    def take_waited(self):
        total, self.waited.total = getattr(self.waited, "total", 0.0), 0.0
        return total

    def reserve(self, delay=2.0):
        """Занять место под чужой запрос (судья), который придёт сразу после ответа RAG."""
        with self.lock:
            now = time.monotonic()
            self._expire(now)
            self.events.append(now + delay)

    def penalize(self):
        """Шлюз ответил 429: считаем окно заполненным."""
        with self.lock:
            now = time.monotonic()
            while len(self.events) < self.limit:
                self.events.append(now)


class Gateway:
    def __init__(self, base_url=None, api_key=None, model=None, rpm=None):
        self.url = (base_url or os.environ.get("LLM_BASE_URL", "https://claude.gigaparsers.ru")).rstrip("/") + "/v1/messages"
        self.headers = {"x-api-key": api_key or os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"}
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-flash")
        self.window = RateWindow(int(rpm or os.environ.get("LLM_RPM", "10")) - 1)
        self.session = requests.Session()

    @traceable(name="deepseek-flash", run_type="llm", metadata={"ls_provider": "anthropic"})
    def create(self, system, messages, tools=None, tool_choice=None, max_tokens=3000, thinking=None):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                "messages": messages, "thinking": thinking or {"type": "disabled"}}
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        for attempt in range(10):
            self.window.acquire()
            try:
                response = self.session.post(self.url, headers=self.headers, json=body, timeout=180)
            except requests.exceptions.RequestException:
                if attempt == 9:
                    raise
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code == 429:
                self.window.penalize()
                continue
            if response.status_code >= 500 and attempt < 9:
                time.sleep(min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()

    def reserve_judge_slot(self):
        self.window.reserve()

    def cooldown(self):
        """Ключ мог только что использоваться другим процессом: ждём, пока окно лимита на шлюзе очистится."""
        print(f"Жду {WINDOW:.0f} с, чтобы очистилось окно лимита шлюза ({self.window.limit + 1} запросов/мин)...",
              flush=True)
        time.sleep(WINDOW + 1)
