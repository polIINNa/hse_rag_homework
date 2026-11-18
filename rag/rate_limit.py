"""Общий лимит учебного API для RAG и судьи в одном процессе."""

import os
import threading
import time

_lock = threading.Lock()
_last_start = 0.0


def wait_for_slot():
    global _last_start
    rpm = float(os.getenv("LLM_RPM", "10"))
    if rpm <= 0:
        raise ValueError("LLM_RPM должен быть положительным")
    with _lock:
        remaining = 60 / rpm - (time.monotonic() - _last_start)
        if remaining > 0:
            time.sleep(remaining)
        _last_start = time.monotonic()
