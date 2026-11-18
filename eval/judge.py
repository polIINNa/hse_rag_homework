from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from time import monotonic, sleep

from requests import HTTPError

from eval.eval_functions import score_answer as official_score_answer


_REQUEST_INTERVAL = 1.05
_MAX_ATTEMPTS = 3
_lock = Lock()
_next_request = 0.0


def _retry_delay(value):
    if value:
        try:
            return int(value)
        except ValueError:
            try:
                return (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
    return 60.0


def score_answer(gold_answer, answer):
    global _next_request

    with _lock:
        for attempt in range(_MAX_ATTEMPTS):
            remaining = _next_request - monotonic()
            if remaining > 0:
                sleep(remaining)
            _next_request = monotonic() + _REQUEST_INTERVAL
            try:
                return official_score_answer(gold_answer, answer)
            except HTTPError as error:
                response = error.response
                if response is None or response.status_code != 429:
                    raise
                retry_delay = _retry_delay(response.headers.get("Retry-After"))
                _next_request = max(_next_request, monotonic() + retry_delay)
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
