"""HTTP requests to the course LLM gateway."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import sleep

from requests.exceptions import ConnectionError, SSLError, Timeout


def post_with_retry(post, url, **kwargs):
    """Retry temporary gateway errors without changing the request payload."""
    delays = (5, 10, 20, 40, 60)
    for attempt in range(len(delays) + 1):
        try:
            response = post(url, **kwargs)
        except (ConnectionError, Timeout) as error:
            if isinstance(error, SSLError) and "UNEXPECTED_EOF_WHILE_READING" not in str(error):
                raise
            if attempt == len(delays):
                raise
            delay = delays[attempt]
            print(f"LLM API: {type(error).__name__}, повтор через {delay} с", flush=True)
            sleep(delay)
            continue
        if response.status_code not in (429, 502, 503, 504) or attempt == len(delays):
            response.raise_for_status()
            return response

        delay = delays[attempt]
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                wait = float(retry_after)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    wait = (retry_at - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    wait = 0
            delay = max(delay, wait)
        print(f"LLM API: HTTP {response.status_code}, повтор через {delay:.0f} с", flush=True)
        response.close()
        sleep(delay)
