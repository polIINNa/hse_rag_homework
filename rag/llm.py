"""Anthropic-compatible gateway; all generator/judge calls share one limiter."""
import json
import os
from pathlib import Path
import threading
import time
import requests
from rag.config import ROOT

class RateLimiter:
    def __init__(self, interval=None):
        self.interval = float(interval if interval is not None else os.getenv('RAG_API_INTERVAL', '6.2'))
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()

LIMITER = RateLimiter()

def retry_call(call, attempts=5):
    for attempt in range(attempts):
        LIMITER.wait()
        try:
            return call()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status not in (429, 500, 502, 503, 504) or attempt + 1 == attempts:
                # Do not print HTTP body, headers or credentials.
                raise RuntimeError(f'LLM HTTP status {status}; request failed') from None
            time.sleep(min(45, 6 * 2 ** attempt))
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 == attempts:
                raise RuntimeError('LLM network failure after retries') from None
            time.sleep(min(30, 3 * 2 ** attempt))

def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result

class LLM:
    def __init__(self):
        self.key = os.getenv('RAG_API_KEY') or os.getenv('LLM_API_KEY')
        if not self.key:
            raise ValueError('Set LLM_API_KEY in .env')
        self.model = os.getenv('RAG_MODEL', 'deepseek-flash')
        self.url = os.getenv('RAG_API_URL', 'https://claude.gigaparsers.ru/v1/messages')
        self.usage = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}
        self.session = requests.Session()

    def complete(self, system, user, max_tokens=1400):
        def request():
            response = self.session.post(self.url, headers={'x-api-key': self.key, 'anthropic-version': '2023-06-01'},
                json={'model': self.model, 'max_tokens': max_tokens, 'temperature': 0,
                      'thinking': {'type': 'disabled'}, 'system': system,
                      'messages': [{'role': 'user', 'content': user}]}, timeout=(20, 150))
            response.raise_for_status()
            body = response.json()
            for metric in ('input_tokens', 'output_tokens'):
                self.usage[metric] += body.get('usage', {}).get(metric, 0)
            self.usage['calls'] += 1
            return ''.join(block['text'] for block in body['content'] if block['type'] == 'text')
        return retry_call(request)
