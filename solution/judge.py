"""Судья из eval/eval_functions.py с повторами: модель и промпт те же, добавлены только повторы запроса."""

import time

import requests

from eval.eval_functions import score_answer
from solution.generate import ATTEMPTS, retry_wait, wait_for_llm_slot


def judge(gold_answer, answer):
    """score_answer, который соблюдает общий лимит запросов к LLM и повторяет запрос при 429 и сбоях сети."""
    for attempt in range(ATTEMPTS):
        wait_for_llm_slot()
        try:
            return score_answer(gold_answer, answer)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(retry_wait(None, attempt))
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 429 or attempt == ATTEMPTS - 1:
                raise
            time.sleep(retry_wait(error.response, attempt))
