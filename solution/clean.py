"""Очистка текста страницы перед индексацией. outputs/pages.jsonl остаётся сырым."""

import re


def clean_text(text):
    """Убирает строки, где нет ничего, кроме номера страницы отчёта (1–3 цифры)."""
    lines = [line for line in text.splitlines() if not re.fullmatch(r"\s*\d{1,3}\s*", line)]
    return "\n".join(lines).strip()
