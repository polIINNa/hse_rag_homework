"""Подзапросы для извлечения двух частей расчёта или названия модели."""

import re

_RATIO = re.compile(
    r"\b(?:со)?отношени[ея]\s+(.+?)\s+к\s+(.+?)"
    r"(?:\s+(?:по\s+(?:обобщ[её]н|данным)|согласно\b|по\s+состоянию\b)|[?]|$)", re.I,
)
_MODEL = re.compile(r"\bкак(?:ую|ая|ие)\s+модел[ьи]", re.I)
_SOURCE_CLAUSE = re.compile(r"\s+(?:согласно|в\s+(?:[а-яё]+\s+){0,2}отч[её]тности)\b", re.I)


def evidence_focuses(question):
    ratio = _RATIO.search(question)
    if ratio:
        return list(ratio.groups())
    if _MODEL.search(question):
        prefix = _SOURCE_CLAUSE.split(question, maxsplit=1)[0]
        if prefix != question:
            return [prefix]
    return []
