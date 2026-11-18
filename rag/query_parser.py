"""Разбор вопроса в метаданные: банк(и), годы, период отчётности → кандидаты doc_id для фильтра поиска."""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from rag.config import DOCUMENTS_PATH

BANK_PATTERNS = {
    "sber": r"сбербанк|(?<![а-яё])сбер(?:а|у|ом|е)?(?![а-яё])",
    "vtb": r"(?<![а-яё])втб(?![а-яё])",
    "tbank": r"(?<![а-яё])т-?\s?банк|тинькофф|tbank|t-bank",
    "domrf": r"дом\.?\s?рф|домрф",
    "mkb": r"(?<![а-яё])мкб(?![а-яё])|московск\w*\s+кредитн\w*\s+банк",
    "rshb": r"россельхоз|(?<![а-яё])рсхб(?![а-яё])|сельскохозяйственн\w*\s+банк",
    "sovkom": r"совком",
    "alfa": r"альфа",
    "gpb": r"газпромбанк|(?<![а-яё])нпф(?![а-яё])|пенсионн\w*\s+фонд",
}
SIX_MONTHS = r"шест\w*\s+месяц|6\s+месяц|полугоди|30\s+июня|промежуточн|1\s+полугод"
REPORT_YEAR = (r"(?:отч[её]тност\w*|отч[её]т\w*|отчетност\w*|примечани\w*)\s+(?:\w+\s+){0,3}?за\s+"
               r"(?:первое\s+полугодие\s+|6\s+месяцев\s+|шесть\s+месяцев\s+)?(20\d\d)")


MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8,
          "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
YEAR_END = r"31\s+декабря|31\.12\.|за\s+(?:20\d\d\s+)?год(?![а-яё])|по\s+итогам\s+20\d\d|годов|в\s+20\d\d\s+году|с\s+20\d\d\s+по\s+20\d\d"


@dataclass
class QueryMeta:
    banks: list = field(default_factory=list)
    years: list = field(default_factory=list)         # все годы из вопроса
    report_years: list = field(default_factory=list)  # «отчётность за 2024 год»
    six_months: bool = False
    year_end: bool = False                           # «31 декабря», «за год»
    latest_date: tuple | None = None                 # (год, месяц) самой поздней даты в вопросе

    @property
    def target_year(self):
        if self.report_years:
            return max(self.report_years)
        if self.latest_date:
            return self.latest_date[0]
        return max(self.years) if self.years else None

    @property
    def target_period(self):
        if self.latest_date and self.latest_date[1] in (6, 12):
            return "6m" if self.latest_date[1] == 6 else "annual"
        if self.six_months and not self.year_end:
            return "6m"
        if self.year_end and not self.six_months:
            return "annual"
        return None

    def describe(self):
        return {"banks": self.banks, "years": self.years, "report_years": self.report_years,
                "target_year": self.target_year, "target_period": self.target_period}


@lru_cache(maxsize=1)
def documents():
    docs = [json.loads(line) for line in DOCUMENTS_PATH.open(encoding="utf-8") if line.strip()]
    for d in docs:
        bank, year, period = d["doc_id"].split("_")
        d.update(bank=bank, year=int(year), period=period)
    return docs


def parse_question(question: str) -> QueryMeta:
    q = question.lower().replace("ё", "е")
    banks = [b for b, pattern in BANK_PATTERNS.items() if re.search(pattern, q)]
    # банк, упомянутый раньше в тексте, — первым
    banks.sort(key=lambda b: re.search(BANK_PATTERNS[b], q).start())
    years = sorted({int(y) for y in re.findall(r"(?<!\d)(20[12]\d)(?!\d)", q)})
    report_years = sorted({int(y) for y in re.findall(REPORT_YEAR, q)})
    dates = [(int(y), MONTHS[m]) for m, y in re.findall(r"\d{1,2}\s+(" + "|".join(MONTHS) + r")\s+(20\d\d)", q)]
    dates += [(int(y), int(m)) for m, y in re.findall(r"\d{2}\.(\d{2})\.(20\d\d)", q)]
    return QueryMeta(
        banks=banks,
        years=years,
        report_years=report_years,
        six_months=bool(re.search(SIX_MONTHS, q)),
        year_end=bool(re.search(YEAR_END, q)),
        latest_date=max(dates) if dates else None,
    )


NOISE = [r"сбербанк\w*", r"(?<![а-я])втб(?![а-я])", r"(?<![а-я])т-?банк\w*", r"дом\.?\s?рф", r"домрф\w*",
         r"(?<![а-я])мкб(?![а-я])", r"московск\w*\s+кредитн\w*\s+банк\w*", r"россельхозбанк\w*", r"совкомбанк\w*",
         r"альфа-банк\w*", r"(?<![а-я])нпф(?![а-я])", r"газпромбанк\w*", r"(?<![а-я])(?:ао|пао|зао)(?![а-я])",
         r"банк\w*", r"групп\w*",
         r"(?<!\d)(?:20\d\d|31|30|1)(?!\d)", r"(?<![а-я])(?:года?|году|декабря|июня|состоянию|согласно)(?![а-я])"]


def search_terms(question: str) -> str:
    """Запрос для поиска без банка и дат: их уже учёл фильтр по метаданным, а в тексте они шумят."""
    q = question.lower().replace("ё", "е")
    for pattern in NOISE:
        q = re.sub(pattern, " ", q)
    return re.sub(r"\s+", " ", q).strip()


def doc_prior(meta: QueryMeta, doc: dict) -> float:
    """Априорная уместность документа для вопроса: 0..1 (банк — жёсткий фильтр, год и период — мягкие)."""
    if meta.banks and doc["bank"] not in meta.banks:
        return 0.0
    score = 1.0
    year = meta.target_year
    if year is not None:
        if doc["year"] == year:
            score *= 1.0
        elif doc["year"] == year + 1 and not meta.report_years:
            score *= 0.35                         # данные прошлого года есть в следующем отчёте как сравнительные
        elif doc["year"] in meta.years:
            score *= 0.6                          # вопрос про изменение между годами
        else:
            score *= 0.05
    period = meta.target_period
    if period:
        score *= 1.0 if doc["period"] == period else 0.3
    return score


def candidate_docs(question: str, bank: str | None = None, year: int | None = None,
                   period: str | None = None) -> dict:
    """{doc_id: prior}. Явные bank/year/period (их передаёт агент) — жёсткие фильтры."""
    meta = parse_question(question)
    if bank:
        meta.banks = [bank]
    result = {}
    for doc in documents():
        if bank and doc["bank"] != bank:
            continue
        if year and doc["year"] != year:
            continue
        if period and doc["period"] != period:
            continue
        prior = doc_prior(meta, doc)
        if prior > 0:
            result[doc["doc_id"]] = prior
    return result
