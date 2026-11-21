"""Определение организации и отчёта по тексту вопроса.

Вопросы почти всегда называют банк и дату («за шесть месяцев, закончившихся 30 июня 2025 года»,
«по состоянию на 31 декабря 2023 года», «согласно отчётности за 2024 год»). По ним сужаем поиск
до нужных отчётов. Отчёт содержит и сравнительные данные прошлого периода, поэтому для сравнения
двух дат берём отчёт за более позднюю, а второй отчёт — только если первый её не покрывает.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ORGANIZATIONS = {
    "alfa": r"альфа",
    "domrf": r"дом\.?\s?рф|домрф",
    "gpb": r"газпромбанк|\bнпф\b",
    "mkb": r"\bмкб\b|московск\w* кредитн",
    "rshb": r"россельхоз|\bрсхб\b|сельскохозяйственн\w* банк",
    "sber": r"сбербанк|\bсбер(а|у|ом|е)?\b",
    "sovkom": r"совком",
    "tbank": r"(?<![а-яе])т-?\s?банк|тинькофф|\btbank\b|t-bank",
    "vtb": r"\bвтб\b",
}

HALF_YEAR = r"шест\w* месяц|6 месяц|полугод|30 июня|30\.06\.|промежуточн"
SHORT_REPORT_PAGES = 30


def load_documents():
    with (ROOT / "data" / "documents.jsonl").open(encoding="utf-8") as source:
        return {d["doc_id"]: d for d in map(json.loads, source) if d}


DOCUMENTS = load_documents()


def normalize(question):
    return question.lower().replace("ё", "е")


def find_organizations(question):
    text = normalize(question)
    return [org for org, pattern in ORGANIZATIONS.items() if re.search(pattern, text)]


def find_dates(text):
    """Отчётные даты из вопроса: (год, "6m" | "annual")."""
    dates = set()
    for y in re.findall(r"30 июня (20\d\d)|30\.06\.(20\d\d)", text):
        dates.add((int("".join(y)), "6m"))
    for y in re.findall(r"(?:шест\w*|6) месяц\w*[^.?;]{0,60}?(20\d\d)|полугоди\w* (20\d\d)", text):
        dates.add((int("".join(y)), "6m"))
    for y in re.findall(r"31 декабря (20\d\d)|31\.12\.(20\d\d)", text):
        dates.add((int("".join(y)), "annual"))
    for y in re.findall(r"(?:за|в|с|по) (20\d\d)(?:\s*(?:и|-|–|—|по)\s*20\d\d)? год", text):
        dates.add((int(y), "annual"))
    for y in re.findall(r"(?:и|-|–|—|по) (20\d\d) год", text):
        dates.add((int(y), "annual"))
    years = {int(y) for y in re.findall(r"\b(20[12]\d)\b", text)}
    half_year = bool(re.search(HALF_YEAR, text))
    for year in years - {d[0] for d in dates}:
        dates.add((year, "6m" if half_year else "annual"))
    return dates


def report_date(text):
    """Отчёт, прямо названный в вопросе: «согласно отчётности за 2024 год»."""
    m = re.search(r"отчет\w*[^.?;]{0,60}?за (?:(?:шест\w*|6) месяц\w*[^.?;]{0,40}?)?(20\d\d)", text)
    if not m:
        return None
    return int(m.group(1)), "6m" if re.search(r"месяц", m.group(0)) else "annual"


def covers(doc_id, date):
    """Есть ли данные на дату в отчёте (своя дата или сравнительная)."""
    _, year, kind = doc_id.split("_")
    year = int(year)
    if kind == "annual":
        return date in {(year, "annual"), (year - 1, "annual")}
    return date in {(year, "6m"), (year - 1, "6m"), (year - 1, "annual")}


def doc_for(org, date):
    year, kind = date
    if kind == "6m":
        options = [f"{org}_{year}_6m", f"{org}_{year + 1}_6m", f"{org}_{year}_annual"]
    else:
        options = [f"{org}_{year}_annual", f"{org}_{year + 1}_annual", f"{org}_{year + 1}_6m"]
    return next((d for d in options if d in DOCUMENTS), None)


def route(question):
    """Вернуть список doc_id для поиска (пустой — организация не распознана)."""
    text = normalize(question)
    organizations = find_organizations(question)
    dates = sorted(find_dates(text), key=lambda d: (d[0], d[1] == "annual"), reverse=True)
    explicit = report_date(text)
    docs = []
    for org in organizations:
        if not dates and not explicit:
            docs.extend(d for d in DOCUMENTS if d.startswith(org + "_"))
            continue
        picked = []
        for date in ([explicit] if explicit else []) + dates:
            if any(covers(d, date) for d in picked):
                continue
            doc = doc_for(org, date)
            if doc and doc not in picked:
                picked.append(doc)
        # Короткие «обобщённые» отчёты содержат только основные формы без примечаний:
        # подробности за этот период есть в сравнительных данных следующего полного отчёта.
        for doc in list(picked):
            if DOCUMENTS[doc]["pages"] < SHORT_REPORT_PAGES:
                _, year, kind = doc.split("_")
                fallback = doc_for(org, (int(year) + 1, kind))
                if fallback and fallback not in picked:
                    picked.append(fallback)
        docs.extend(picked)
    return list(dict.fromkeys(docs))


def org_documents(organizations):
    return [d for d in DOCUMENTS if d.split("_")[0] in organizations]
