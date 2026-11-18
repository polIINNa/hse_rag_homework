from functools import lru_cache
from collections import Counter, defaultdict
import json
import re

import numpy as np
from pymorphy3 import MorphAnalyzer
from rank_bm25 import BM25Okapi

from .clients import Embedder
from .config import INDEX

ALIASES = {
    "alfa": r"альфа",
    "domrf": r"дом[.\s-]*рф",
    "gpb": r"газпромбанк|газпром.*фонд",
    "mkb": r"\bмкб\b|московск\w* кредитн",
    "rshb": r"россельхоз|\bрсхб\b",
    "sber": r"сбер",
    "sovkom": r"совком",
    "tbank": r"т[\s-]*банк|тинькофф|тинькоф",
    "vtb": r"\bвтб\b",
}
STOP = set("и в во на по из за к с со у о об от до для или как какой каков сколько составлять составить согласно данные отчет отчетность финансовый консолидировать группа банк год месяц рубль миллион миллиард тысяча ао пао зао этот тот быть который также величина размер значение общий весь состояние конец начало период закончиться июнь декабрь январь июль полугодие шесть три первое второй первое итоговый".split())
BANK_TERMS = set("дом рф сбербанк совкомбанк тбанк тинькофф втб мкб газпромбанк фонд альфа россельхозбанк".split())
MORPH = MorphAnalyzer()


@lru_cache(maxsize=100_000)
def lemma(word):
    return MORPH.parse(word)[0].normal_form


def tokens(text):
    words = re.findall(r"[а-яёa-z]+|\d+(?:[.,]\d+)?", text.lower().replace("ё", "е"))
    return [w for token in words if len(token) > 1 and (w := lemma(token)) not in STOP]


def page_key(page):
    return f"{page['doc_id']}:{page['page']}"


class Retriever:
    def __init__(self, dense=True):
        self.pages = [json.loads(x) for x in (INDEX / "pages.jsonl").read_text().splitlines()]
        unit_counts = defaultdict(Counter)
        for page in self.pages:
            text = re.sub(r"\s+", " ", page["text"].lower())
            for pattern, unit in [(r"миллиард\w* (?:российск\w* )?руб|млрд\.? руб", "млрд руб."),
                                  (r"миллион\w* (?:российск\w* )?руб|млн\.? руб", "млн руб."),
                                  (r"тысяч\w* белорусск\w* руб", "тыс. белорусских рублей"),
                                  (r"тысяч\w* (?:российск\w* )?руб|тыс\.? руб", "тыс. руб.")]:
                if re.search(pattern, text):
                    unit_counts[page["doc_id"]][unit] += 1
        for page in self.pages:
            counts = unit_counts[page["doc_id"]]
            page["document_units"] = counts.most_common(1)[0][0] if counts else "не определены; см. заголовок таблицы"
            if page["doc_id"].startswith("alfa_"):
                page["number_format"] = "В исходных отчётах Альфа-Банка запятые в суммах отделяют группы тысяч (например 518,983 = 518 983). Проценты и даты проверяй отдельно."
        self.by_key = {page_key(p): p for p in self.pages}
        self.page_index = {page_key(p): i for i, p in enumerate(self.pages)}
        self.tokenized = [tokens(p["text"]) for p in self.pages]
        self.bm25 = BM25Okapi(self.tokenized)
        self.dense = dense
        if dense:
            self.groups = [json.loads(x) for x in (INDEX / "groups.jsonl").read_text().splitlines()]
            self.vectors = np.load(INDEX / "vectors.npy", allow_pickle=False)
            if self.vectors.shape != (len(self.groups), 1024):
                raise ValueError("Индекс и матрица векторов не совпадают")
            self.embedder = Embedder()

    def document_prior(self, question):
        lower = question.lower().replace("ё", "е")
        banks = {bank for bank, pattern in ALIASES.items() if re.search(pattern, lower)}
        years = [int(x) for x in re.findall(r"\b20[12]\d\b", question)]
        report_year = re.search(r"отчетност\w*\s+за\s+(20\d{2})", lower)
        preferred = int(report_year[1]) if report_year else max(years, default=0)
        interim = bool(re.search(r"30\s+июня|шесть\s+месяц|6\s+месяц|полугоди", lower))
        values = []
        for page in self.pages:
            bank, year, period = page["doc_id"].split("_")
            prior = 1.0
            if banks and bank not in banks:
                prior = 0.0
            if preferred:
                distance = int(year) - preferred
                prior *= 1.0 if distance == 0 else .63 if distance == 1 else .48 if int(year) in years else .22
            if interim and period == "annual":
                prior *= .5
            if not interim and period == "6m":
                prior *= .55
            values.append(prior)
        return np.asarray(values)

    def search(self, question, k=18, method="hybrid"):
        if method not in {"hybrid", "vector", "bm25"}:
            raise ValueError("Неизвестный метод поиска")
        if method == "vector" and not self.dense:
            raise ValueError("Для векторного baseline нужен построенный индекс")
        query_tokens = [w for w in tokens(question) if w not in BANK_TERMS and not re.fullmatch(r"20\d{2}|30|31|6", w)]
        lexical = np.maximum(0, self.bm25.get_scores(query_tokens))
        prior = self.document_prior(question)
        lexical *= prior
        dense = np.zeros(len(self.pages))
        if self.dense and method != "bm25":
            # Вопросы намеренно не кэшируются: time включает реальный запрос эмбеддера.
            query_vector = self.embedder.embed(question, use_cache=False)
            similarities = self.vectors @ query_vector
            for group, score in zip(self.groups, similarities):
                for number in group["pages"]:
                    idx = self.page_index[f"{group['doc_id']}:{number}"]
                    dense[idx] = max(dense[idx], float(score))
            dense *= prior
        fused = np.zeros(len(self.pages))
        if method in {"vector", "bm25"}:
            scores = dense if method == "vector" else lexical
            order = np.argsort(-scores, kind="stable")
            return [{**self.pages[i], "score": float(scores[i]), "key": page_key(self.pages[i])}
                    for i in order if prior[i] > 0 and scores[i] > 0][:k]
        for scores, weight in ((lexical, 1.0), (dense, .65)):
            for rank, idx in enumerate(np.argsort(-scores, kind="stable")[:100]):
                if scores[idx] > 0:
                    fused[idx] += weight / (25 + rank)
        # Небольшой бонус за долю найденных смысловых слов запроса.
        terms = {x for x in query_tokens if not x.isdigit() and x not in {"альфа", "втб", "сбербанк"}}
        if terms:
            coverage = np.asarray([len(terms & set(words)) / len(terms) for words in self.tokenized])
            fused *= .6 + .4 * coverage
        order = np.argsort(-fused, kind="stable")
        return [{**self.pages[i], "score": float(fused[i]), "key": page_key(self.pages[i])}
                for i in order if prior[i] > 0 and fused[i] > 0][:k]

    def snippet(self, page, question, limit=1500):
        lines = page["text"].splitlines()
        terms = set(tokens(question))
        ranked = sorted(range(len(lines)), key=lambda i: -len(terms & set(tokens(lines[i]))))
        chosen = set(range(min(5, len(lines))))
        for i in ranked[:8]:
            chosen.update(range(max(0, i-1), min(len(lines), i+2)))
        text = "\n".join(lines[i] for i in sorted(chosen))
        return text[:limit]
