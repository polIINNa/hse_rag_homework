"""Обучение реранкера на размеченных страницах train."""

import hashlib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rag.retrieval import excerpt, report_hint, row_match, row_query

PILOT_SIZE = 20
POOL_SIZE = 16


def features(pages, question, idf):
    weights, pairs = row_query(question, idf)
    primary, years, period = report_hint(question)
    maximum = max(p["score"] for p in pages)
    result = []
    for page in pages:
        text = page["text"]
        lines = text.splitlines()
        previous = ""
        row_score = line_score = row_count = 0
        for line in lines:
            label, separator, _ = line.partition("|")
            line_score = max(line_score, row_match(line, weights, pairs))
            if separator and sum(char.isdigit() for char in line) >= 4:
                row_count += 1
                row_score = max(row_score, row_match(previous + " " + label, weights, pairs))
            previous = label
        source = page["doc_id"].split("_")
        selected = page.get("excerpt") or excerpt(text, question, idf=idf)
        result.append([
            page["score"], page["score"] - maximum, row_score, line_score,
            row_match(selected, weights, pairs), row_match(text, weights, pairs),
            len(lines) / 50, row_count / max(1, len(lines)),
            primary in source, period in source, bool(years.intersection(source)),
            page["page"] <= 10, page["page"] <= 3,
        ])
    return result


def training_fold(question_id):
    return int(hashlib.sha256(question_id.encode()).hexdigest()[:8], 16) % 3 != 0


def fit_ranker(index, items):
    xs, ys = [], []
    for item in items[PILOT_SIZE:]:
        if not training_fold(item["question_id"]):
            continue
        question = item["question"]
        pages = index.search(question, top_k=POOL_SIZE, rerank=False)
        if not pages:
            continue
        relevant = {(doc["doc_id"], p) for doc in item["gold_context"] for p in doc["pages"]}
        xs.extend(features(pages, question, index.term_idf))
        ys.extend((page["doc_id"], page["page"]) in relevant for page in pages)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    model.fit(xs, ys)
    return model, len(ys), int(sum(ys))


def rerank_with_model(model, pages, question, idf):
    probabilities = model.predict_proba(features(pages, question, idf))[:, 1]
    return [page for _, page in sorted(zip(probabilities, pages), key=lambda pair: -pair[0])]
