from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
from rank_bm25 import BM25Okapi
import snowballstemmer

from rag.corpus import Chunk, Document
from rag.models import Embedder


ALIASES = {
    "alfa": r"альфа[\s\-‑–]*банк\w*|alfa[ -]?bank",
    "domrf": r"дом\s*\.?\s*рф|dom[ .]?rf",
    "gpb": r"газпромбанк\w*(?:[\s\-‑–]*фонд\w*)?|гпб",
    "mkb": r"мкб|московск\w+\s+кредитн\w+\s+банк\w*",
    "rshb": r"рсхб|россельхозбанк\w*",
    "sber": r"сбер(?:банк)?(?:а|у|ом|е)?|sberbank",
    "sovkom": r"совкомбанк\w*",
    "tbank": r"т[\s\-‑–]*банк\w*|тинькофф|тинькоф|t[ -]?bank|tinkoff",
    "vtb": r"втб|vtb",
}
BANK_NAMES = re.compile(r"\b(?:" + "|".join(ALIASES.values()) + r")\b", re.I)
REPORT_DATES = re.compile(r"\b20\d{2}\b|\b(?:30\s+июня|31\s+декабря|1\s+января)\b", re.I)
STEMMER = snowballstemmer.stemmer("russian")
STOP_WORDS = frozenset(
    "а в во для за и из или к на о об от по под при с со у через что это как который "
    "какой был были год года году годы лет какие каков каковы сколько согласно между".split()
)


@lru_cache(maxsize=65536)
def stem(word):
    return STEMMER.stemWord(word)


REPORT_STEMS = frozenset(
    stem(word.replace("ё", "е"))
    for word in (
        "ао оао зао пао пao группа банк отчет отчётность финансовый консолидировано "
        "консолидированный обобщенный информация данным разъяснение примечание "
        "закончившийся конец состояние дата россия российский сбербанк втб рсхб "
        "россельхозбанк дом рф мкб совкомбанк"
    ).split()
)


def terms(text):
    text = REPORT_DATES.sub(" ", BANK_NAMES.sub(" ", text))
    words = re.findall(r"[^\W\d_]+|\d+", text.casefold().replace("ё", "е"))
    stems = [stem(word) for word in words if word not in STOP_WORDS and len(word) > 1]
    tokens = [word for word in stems if word not in REPORT_STEMS]
    return tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]


def organizations(question):
    return tuple(bank for bank, aliases in ALIASES.items() if re.search(rf"\b(?:{aliases})\b", question, re.I))


@dataclass(frozen=True, slots=True)
class Evidence:
    doc_id: str
    page: int
    label: str
    text: str

    @property
    def source(self):
        return self.doc_id, self.page


class ReportIndex:
    def __init__(self, documents, chunks, vectors, embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.labels = {doc.doc_id: doc.label for doc in documents}
        self.lexical = BM25Okapi([terms(chunk.text) for chunk in chunks])
        self.all_rows = np.arange(len(chunks))
        self.first_row_by_page = {}
        rows_by_bank = defaultdict(list)
        for row, chunk in enumerate(chunks):
            self.first_row_by_page.setdefault(chunk.source, row)
            rows_by_bank[chunk.doc_id.split("_", 1)[0]].append(row)
        self.rows_by_bank = {bank: np.array(rows) for bank, rows in rows_by_bank.items()}

    @classmethod
    def load(cls, path, embedder=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Index missing at {path}. Run `python -m rag build` first.")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            vectors = archive["vectors"]
        chunks = [Chunk(**chunk) for chunk in metadata["chunks"]]
        if metadata["version"] != 1 or vectors.ndim != 2 or len(vectors) != len(chunks) or not chunks:
            raise ValueError(f"Invalid index at {path}. Rebuild it with `python -m rag build`.")
        documents = [Document(**doc) for doc in metadata["documents"]]
        encoder = embedder or Embedder(metadata["embedding_model"], metadata["embedding_revision"])
        return cls(documents, chunks, vectors, encoder)

    def _rank_pages(self, eligible, scores, positive_only=False):
        ordered = eligible[np.argsort(-scores[eligible], kind="stable")]
        ranked = {}
        for row in ordered:
            if positive_only and scores[row] <= 0:
                break
            ranked.setdefault(self.chunks[row].source, row)
            if len(ranked) == 48:
                break
        return ranked

    def _fuse(self, rows, dense, lexical, hint, excluded):
        scores = defaultdict(float)
        matches = defaultdict(list)
        report_years = set(re.findall(r"\b20\d{2}\b", hint))
        interim = bool(re.search(r"(?:6|шесть|шести)\s+месяц|полугод|30\s+июня", hint, re.I))
        annual = bool(re.search(r"годов\w+\s+(?:финансов\w+\s+)?отч|31\s+декабря", hint, re.I))
        for weight, ranks in (
            (1.0, self._rank_pages(rows, dense)),
            (0.85, self._rank_pages(rows, lexical, positive_only=True)),
        ):
            for rank, (source, row) in enumerate(ranks.items(), 1):
                if source in excluded:
                    continue
                scores[source] += weight / (rank + 18)
                if row not in matches[source]:
                    matches[source].append(row)
        for source in scores:
            _, year, period = source[0].split("_", 2)
            scores[source] += 0.007 * (year in report_years)
            scores[source] += 0.005 * ((interim and period == "6m") or (annual and period == "annual"))
        return [matches[source] for source in sorted(scores, key=scores.get, reverse=True)]

    def search(self, query, limit, *, original_question=None, excluded=()):
        hint = original_question or query
        banks = [bank for bank in organizations(query) or organizations(hint) if bank in self.rows_by_bank]
        groups = [self.rows_by_bank[bank] for bank in banks] or [self.all_rows]
        dense_scores = self.vectors @ self.embedder.encode([query], query=True)[0]
        lexical_scores = self.lexical.get_scores(list(dict.fromkeys(terms(query))))
        ranked_groups = [self._fuse(rows, dense_scores, lexical_scores, hint, excluded) for rows in groups]
        selected = [rows for cycle in zip_longest(*ranked_groups) for rows in cycle if rows is not None][:limit]
        evidence = []
        for rows in selected:
            best = self.chunks[rows[0]]
            fragments = [self.chunks[row].text for row in sorted(rows)]
            first_row = self.first_row_by_page[best.source]
            if first_row not in rows:
                lead = "\n".join(self.chunks[first_row].text.splitlines()[:10])[:480]
                fragments.insert(0, lead)
            evidence.append(Evidence(best.doc_id, best.page, self.labels[best.doc_id], "\n\n".join(fragments)))
        return evidence


def save_index(path, documents, chunks, model_name, embedder, batch_size):
    if not chunks:
        raise ValueError("No usable text was extracted from the PDF reports")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = {doc.doc_id: doc.label for doc in documents}
    texts = [f"{labels[chunk.doc_id]}\n{chunk.text}" for chunk in chunks]
    vectors = embedder.encode(texts, batch_size=batch_size)
    metadata = json.dumps({
        "version": 1, "embedding_model": model_name, "embedding_revision": embedder.revision,
        "documents": [asdict(doc) for doc in documents], "chunks": [asdict(chunk) for chunk in chunks],
    }, ensure_ascii=False)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as target:
        temporary_path = Path(target.name)
    try:
        np.savez_compressed(temporary_path, vectors=vectors, metadata=metadata)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
