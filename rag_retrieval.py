import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from threading import local

import numpy as np
import snowballstemmer
from fastembed import TextEmbedding
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rag_corpus import DATA, INDEX, load_pages, passages, read_jsonl


MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REPO = "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
MODEL_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
MODEL_FILES = [
    "config.json", "model_optimized.onnx", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json",
]
ALIASES = {
    "alfa": r"альфа[\s‑–—-]*банк",
    "domrf": r"дом\s*\.?\s*рф",
    "gpb": r"газпромбанк|(?<!\w)нпф(?!\w)",
    "mkb": r"(?<!\w)мкб(?!\w)|московск\w*\s+кредитн\w*\s+банк",
    "rshb": r"россельхозбанк|(?<!\w)рсхб(?!\w)",
    "sber": r"сбербанк|(?<!\w)сбер\b",
    "sovkom": r"совкомбанк",
    "tbank": r"тинькофф|(?<!\w)[тt][\s‑–—-]*банк",
    "vtb": r"(?<!\w)втб(?!\w)",
}
YEAR = r"\b20[12]\d\b"
INTERIM = r"(?:30\s+июня|перв\w*\s+полугод|1[ -]е\s+полугод|6\s+месяц|шест[ьи]\s+месяц|промежуточн)"
ANNUAL = r"(?:31\s+декабря|годов\w*\s+(?:финансов\w*\s+)?отч[её]т)"
REPORT_YEAR = rf"отч[её]т(?:ност)?\w*[^.?!;]{{0,32}}?\bза\s+(?:[^.?!;\d]{{0,35}})?({YEAR})"
WORD = re.compile(r"[а-яa-z]+|\d+[.,]\d+|\d+")
STEMMERS = local()


@lru_cache(maxsize=40000)
def stem(token):
    if token.isascii():
        return token
    if not hasattr(STEMMERS, "russian"):
        STEMMERS.russian = snowballstemmer.stemmer("russian")
    return STEMMERS.russian.stemWord(token)


STOP_WORDS = set(map(stem, """
в во из с со и или а о об от до для на у по за к ко при как по
насколько какой каков какова каковы каково какую сколько какая кто
согласно составлять составил составила составили составляет быть был
группа год день месяц декабрь июнь рубль российский банк отчет
отчетность консолидированный промежуточный обобщенный финансовый
пао ао мсфо данный состояние итоговый показатель общий отчетный
сбербанк втб мкб тинькофф тбанк совкомбанк россельхозбанк домрф
газпромбанк нпф банк альфа рсхб сбер рф
""".split()))


def terms(text):
    words = [stem(word) for word in WORD.findall(text.lower().replace("ё", "е"))]
    words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


def report_weights(question, reports):
    question = question.casefold()
    banks = {bank for bank, alias in ALIASES.items() if re.search(alias, question)}
    explicit = set(re.findall(REPORT_YEAR, question))
    years = explicit or set(re.findall(YEAR, question))
    interim = bool(re.search(INTERIM, question))
    annual = bool(re.search(ANNUAL, question))
    weights = {}
    for doc_id in reports:
        bank, year, period = doc_id.split("_", 2)
        if banks and bank not in banks:
            continue
        weight = 0.0
        if years:
            if year in years:
                weight += 0.22
            elif not explicit and str(int(year) - 1) in years:
                weight += 0.09
            else:
                weight -= 0.19 if not explicit else 0.32
        if interim != annual:
            weight += 0.13 if (period == "6m") == interim else -0.09
        elif not interim and years:
            weight += 0.09 if period == "annual" else 0.0
        weights[doc_id] = weight
    return weights


def load_encoder():
    options = {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "cache_dir": INDEX / "models",
        "allow_patterns": MODEL_FILES,
    }
    try:
        path = snapshot_download(**options, local_files_only=True)
    except LocalEntryNotFoundError:
        path = None
    if path is None or not all((Path(path) / name).is_file() for name in MODEL_FILES):
        path = snapshot_download(**options)
    return TextEmbedding(MODEL, specific_model_path=path, threads=4)


class Retrieval:
    def __init__(self):
        self.reports = {
            item["doc_id"]: item for item in read_jsonl(DATA / "documents.jsonl")
        }
        self.pages = load_pages()
        self.chunks = []
        chunk_pages = []
        for index, page in enumerate(self.pages):
            for chunk in passages(page["text"], target=1850):
                self.chunks.append(chunk)
                chunk_pages.append(index)
        self.chunk_pages = np.array(chunk_pages)
        self.lexical = TfidfVectorizer(analyzer=terms, sublinear_tf=True, dtype=np.float32)
        self.sparse = self.lexical.fit_transform(self.chunks)
        self.encoder = load_encoder()
        self.dense = self.load_embeddings()

    def load_embeddings(self):
        path = INDEX / "minilm-v1.npy"
        if not path.exists():
            vectors = []
            for index, vector in enumerate(self.encoder.embed(self.chunks, batch_size=24), 1):
                vectors.append(vector)
                if index % 1000 == 0:
                    print(f"Embedded {index}/{len(self.chunks)} passages", flush=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as output:
                np.save(output, normalize(np.asarray(vectors, dtype=np.float32)), allow_pickle=False)
            tmp.replace(path)
        vectors = np.load(path, allow_pickle=False)
        if len(vectors) != len(self.chunks):
            raise ValueError(f"Passage count changed. Rebuild {path}.")
        return vectors

    def score_pages(self, weights, queries):
        query_sparse = self.lexical.transform(queries)
        query_dense = normalize(np.asarray(list(self.encoder.embed(queries, batch_size=len(queries))), dtype=np.float32))
        lexical = (self.sparse @ query_sparse.T).toarray()
        semantic = np.einsum("ij,kj->ik", self.dense, query_dense)
        prior = np.array([weights.get(page["doc_id"], -np.inf) for page in self.pages])
        allowed = np.isfinite(prior[self.chunk_pages])
        lexical /= np.maximum(lexical[allowed].max(axis=0), 0.1)
        semantic_max = semantic[allowed].max(axis=0)
        semantic = np.maximum(0, (semantic - 0.2) / np.maximum(semantic_max - 0.2, 0.1))
        scores = 0.70 * lexical + 0.30 * semantic
        page_scores = np.full((len(self.pages), len(queries)), -np.inf)
        np.maximum.at(page_scores, self.chunk_pages, scores)
        return page_scores + prior[:, None]

    def search(self, question, queries=(), documents=(), limit=12, exclude=()):
        weights = report_weights(question, self.reports)
        if documents:
            weights = {key: max(weights.get(key, 0), 0) for key in documents if key in self.reports}
            if not weights:
                weights = report_weights(question, self.reports)
        queries = list(dict.fromkeys([question, *queries]))
        scores = self.score_pages(weights, queries)
        ranked = np.argsort(-scores, axis=0, kind="stable")
        excluded = set(exclude)
        chosen = {}

        def add(index):
            page = self.pages[index]
            key = page["doc_id"], page["page"]
            if key in excluded or key in chosen or not np.isfinite(scores[index].max()):
                return False
            chosen[key] = page
            return True

        groups = defaultdict(list)
        for index in ranked[:, 0]:
            if np.isfinite(scores[index, 0]):
                groups[self.pages[index]["doc_id"].split("_", 1)[0]].append(index)
        if 1 < len(groups) <= 3:
            for group in groups.values():
                count = 0
                for index in group:
                    count += add(index)
                    if count >= 3 or len(chosen) >= limit:
                        break
        for row in ranked[:limit + len(excluded)]:
            for index in row:
                if len(chosen) < limit:
                    add(index)
        return list(chosen.values())


if __name__ == "__main__":
    index = Retrieval()
    print(f"Indexed {len(index.pages)} pages and {len(index.chunks)} passages.")
