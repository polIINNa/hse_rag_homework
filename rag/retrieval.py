"""Гибридный поиск по страницам отчётов."""

from collections import defaultdict
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import re
import sqlite3
import time

import numpy as np

from .context import ContextBuilder
from .embeddings import Client, cache_open, cached, chunks, text_hash


BANKS = {
    'alfa': r'альфа',
    'domrf': r'дом[\s.]*рф',
    'gpb': r'газпромбанк',
    'mkb': r'\bмкб\b|московск\w*\s+кредитн',
    'rshb': r'россельхоз',
    'sber': r'сбер',
    'sovkom': r'совком',
    'vtb': r'\bвтб\b',
    'tbank': r'т[\s\-‑–]*банк|тинькофф|\bткс\b|т[\-‑–]технолог',
}
STOPWORDS = set(
    ('а и или в во на по из к с со за от до для при о об у как какие какой какая каковы сколько что это ли '
     'был была были год года году годы согласно отчет отчетность отчетности банк банка банке группе группа '
     'рублей руб млн тыс').split()
)


def banks_in_question(question):
    text = question.lower().replace('ё', 'е')
    return {bank for bank, pattern in BANKS.items() if re.search(pattern, text)}


@lru_cache(maxsize=100000)
def stem(word):
    import snowballstemmer

    if not hasattr(stem, 'engine'):
        stem.engine = snowballstemmer.stemmer('russian')
    return stem.engine.stemWord(word)


def stem_tokens(text):
    return [stem(w) for w in re.findall(r'[a-zа-я0-9]+', text.lower().replace('ё', 'е')) if len(w) > 1]


def chunk_signature(rows):
    return hashlib.sha256(''.join(row['chunk_id'] for row in rows).encode()).hexdigest()


def prepare_lexical(corpus, target):
    rows = chunks(corpus)
    temporary = target.with_suffix('.building.sqlite')
    temporary.touch(exist_ok=False)
    try:
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.executescript('''
                CREATE TABLE items(rowid INTEGER PRIMARY KEY, doc_id TEXT, pdf_page INTEGER);
                CREATE VIRTUAL TABLE terms USING fts5(headings, body);
                CREATE TABLE metadata(signature TEXT);
            ''')
            for rowid, row in enumerate(rows, 1):
                headings = ' '.join(
                    line for line in row['text'].splitlines()
                    if sum(c.isalpha() for c in line) >= 5 and line.isupper()
                )
                connection.execute('INSERT INTO items VALUES (?, ?, ?)', (rowid, row['doc_id'], row['pdf_page']))
                connection.execute('INSERT INTO terms(rowid, headings, body) VALUES (?, ?, ?)',
                                   (rowid, ' '.join(stem_tokens(headings)), ' '.join(stem_tokens(row['text']))))
            connection.execute('INSERT INTO metadata VALUES (?)', (chunk_signature(rows),))
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Полнотекстовый индекс: {len(rows)} чанков', flush=True)


def stem_search(question, index, top_k=30, *, doc_ids=None, exclude=()):
    stops = {stem(word) for word in STOPWORDS}
    words = list(dict.fromkeys(w for w in stem_tokens(question) if w not in stops))[:64]
    if not words or (doc_ids is not None and not doc_ids):
        return []

    conditions = ['terms MATCH ?']
    parameters = [' OR '.join(f'"{word}"' for word in words)]
    if doc_ids is not None:
        placeholders = ', '.join('?' for _ in doc_ids)
        conditions.append(f'i.doc_id IN ({placeholders})')
        parameters.extend(sorted(doc_ids))
    for doc_id, page in exclude:
        conditions.append('NOT (i.doc_id = ? AND i.pdf_page = ?)')
        parameters.extend((doc_id, page))

    query = f"""
        SELECT i.doc_id, i.pdf_page FROM terms
        JOIN items i ON i.rowid = terms.rowid
        WHERE {' AND '.join(conditions)}
        ORDER BY bm25(terms, 4.0, 1.0)
        LIMIT ?
    """
    connection = sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        rows = connection.execute(query, (*parameters, top_k * 40)).fetchall()
        return list(dict.fromkeys(rows))[:top_k]
    finally:
        connection.close()


def word_search(question, index, top_k=30):
    """Обычный BM25 для диагностики гибридной выдачи."""
    words = list(dict.fromkeys(
        word for word in re.findall(r'[a-zа-я0-9]+', question.lower().replace('ё', 'е'))
        if len(word) > 1 and word not in STOPWORDS
    ))[:64]
    if not words:
        return []
    query = ' OR '.join(f'"{word}"' for word in words)
    connection = sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute('''
            SELECT c.*, bm25(chunks_fts) AS rank,
                   snippet(chunks_fts, 0, '[', ']', ' … ', 32) AS excerpt
            FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
        ''', (query, max(200, top_k * 40))).fetchall()
        found = {}
        for row in rows:
            key = row['doc_id'], row['pdf_page']
            if key in found:
                continue
            found[key] = {'doc_id': key[0], 'pdf_page': key[1], 'excerpt': row['excerpt']}
            if len(found) == top_k:
                break
        return list(found.values())
    finally:
        connection.close()


def report_years(question):
    text = question.lower().replace('ё', 'е')
    if len(banks_in_question(text)) > 1:
        return set()
    pattern = r'(?:отчетност[ьи]|отчете)[^.!?;]{0,90}?\bза\b[^.!?;]{0,45}?\b(202[2-5])\b'
    years = set()
    for match in re.finditer(pattern, text):
        years.add(match.group(1))
        extra = re.match(r'\s*(?:и|,)\s*(?:за\s+)?(202[2-5])\b', text[match.end():])
        if extra:
            years.add(extra.group(1))
    return years


def fuse(rankings, banks=None, years=None, weights=None):
    """Объединяет выдачу через RRF, учитывая банк и год отчёта."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scale = len(rankings) / sum(weights)
    values = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, key in enumerate(ranking, 1):
            values[key] += scale * weight / (60 + rank)
    if years:
        for key in values:
            if key[0].split('_')[1] in years:
                values[key] += 0.004
    candidates = (key for key in values if not banks or key[0].split('_')[0] in banks)
    return sorted(candidates, key=lambda key: (-values[key], key))


def context(keys):
    grouped = {}
    for doc_id, page in keys:
        grouped.setdefault(doc_id, []).append(page)
    return [{'doc_id': doc_id, 'pages': pages} for doc_id, pages in grouped.items()]


class Retriever:
    def __init__(self, root, cache_queries=True):
        self.root = root
        self.cache_queries = cache_queries
        self.index = root / 'index/corpus.sqlite'
        self.stem_index = root / 'index/lexical_stems.sqlite'
        if not self.index.exists() or not self.stem_index.exists():
            raise ValueError('Build the text index first: python -m scripts.build_index')
        self.rows = chunks(self.index)
        lexical = sqlite3.connect(self.stem_index.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            signature = lexical.execute('SELECT signature FROM metadata').fetchone()[0]
        finally:
            lexical.close()
        if signature != chunk_signature(self.rows):
            raise ValueError('Rebuild the stale index: python -m scripts.build_index')

        vector_path = root / 'index/embedding_cache.sqlite'
        if not vector_path.exists():
            raise ValueError('Build embeddings first: python -m scripts.build_embeddings')
        hashes = [text_hash(row['text']) for row in self.rows]
        self.connection = cache_open(vector_path)
        try:
            vectors = cached(self.connection, hashes)
            if len(vectors) != len(set(hashes)) or not hashes:
                raise ValueError('Complete the vector index: python -m scripts.build_embeddings')
            self.matrix = np.stack([vectors[key] for key in hashes])
            self.context_builder = ContextBuilder(self.index)
        except BaseException:
            self.connection.close()
            raise
        self.client = None

    def close(self):
        self.context_builder.close()
        self.connection.close()

    def retrieve(self, question, top_k=5):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        ranked, timing = self.rank(question)
        start = time.perf_counter()
        pages = []
        for doc_id, number in ranked[:top_k]:
            page = self.context_builder.source_page(doc_id, number, 'retrieved_page')
            pages.append(page)
        pages, diagnostics = self.context_builder.build(pages)
        timing['context_seconds'] = time.perf_counter() - start
        return {
            'method': 'hybrid_weighted',
            'context': context((page['doc_id'], page['pdf_page']) for page in pages),
            'page_context': pages,
            'timing': timing,
            'context_diagnostics': diagnostics,
        }

    def search_more(self, question, query, pages, doc_id=None):
        """Найти одну ещё не показанную страницу по запросу агента."""
        banks = banks_in_question(question)
        available_docs = {row['doc_id'] for row in self.rows}
        if banks:
            available_docs = {doc for doc in available_docs if doc.split('_')[0] in banks}
        known_docs = {page['doc_id'] for page in pages if page.get('context_role') != 'unit_reference'}
        if doc_id is not None and doc_id in known_docs and doc_id in available_docs:
            available_docs = {doc_id}

        seen = {(page['doc_id'], page['pdf_page']) for page in pages}
        hits = stem_search(query, self.stem_index, top_k=1, doc_ids=available_docs, exclude=seen)
        if not hits:
            return None
        doc, number = hits[0]
        page = self.context_builder.source_page(doc, number, 'followup_page')
        units = next((p['unit_context'] for p in pages if p['doc_id'] == doc and 'unit_context' in p), None)
        if units and (doc, units['source_pdf_page']) in seen:
            page['unit_context'] = units
        return page

    def query_vector(self, question):
        key = text_hash(question)
        if self.cache_queries:
            row = self.connection.execute('SELECT dim, vector FROM vectors WHERE text_sha256 = ?', (key,)).fetchone()
            if row and row[0] == 1024 and len(row[1]) == 4096:
                vector = np.frombuffer(row[1], dtype=np.float32)
                if np.isfinite(vector).all() and abs(float(np.linalg.norm(vector)) - 1) < 1e-3:
                    return vector, False
        if self.client is None:
            self.client = Client(self.root)
            self.client.limits()
        vector = self.client.embed(question)
        if self.cache_queries:
            self.connection.execute('INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)', (key, 1024, vector.tobytes()))
            self.connection.commit()
        return vector, True

    def rank(self, question):
        start = time.perf_counter()
        vector, api_called = self.query_vector(question)
        embedding_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = self.matrix @ vector
        vector_pages = list(dict.fromkeys(
            (self.rows[i]['doc_id'], self.rows[i]['pdf_page'])
            for i in np.argsort(-scores, kind='stable')
        ))
        # В полном прогоне обычная полнотекстовая выдача тоже входила в измеряемый поиск.
        plain_bm25 = word_search(question, self.index)
        lexical = stem_search(question, self.stem_index)
        banks, years = banks_in_question(question), report_years(question)
        ranked = fuse([vector_pages[:100], lexical], banks, years, weights=[1.5, 1.0])
        return ranked, {
            'embedding_api_called': api_called,
            'embedding_seconds': embedding_seconds,
            'ranking_seconds': time.perf_counter() - start,
            'detected_banks': sorted(banks),
            'explicit_report_years': sorted(years),
            'plain_bm25_hits': plain_bm25,
        }
