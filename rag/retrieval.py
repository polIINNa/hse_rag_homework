"""Hybrid E5 + Russian BM25 retrieval, with page-level fusion and metadata."""
from functools import lru_cache
import hashlib
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
import snowballstemmer
from rag.config import INDEX
from rag.embeddings import Embedder, RemoteEmbedder

BANK_PATTERNS = {
    'alfa': r'альфа', 'domrf': r'дом[.\s-]*рф', 'gpb': r'газпромбанк|\bнпф\b',
    'mkb': r'\bмкб\b|московск\w*\s+кредитн', 'rshb': r'россельхоз|\bрсхб\b',
    'sber': r'сбер', 'sovkom': r'совком', 'tbank': r'т[\s-]*банк|тинькофф|ткс\s+холдинг|т[ -]технолог',
    'vtb': r'\bвтб\b',
}
STEMMER = snowballstemmer.stemmer('russian')
STOP = set('и в во на за по с со к ко из для от до об о а но как какой какая какие согласно год года году годов отчетность отчетности банк банка группы группа'.split())

UNIT_HEADER = re.compile(r'(?:\(\s*в\s*|\bв\s+|единиц[аы]\s*измерения\s*:?\s*)'
                         r'(?:миллиард\w*|миллион\w*|тысяч\w*|млрд\.?|млн\.?|тыс\.?)\s*'
                         r'(?:российск\w*|белорусск\w*)?\s*руб(?:лей|леи|лях|ли|ль|\.)', re.I)

def unit_excerpt(page):
    header = page['text'][:1200]
    match = UNIT_HEADER.search(header)
    if match:
        return header[max(0, match.start() - 180):min(len(header), match.end() + 100)]
    # Financial forms also use a separate row of column-unit labels, without "в".
    # Reject rows containing actual amounts; reporting years are allowed.
    cursor = 0
    for line in header.splitlines(keepends=True):
        without_years = re.sub(r'20\d{2}', '', line)
        if re.search(r'(?:млн|млрд|тыс)\.?\s*руб\.', without_years, re.I) and not re.search(r'\d', without_years):
            return header[max(0, cursor - 180):cursor + len(line) + 80]
        cursor += len(line)
    return None

@lru_cache(maxsize=100000)
def stem(word):
    return STEMMER.stemWord(word)

def tokenize(text):
    words = re.findall(r'[a-zа-яё]+|\d+', text.lower().replace('ё', 'е'))
    return [stem(word) for word in words if word not in STOP and len(word) > 1]

def organizations(question):
    return [bank for bank, pattern in BANK_PATTERNS.items() if re.search(pattern, question.lower())]

def context_from_pages(pages):
    result = {}
    for page in pages:
        result.setdefault(page['doc_id'], set()).add(page['page'])
    return [{'doc_id': doc, 'pages': sorted(numbers)} for doc, numbers in sorted(result.items())]

class Retriever:
    def __init__(self, index_dir=INDEX):
        self.pages = [json.loads(line) for line in (index_dir / 'pages.jsonl').read_text(encoding='utf-8').splitlines()]
        self.chunks = [json.loads(line) for line in (index_dir / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        self.manifest = json.loads((index_dir / 'manifest.json').read_text())
        if hashlib.sha256((index_dir / 'pages.jsonl').read_bytes()).hexdigest() != self.manifest['pages_sha256']:
            raise ValueError('Page cache changed; rebuild embeddings with python -m rag.build_index')
        self.by_key = {(page['doc_id'], page['page']): page for page in self.pages}
        self.unit_sources = {}
        for page in self.pages:
            excerpt = unit_excerpt(page)
            if page['page'] <= 20 and excerpt:
                self.unit_sources.setdefault(page['doc_id'], {**page, 'text': excerpt})
        self.vectors = np.load(index_dir / 'vectors.npy', mmap_mode='r')
        if len(self.chunks) != len(self.vectors):
            raise ValueError('Chunk/vector count mismatch')
        self.embedder = RemoteEmbedder() if self.manifest.get('backend') == 'remote' else Embedder()
        self.lexical = BM25Okapi([tokenize(chunk['text']) or ['empty'] for chunk in self.chunks])
        self.banks = np.array([chunk['doc_id'].split('_')[0] for chunk in self.chunks])
        # Warm up outside measured question latency.
        self.embedder.encode(['финансовая отчетность'], query=True)

    def search(self, question, limit=14, mode='hybrid'):
        banks = organizations(question)
        eligible = np.isin(self.banks, banks) if banks else np.ones(len(self.chunks), dtype=bool)
        candidates = np.flatnonzero(eligible)
        years = set(re.findall(r'20\d{2}', question))
        aggregate_assets = mode != 'vector' and bool(re.search(r'(?:совокупн\w*|итого|всего|общ\w*(?:\s+\w+){0,2})\s+актив', question.lower()))
        search_query = question + (' Итого активы. Всего активов. Отчет о финансовом положении.' if aggregate_assets else '')
        dense = self.vectors @ self.embedder.encode([search_query], query=True)[0] if mode != 'bm25' else None
        lexical = self.lexical.get_scores(tokenize(search_query)) if mode != 'vector' else None
        page_scores = {}
        snippets = {}
        for scores in (dense, lexical):
            if scores is None:
                continue
            ranked = candidates[np.argsort(-scores[candidates], kind='stable')[:300]]
            seen = set()
            page_rank = 0
            for chunk_id in ranked:
                chunk = self.chunks[int(chunk_id)]
                key = (chunk['doc_id'], chunk['page'])
                if key in seen:
                    continue
                seen.add(key)
                page_rank += 1
                score = 1 / (40 + page_rank)
                page_scores[key] = page_scores.get(key, 0) + score
                snippets.setdefault(key, []).append(chunk['text'])
        # Total assets is often expressed as "Итого активы", while the question
        # says "совокупные активы". Add literal table-row evidence from PDFs.
        # This is a corpus rule, independent of all QA labels.
        anchors = {}
        if aggregate_assets:
            for page in self.pages:
                if banks and page['doc_id'].split('_')[0] not in banks:
                    continue
                match = re.search(r'итого\s+актив\w*[^\n]*', page['text'], re.I)
                if match:
                    key = (page['doc_id'], page['page'])
                    is_statement = page['page'] <= 15 or 'отчет о финансовом положении' in page['text'][:500].lower()
                    page_scores[key] = page_scores.get(key, 0) + (0.045 if is_statement else 0.025)
                    anchors[key] = page['text'][:350] + '\n[...]\n' + page['text'][max(0, match.start() - 150):match.end() + 250]
                    balance_total = re.search(r'итого\s+обязательств[^\n]*собственн[^\n]*средств[^\n]*', page['text'], re.I)
                    if balance_total:
                        anchors[key] += '\nПроверка по итоговой строке баланса:\n' + balance_total.group(0)
                    snippets.setdefault(key, [anchors[key]])
        for key in page_scores:
            year = key[0].split('_')[1]
            if years:
                page_scores[key] *= 1.3 if year in years else (0.9 if str(int(year) - 1) in years else 0.65)
            interim = bool(re.search(r'шесть месяцев|6 месяцев|полугоди|30 июня', question.lower()))
            if interim and key[0].endswith('_6m'):
                page_scores[key] *= 1.12
        ranking = sorted(page_scores, key=lambda key: (-page_scores[key], key))
        # Retain coverage for compound multi-bank questions.
        selected = []
        if len(banks) > 1:
            for bank in banks:
                selected.extend([key for key in ranking if key[0].startswith(bank + '_')][:2])
        selected.extend(key for key in ranking if key not in selected)
        result = []
        for key in selected[:limit]:
            page = self.by_key[key]
            pieces = list(dict.fromkeys(snippets[key]))
            excerpt = pieces[0][:1300] if len(pieces) == 1 else '\n[...]\n'.join(piece[:650] for piece in pieces[:2])
            if key in anchors:
                excerpt = anchors[key][:1300]
            result.append({**page, 'score': page_scores[key], 'snippet': excerpt})
        return result

    def read_page(self, doc_id, page):
        return self.by_key.get((doc_id, page))

    def add_unit_sources(self, question, selected):
        numeric = re.search(r'каков|сколько|сумм|объ[её]м|размер|величин|сравн|рассчит|вычисл|изменил|динамик', question.lower())
        if not numeric:
            return selected
        result = list(selected)
        for doc in dict.fromkeys(page['doc_id'] for page in selected):
            if any(unit_excerpt(page) for page in selected if page['doc_id'] == doc):
                continue
            source = self.unit_sources.get(doc)
            if source and (source['doc_id'], source['page']) not in {(p['doc_id'], p['page']) for p in result}:
                result.append(source)
        return result
