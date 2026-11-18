"""Построить корпус чанков и полнотекстовый индекс SQLite."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from rag.retrieval import prepare_lexical

BODY_CHARS = 6000
OVERLAP = 350


def windows(text, size=BODY_CHARS, overlap=OVERLAP):
    """Делит исходный текст на фрагменты с ограниченным перекрытием и сохраняет все непробельные символы."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError('Require size > overlap >= 0')
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start + size // 2, end)
            if boundary < 0:
                boundary = text.rfind(' ', start + size // 2, end)
            if boundary >= 0:
                end = boundary + 1
        yield start, end, text[start:end]
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        boundary = text.find('\n', next_start, end)
        if boundary >= 0:
            next_start = boundary + 1
        start = min(next_start, end)


def page_context(page):
    lines = [x.strip() for x in page['text'].splitlines() if x.strip()]
    heading = []
    for line in lines[:12]:
        if len('\n'.join(heading + [line])) > 1000:
            break
        heading.append(line)
    units = [x for x in lines if re.search(r'(?:в |\()(?:тыс|миллион|миллиард|млн|млрд|тысяч)', x, re.I)]
    return (f"Организация: {page['organization']}\nДокумент: {page['doc_id']}\n"
            f"PDF-страница: {page['pdf_page']}\n"
            + ('Единицы из текста страницы: ' + ' | '.join(dict.fromkeys(units)) + '\n' if units else '')
            + 'Начало страницы для контекста:\n' + '\n'.join(heading) + '\n')


def table_blocks(table):
    cells = table['cells']
    headers = {}
    for column in range(table['cols']):
        path = [c['text'].strip() for c in sorted(cells, key=lambda c: c['start_row_offset_idx'])
                if c['column_header'] and c['start_col_offset_idx'] <= column < c['end_col_offset_idx'] and c['text'].strip()]
        headers[column] = ' / '.join(dict.fromkeys(path)) or f'Колонка {column+1}'
    previous_labels = []
    for row in range(table['rows']):
        values = sorted([c for c in cells if c['start_row_offset_idx'] == row and not c['column_header'] and c['text'].strip()],
                        key=lambda c: c['start_col_offset_idx'])
        if not values:
            continue
        label = next((c['text'].strip() for c in values if c['start_col_offset_idx'] == 0), '')
        result = [f'Строка {row+1}: {label or "название не выделено"}']
        if previous_labels:
            result.append('Предшествующие названия строк (контекст): ' + ' | '.join(previous_labels[-2:]))
        for cell in values:
            if cell['start_col_offset_idx'] == 0 and label:
                continue
            columns = list(range(cell['start_col_offset_idx'], cell['end_col_offset_idx']))
            header = '; '.join(headers[c] for c in columns)
            span_note = ' [объединённая ячейка]' if cell['col_span'] > 1 or cell['row_span'] > 1 else ''
            result.append(f"{header}{span_note}: {cell['text'].strip()}")
        if label:
            previous_labels.append(label)
        yield '\n'.join(result)


def chunks_for_page(page, body_chars=BODY_CHARS, overlap_chars=OVERLAP):
    if not page['text_available']:
        return
    prefix = page_context(page)
    for start, end, body in windows(page['text'],body_chars,overlap_chars):
        yield {'kind': 'text', 'text': prefix + '\nФрагмент исходного текста:\n' + body,
               'source_span': [start, end], 'flags': page['quality_flags']}
    tables = page.get('tables', {})
    if tables.get('status') != 'complete':
        return
    for number, table in enumerate(tables['items']):
        blocks = list(table_blocks(table))
        if not blocks:
            continue
        # В каждом фрагменте повторяем контекст страницы и явно указываем период для каждой строки.
        batch = []
        for block in blocks:
            if batch and len('\n\n'.join(batch + [block])) > body_chars:
                yield {'kind': 'table', 'text': prefix + '\nСтруктура Docling, требует проверки.\n' + '\n\n'.join(batch),
                       'table_index': number, 'flags': list(dict.fromkeys(page['quality_flags'] + ['table_structure_unverified']))}
                batch = []
            batch.append(block)
        if batch:
            yield {'kind': 'table', 'text': prefix + '\nСтруктура Docling, требует проверки.\n' + '\n\n'.join(batch),
                   'table_index': number, 'flags': list(dict.fromkeys(page['quality_flags'] + ['table_structure_unverified']))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path('outputs/corpus/pages.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('index'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'corpus.sqlite'
    temporary = args.output / 'corpus.building.sqlite'
    try:
        temporary.touch(exist_ok=False)
    except FileExistsError:
        raise SystemExit(f'Build file already exists: {temporary}. Check that no build is running before removing it.')
    connection = sqlite3.connect(temporary)
    counts = Counter()
    try:
        connection.executescript('''
            PRAGMA foreign_keys=ON;
            CREATE TABLE pages(doc_id TEXT, pdf_page INTEGER, organization TEXT, source TEXT,
                text TEXT, quality_flags TEXT, source_sha256 TEXT, PRIMARY KEY(doc_id,pdf_page));
            CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT,
                pdf_page INTEGER, kind TEXT, text TEXT, search_text TEXT, metadata TEXT,
                FOREIGN KEY(doc_id,pdf_page) REFERENCES pages(doc_id,pdf_page));
            CREATE VIRTUAL TABLE chunks_fts USING fts5(search_text, content='chunks', content_rowid='rowid', tokenize='unicode61');
            CREATE TABLE build_metadata(key TEXT PRIMARY KEY, value TEXT);
        ''')
        source_hash = hashlib.sha256()
        with args.corpus.open('rb') as source:
            for line in source:
                source_hash.update(line)
                page = json.loads(line)
                connection.execute('INSERT INTO pages VALUES (?,?,?,?,?,?,?)',
                    (page['doc_id'], page['pdf_page'], page['organization'], page['source'], page['text'],
                     json.dumps(page['quality_flags'], ensure_ascii=False), page['source_sha256']))
                counts['pages'] += 1
                page_chunks = list(chunks_for_page(page))
                if not page_chunks:
                    counts['pages_without_chunks'] += 1
                for index, chunk in enumerate(page_chunks):
                    identifier = hashlib.sha256((f"{page['doc_id']}:{page['pdf_page']}:{chunk['kind']}:{index}:" + chunk['text']).encode()).hexdigest()[:24]
                    metadata = {k:v for k,v in chunk.items() if k not in ['text','kind']}
                    search_text = chunk['text'].lower().replace('ё', 'е')
                    cursor = connection.execute(
                        'INSERT INTO chunks(chunk_id,doc_id,pdf_page,kind,text,search_text,metadata) VALUES (?,?,?,?,?,?,?)',
                        (identifier, page['doc_id'], page['pdf_page'], chunk['kind'], chunk['text'],
                         search_text, json.dumps(metadata, ensure_ascii=False)),
                    )
                    connection.execute('INSERT INTO chunks_fts(rowid,search_text) VALUES (?,?)',
                                       (cursor.lastrowid, search_text))
                    counts['chunks'] += 1
                    counts[chunk['kind']+'_chunks'] += 1
                if counts['pages'] % 500 == 0:
                    print(f"Pages: {counts['pages']}; chunks: {counts['chunks']}", flush=True)
        metadata = {'format_version':1,'body_chars':BODY_CHARS,'overlap_chars':OVERLAP,'corpus_sha256':source_hash.hexdigest(),
                    'counts':dict(counts),'search':'SQLite FTS5/BM25 over stemmed chunks'}
        connection.execute('INSERT INTO build_metadata VALUES (?,?)', ('config',json.dumps(metadata,ensure_ascii=False)))
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
        connection.execute("INSERT INTO chunks_fts(chunks_fts,rank) VALUES('integrity-check',1)")
        connection.commit()
        assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        connection.close()
        temporary.replace(target)
        (args.output/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        print(json.dumps(metadata,ensure_ascii=False),flush=True)
    except BaseException:
        connection.close()
        # При ошибке удаляем временную базу и сохраняем предыдущий рабочий индекс.
        temporary.unlink(missing_ok=True)
        raise


    prepare_lexical(target, args.output / 'lexical_stems.sqlite')


if __name__ == '__main__':
    main()
