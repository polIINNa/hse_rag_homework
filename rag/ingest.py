"""Extract every PDF page, preserving table alignment and physical page numbers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import pymupdf
from rag.config import DATA, INDEX, TESSDATA

def extract_page_text(page):
    text = page.get_text('text', sort=True)
    letters = len(re.findall(r'[A-Za-zА-Яа-яЁё]', text))
    bad = text.count('\ufffd') > max(10, len(text) * 0.03)
    ocr = (letters < 70 and bool(page.get_images())) or bad
    if ocr and page.rotation:
        # OCR the displayed image: get_textpage_ocr can retain coordinates from
        # before a /Rotate=180 transform and reverse the reading order.
        pixmap = page.get_pixmap(dpi=200, alpha=False)
        data = pixmap.pdfocr_tobytes(language='rus+eng', tessdata=str(TESSDATA))
        with pymupdf.open('pdf', data) as upright:
            text = upright[0].get_text('text', sort=True)
    elif ocr:
        textpage = page.get_textpage_ocr(language='rus+eng', dpi=200, full=True, tessdata=str(TESSDATA))
        text = page.get_text('text', textpage=textpage, sort=True)
    text = text.replace('\x00', '').replace('\u00a0', ' ')
    return '\n'.join(line.rstrip() for line in text.splitlines()).strip(), ocr

def extract_document(document):
    source = DATA / 'reports' / document['path']
    destination = INDEX / 'pages' / (document['doc_id'] + '.jsonl')
    if destination.exists():
        rows = [json.loads(line) for line in destination.read_text(encoding='utf-8').splitlines()]
        if len(rows) == document['pages']:
            return {'doc_id': document['doc_id'], 'pages': len(rows), 'ocr': sum(r['ocr'] for r in rows), 'cached': True}
    rows = []
    with pymupdf.open(source) as pdf:
        if len(pdf) != document['pages']:
            raise ValueError(f"Page count mismatch: {document['doc_id']}")
        for index, page in enumerate(pdf):
            text, ocr = extract_page_text(page)
            rows.append({'doc_id': document['doc_id'], 'page': index + 1,
                         'organization': document['organization'], 'path': document['path'],
                         'text': text, 'ocr': ocr})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    temporary.replace(destination)
    return {'doc_id': document['doc_id'], 'pages': len(rows), 'ocr': sum(r['ocr'] for r in rows), 'cached': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    INDEX.mkdir(parents=True, exist_ok=True)
    documents = [json.loads(line) for line in (DATA / 'documents.jsonl').read_text(encoding='utf-8').splitlines()]
    started = time.perf_counter()
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(extract_document, documents):
            summaries.append(result)
            print(json.dumps(result), flush=True)
    pages = []
    for document in documents:
        pages.extend(json.loads(line) for line in (INDEX / 'pages' / (document['doc_id'] + '.jsonl')).read_text(encoding='utf-8').splitlines())
    serialized = ''.join(json.dumps(page, ensure_ascii=False) + '\n' for page in pages)
    (INDEX / 'pages.jsonl').write_text(serialized, encoding='utf-8')
    report = {'documents': len(documents), 'pages': len(pages), 'ocr_pages': sum(p['ocr'] for p in pages),
              'empty_pages': [{'doc_id': p['doc_id'], 'page': p['page']} for p in pages if len(p['text']) < 50],
              'seconds': time.perf_counter() - started, 'sha256': hashlib.sha256(serialized.encode()).hexdigest()}
    (INDEX / 'ingestion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
