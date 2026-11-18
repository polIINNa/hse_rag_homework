"""Текст всех страниц отчётов в outputs/pages.jsonl.

Из корня репозитория: python -m solution.extract

Текст берётся из текстового слоя PDF. Страницы без него (сканы) распознаются Tesseract
с русским языком.
"""

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import pymupdf
import pytesseract

from solution.documents import OUTPUTS_DIR, load_documents

PAGES_PATH = OUTPUTS_DIR / "pages.jsonl"
MIN_LETTERS = 50  # меньше букв на странице — считаем, что текстового слоя нет (скан)
ROW_TOLERANCE = 3  # pt: слова с близким центром по вертикали стоят в одной строке
CELL_GAP = 15  # pt: промежуток шире — граница ячеек таблицы


def letters_count(text):
    return len(re.findall(r"[А-Яа-яЁёA-Za-z]", text))


def page_text(page):
    """Текст по строкам страницы: строка таблицы собирается целиком, ячейки разделены тремя пробелами."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows = []  # (центр первого слова строки, слова строки)
    for word in words:
        center = (word[1] + word[3]) / 2
        if rows and abs(center - rows[-1][0]) <= ROW_TOLERANCE:
            rows[-1][1].append(word)
        else:
            rows.append((center, [word]))

    lines = []
    for _, row in rows:
        row.sort(key=lambda w: w[0])
        line = row[0][4]
        for previous, word in zip(row, row[1:]):
            line += ("   " if word[0] - previous[2] > CELL_GAP else " ") + word[4]

        lines.append(line)

    return "\n".join(lines)


def ocr_page(page, dpi=300, lang="rus+eng"):
    """Фотографируем страницу и распознаём картинку Tesseract."""
    return pytesseract.image_to_string(page.get_pixmap(dpi=dpi).pil_image(), lang=lang)


def extract_pages(document):
    pages = []
    with pymupdf.open(document["path"]) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            text = page_text(page)
            source = "text"
            if letters_count(text) < MIN_LETTERS:
                text = ocr_page(page)
                source = "ocr"

            pages.append({"doc_id": document["doc_id"], "page": page_no, "source": source, "text": text})
            
    return pages


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    # Документы разбираются в нескольких процессах, поэтому каждому tesseract хватит одного потока.
    os.environ["OMP_THREAD_LIMIT"] = "1"

    documents = load_documents()
    total = 0
    with ProcessPoolExecutor() as pool, open(PAGES_PATH, "w", encoding="utf-8") as output:
        # map возвращает результаты в порядке documents, так что файл не зависит от того, какой процесс закончил раньше
        for document, pages in zip(documents, pool.map(extract_pages, documents)):
            for page in pages:
                output.write(json.dumps(page, ensure_ascii=False) + "\n")

            ocr = sum(page["source"] == "ocr" for page in pages)
            print(f"{document['doc_id']}: {len(pages)} стр., из них OCR {ocr}", flush=True)
            total += len(pages)
    print(f"Готово: {total} стр. в {PAGES_PATH}")


if __name__ == "__main__":
    main()
