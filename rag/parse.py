"""Парсинг PDF → data/pages.jsonl.gz (одна строка = одна страница с doc_id и номером страницы).

Из корня репозитория:  python -m rag.parse

- Страницы с текстовым слоем читаем простой библиотекой (pypdfium2) — текст точный, ничего не распознаём.
- Страницы без текстового слоя (сканы: alfa_2024, gpb_2023, gpb_2025, отчёты о финансовом положении
  РСХБ/Совкомбанка/ВТБ, аудиторские заключения) распознаём Tesseract OCR (rus+eng, 300 dpi).
- Очистка: неразрывные пробелы, переносы слов, лишние пробелы и пустые строки.
"""

import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rag.config import DOCUMENTS_PATH, PAGES_PATH, REPORTS_DIR

MIN_LETTERS = 200      # меньше букв в текстовом слое — считаем страницу сканом и отправляем в OCR
OCR_DPI = 300
OCR_CONFIG = "--psm 6"  # psm 6: страница как один блок — строки таблиц остаются строками

BANK_NAMES = {
    "alfa": "ЗАО «Альфа-Банк» (Беларусь)",
    "domrf": "АО «Банк ДОМ.РФ»",
    "gpb": "АО «НПФ Газпромбанк-фонд»",
    "mkb": "ПАО «Московский кредитный банк» (МКБ)",
    "rshb": "АО «Россельхозбанк» (РСХБ)",
    "sber": "ПАО Сбербанк",
    "sovkom": "ПАО «Совкомбанк»",
    "tbank": "АО «ТБанк» (Т-Банк)",
    "vtb": "Банк ВТБ (ПАО)",
}
PERIOD_NAMES = {"annual": "годовая отчётность за {year} год",
                "6m": "промежуточная отчётность за 6 месяцев {year} года (на 30 июня {year})"}


def letters_count(text):
    return len(re.findall(r"[А-Яа-яЁёA-Za-z]", text))


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").replace(" ", " ")
    text = text.replace("￾", "").replace("\x02", "")
    text = re.sub(r"([а-яёa-z])-[ \t]*\n[ \t]*([а-яёa-z])", r"\1\2", text)  # перенос слова
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def doc_meta(doc):
    bank, year, period = doc["doc_id"].split("_")
    return {
        "doc_id": doc["doc_id"],
        "bank": bank,
        "year": int(year),
        "period": period,
        "organization": doc["organization"],
        "title": f"{BANK_NAMES[bank]}: {PERIOD_NAMES[period].format(year=year)}",
    }


def ocr_page(args):
    path, page_index = args
    import pypdfium2 as pdfium
    import pytesseract

    pdf = pdfium.PdfDocument(path)
    image = pdf[page_index].render(scale=OCR_DPI / 72).to_pil()
    return pytesseract.image_to_string(image, lang="rus+eng", config=OCR_CONFIG)


def parse_document(doc, pool):
    import pypdfium2 as pdfium

    path = str(REPORTS_DIR / doc["path"])
    pdf = pdfium.PdfDocument(path)
    texts = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    scans = [i for i, text in enumerate(texts) if letters_count(text) < MIN_LETTERS]
    for i, text in zip(scans, pool.map(ocr_page, [(path, i) for i in scans])):
        if letters_count(text) > letters_count(texts[i]):
            texts[i] = text
    meta = doc_meta(doc)
    return [
        {**meta, "page": i + 1, "source": "ocr" if i in scans else "text", "text": clean_text(text)}
        for i, text in enumerate(texts)
    ]


def main():
    documents = [json.loads(line) for line in DOCUMENTS_PATH.open(encoding="utf-8") if line.strip()]
    missing = [d["path"] for d in documents if not (REPORTS_DIR / d["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Нет PDF в {REPORTS_DIR}: {missing[:3]}… Распакуйте data/reports.zip (см. README).")
    pages = []
    with ProcessPoolExecutor() as pool:
        for doc in documents:
            doc_pages = parse_document(doc, pool)
            pages += doc_pages
            ocr = sum(p["source"] == "ocr" for p in doc_pages)
            print(f"{doc['doc_id']:<20} страниц: {len(doc_pages):>3}  из них OCR: {ocr}", flush=True)
    with gzip.open(PAGES_PATH, "wt", encoding="utf-8") as out:
        for page in pages:
            out.write(json.dumps(page, ensure_ascii=False) + "\n")
    print(f"Готово: {len(pages)} страниц → {PAGES_PATH}")


if __name__ == "__main__":
    main()
