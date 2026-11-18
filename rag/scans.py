"""Повторное OCR табличной области на страницах, распознанных со сканов."""

import re
from collections import defaultdict

import fitz
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening

from rag.retrieval import compact_text


def horizontal_lines(dark):
    width = dark.shape[1]
    kernel = np.ones((1, max(60, width // 10)), dtype=bool)
    return binary_opening(dark, structure=kernel)


def table_crop(page):
    image = page.get_pixmap(dpi=130, colorspace=fitz.csGRAY)
    gray = np.frombuffer(image.samples, dtype=np.uint8).reshape(image.height, image.width)
    horizontal = horizontal_lines(gray < 170)
    rows = np.flatnonzero(horizontal.sum(axis=1) > image.width * 0.25)
    if len(rows) < 5 or np.count_nonzero(np.diff(rows) > 2) < 4:
        return None
    columns = np.flatnonzero(horizontal[rows].sum(axis=0) >= 2)
    if not len(columns):
        return None
    x_scale, y_scale = page.rect.width / image.width, page.rect.height / image.height
    region = fitz.Rect((columns[0] - 6) * x_scale, (rows[0] - 6) * y_scale,
                       (columns[-1] + 6) * x_scale, (rows[-1] + 6) * y_scale)
    return region & page.rect


def read_scan_table(page, tessdata):
    crop = table_crop(page)
    if crop is None or crop.width < 100 or crop.height < 50:
        return ""
    image = page.get_pixmap(dpi=250, clip=crop)
    rgb = np.frombuffer(image.samples, dtype=np.uint8).reshape(image.height, image.width, 3).copy()
    dark = rgb.mean(axis=2) < 170
    vertical_kernel = np.ones((max(55, image.height // 6), 1), dtype=bool)
    rulings = horizontal_lines(dark) | binary_opening(dark, structure=vertical_kernel)
    rgb[binary_dilation(rulings, iterations=1)] = 255
    rgb = np.pad(rgb, ((18, 18), (18, 18), (0, 0)), constant_values=255)
    cleaned = fitz.Pixmap(fitz.csRGB, rgb.shape[1], rgb.shape[0], rgb.tobytes(), False)
    cleaned.set_dpi(250, 250)
    recognized = cleaned.pdfocr_tobytes(compress=False, language="rus+eng", tessdata=str(tessdata))
    with fitz.open(stream=recognized, filetype="pdf") as overlay:
        layer = overlay[0]
        text = layer.get_text(textpage=layer.get_textpage(flags=0), sort=True)
    if len(re.findall(r"\d", text)) < 30:
        return ""
    return compact_text(text)


def add_scan_evidence(index, reports, documents, tessdata):
    scan_pages = defaultdict(list)
    for page in index.pages:
        if page.get("ocr"):
            scan_pages[page["doc_id"]].append(page)
    count = 0
    for doc in documents:
        if doc["doc_id"] not in scan_pages:
            continue
        doc_count = 0
        with fitz.open(reports / doc["path"]) as pdf:
            for page in scan_pages[doc["doc_id"]]:
                text = read_scan_table(pdf[page["page"] - 1], tessdata)
                if text:
                    page["ocr_detail"] = text
                    doc_count += 1
        count += doc_count
        print(f"OCR таблиц {doc['doc_id']}: {doc_count}", flush=True)
    return count
