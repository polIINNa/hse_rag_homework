"""Убрать длинные линии таблиц, сохранив текст и числа, перед OCR."""

import io

import numpy as np
import pymupdf
from PIL import Image


def remove_grid(gray):
    dark = gray < 150
    mask = np.zeros_like(dark)
    axes = [(dark, mask, max(80, gray.shape[1] // 20)),
            (dark.T, mask.T, max(80, gray.shape[0] // 30))]
    for source, target, length in axes:
        for i, row in enumerate(source):
            filled = row.copy()
            filled[1:-1] |= row[:-2] & row[2:]
            edges = np.diff(np.r_[False, filled, False].astype(np.int8))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if end - start >= length:
                    target[i, start:end] = True
    expanded = mask.copy()
    expanded[1:] |= mask[:-1]
    expanded[:-1] |= mask[1:]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    result = gray.copy()
    result[expanded] = 255
    return result


def recognize(page, tessdata):
    pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    image = io.BytesIO()
    Image.fromarray(remove_grid(gray)).save(image, format="PNG")
    with pymupdf.open() as clean:
        raster = clean.new_page(width=page.rect.width, height=page.rect.height)
        raster.insert_image(raster.rect, stream=image.getvalue())
        textpage = raster.get_textpage_ocr(language="rus+eng", dpi=300, full=True, tessdata=str(tessdata))
        return raster.get_text("text", textpage=textpage, sort=True)
