import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess
import zipfile

import fitz


ROOT = Path(__file__).resolve().parent


def parse_report(item):
    document, content = item
    pdf = fitz.open(stream=content, filetype="pdf")
    layout = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", "-", "-"],
        input=content, capture_output=True, check=True,
    ).stdout.decode("utf-8").split("\f")
    pages = []
    for number, page in enumerate(pdf, start=1):
        text = layout[number - 1].strip()
        if len(text) < 100:
            pixmap = page.get_pixmap(dpi=300)
            text = subprocess.run(
                ["tesseract", "stdin", "stdout", "-l", "rus+eng", "--psm", "3", "-c", "preserve_interword_spaces=1"],
                input=pixmap.tobytes("png"), capture_output=True, check=True,
            ).stdout.decode("utf-8").strip()
        pages.append({
            "doc_id": document["doc_id"], "page": number,
            "organization": document["organization"], "text": text,
        })
    print(document["doc_id"], len(pages), flush=True)
    return pages


def prepare(archive_path):
    documents = [json.loads(line) for line in (ROOT / "data/documents.jsonl").read_text().splitlines()]
    with zipfile.ZipFile(archive_path) as archive:
        jobs = [(doc, archive.read(doc["path"])) for doc in documents]
    folder = ROOT / "assets"
    folder.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool, (folder / "pages.new.jsonl").open("w") as file:
        for pages in pool.map(parse_report, jobs):
            for page in pages:
                file.write(json.dumps(page, ensure_ascii=False) + "\n")
                file.flush()

    (folder / "pages.new.jsonl").replace(folder / "pages.jsonl")
    pages = [json.loads(line) for line in (folder / "pages.jsonl").read_text().splitlines()]
    units = defaultdict(Counter)
    pattern = r"(?:в\s+)?(?:тысячах|миллионах|миллиардах)\s+(?:российских|белорусских)\s+рублей|(?:в\s+)?(?:тыс\.?|млн|млрд)\s*руб\.?"
    for page in pages:
        units[page["doc_id"]].update(re.findall(pattern, page["text"].lower()))
    unit_map = {doc: counts.most_common(1)[0][0] for doc, counts in units.items() if counts}
    (folder / "units.json").write_text(json.dumps(unit_map, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    prepare(parser.parse_args().archive)
