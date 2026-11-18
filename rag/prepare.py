"""Download public corpus and OCR weights. No labels are used here."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import requests
from rag.config import DATA, MODEL, TESSDATA

REPORTS_SHA = '2aaa2b146d21548bcea513f0be1564cf17dac47e819b1561d75cf7831bc1ef72'
MODEL_REVISION = '761b726dd34fb83930e26aab4e9ac3899aa1fa78'

def download(url, path, sha256=None):
    path = Path(path)
    if path.exists() and (not sha256 or hashlib.sha256(path.read_bytes()).hexdigest() == sha256):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with temp.open('wb') as target:
            for chunk in response.iter_content(1024 * 1024):
                target.write(chunk)
    if sha256 and hashlib.sha256(temp.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'Checksum mismatch: {path.name}')
    temp.replace(path)
    print(f'Downloaded {path.name}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-model', action='store_true')
    args = parser.parse_args()
    documents = [json.loads(line) for line in (DATA / 'documents.jsonl').read_text(encoding='utf-8').splitlines()]
    if not all((DATA / 'reports' / d['path']).exists() for d in documents):
        archive = DATA / 'reports.zip'
        download('https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip', archive, REPORTS_SHA)
        with zipfile.ZipFile(archive) as source:
            root = (DATA / 'reports').resolve()
            for member in source.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError('Unsafe archive member')
            source.extractall(root)
    for language in ('rus', 'eng'):
        download(f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/{language}.traineddata', TESSDATA / f'{language}.traineddata')
    if args.local_model:
        # Pin the model revision for reproducible indexing on another machine.
        revision_file = MODEL / 'revision.txt'
        if revision_file.exists():
            revision = revision_file.read_text().strip()
        else:
            revision = MODEL_REVISION
            MODEL.mkdir(parents=True, exist_ok=True)
            revision_file.write_text(revision)
        for name in ('tokenizer.json', 'onnx/model_quantized.onnx'):
            download(f'https://huggingface.co/Xenova/multilingual-e5-small/resolve/{revision}/{name}', MODEL / name)

if __name__ == '__main__':
    main()
