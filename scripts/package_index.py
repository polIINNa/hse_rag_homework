"""Упаковать проверенный индекс для повторного запуска без OCR и индексации."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pages = [json.loads(s) for s in (INDEX / "pages.jsonl").read_text().splitlines()]
    groups = [json.loads(s) for s in (INDEX / "groups.jsonl").read_text().splitlines()]
    vectors = np.load(INDEX / "vectors.npy", allow_pickle=False)
    keys = {(p["doc_id"], p["page"]) for p in pages}
    if len(keys) != len(pages) or len(pages) != 3925:
        raise ValueError("Нарушена целостность страниц")
    if vectors.shape != (len(groups), 1024) or not np.isfinite(vectors).all():
        raise ValueError("Неверная матрица эмбеддингов")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Векторы не нормированы")
    coverage = {(g["doc_id"], p) for g in groups for p in g["pages"]}
    if coverage != keys:
        raise ValueError("Индекс покрывает не все страницы")
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    names = ["pages.jsonl", "groups.jsonl", "vectors.npy", "metadata.json"]
    checksums = {name: digest(INDEX / name) for name in names}
    with zipfile.ZipFile(out / "index.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for name in names:
            target.write(INDEX / name, arcname=name)
    checksums["index.zip"] = digest(out / "index.zip")
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2))
    documents = [json.loads(s) for s in (ROOT / "data" / "documents.jsonl").read_text().splitlines()]
    source = [{"doc_id": d["doc_id"], "path": d["path"], "pages": d["pages"],
               "sha256": digest(ROOT / "data" / "reports" / d["path"])} for d in documents]
    (out / "source_documents.json").write_text(json.dumps(source, ensure_ascii=False, indent=2))
    print(f"{len(pages)} страниц, {len(groups)} векторов, архив {(out / 'index.zip').stat().st_size / 2**20:.1f} МБ")


if __name__ == "__main__":
    main()
