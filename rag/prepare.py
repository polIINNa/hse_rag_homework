"""Восстановить готовый индекс из артефакта, не вызывая внешние API."""
import hashlib
import json
from pathlib import Path
import zipfile

from .config import INDEX, ROOT

def main():
    archive = ROOT / "artifacts" / "index.zip"
    checksums = json.loads((ROOT / "artifacts" / "checksums.json").read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != checksums["index.zip"]:
        raise ValueError("Контрольная сумма архива не совпала")
    with zipfile.ZipFile(archive) as source:
        allowed = {"pages.jsonl", "groups.jsonl", "vectors.npy", "metadata.json"}
        if set(source.namelist()) != allowed:
            raise ValueError("Неожиданное содержимое архива")
        INDEX.mkdir(exist_ok=True)
        for name in sorted(allowed):
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != checksums[name]:
                raise ValueError(f"Повреждён файл {name}")
            (INDEX / name).write_bytes(data)
    print("Индекс готов:", INDEX)


if __name__ == "__main__":
    main()

