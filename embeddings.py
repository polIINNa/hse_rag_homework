from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import Lock
import time

import numpy as np
import requests


class Embedder:
    def __init__(self):
        self.key = os.environ["EMB_KEY"]
        self.lock = Lock()
        self.next_request = 0.0

    def embed(self, text):
        while True:
            with self.lock:
                delay = max(0, self.next_request - time.monotonic())
                self.next_request = max(self.next_request, time.monotonic()) + 1.05
            time.sleep(delay)
            response = requests.post(
                "https://embeddings.parsers360.ru:10443/embed",
                headers={"Authorization": "Bearer " + self.key},
                json={"text": text[:20000]}, timeout=90,
            )
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 5)))
                continue
            response.raise_for_status()
            return np.asarray(response.json()["embedding"], dtype=np.float32)


def build():
    folder = Path(__file__).resolve().parent / "assets"
    pages = [json.loads(line) for line in (folder / "pages.jsonl").read_text().splitlines()]
    groups = []
    current = []
    for i, page in enumerate(pages):
        if current and (
            page["doc_id"] != pages[current[0]]["doc_id"]
            or len(current) == 8
        ):
            groups.append(current)
            current = []
        current.append(i)
    groups.append(current)
    documents = [
        pages[ids[0]]["organization"] + " " + pages[ids[0]]["doc_id"] + "\n"
        + "\n".join(pages[j]["text"][:1800] for j in ids)
        for ids in groups
    ]
    client = Embedder()
    vectors = []
    checkpoint = folder / "embedding_checkpoint.jsonl"
    if checkpoint.exists():
        vectors = [json.loads(line)["embedding"] for line in checkpoint.read_text().splitlines()]
    with checkpoint.open("a") as file, ThreadPoolExecutor(max_workers=4) as pool:
        for vector in pool.map(client.embed, documents[len(vectors):]):
            vectors.append(vector.tolist())
            file.write(json.dumps({"embedding": vector.tolist()}) + "\n")
            file.flush()
            if len(vectors) % 25 == 0:
                print("embeddings", len(vectors), "/", len(groups), flush=True)
    np.savez_compressed(folder / "vectors.npz", vectors=np.asarray(vectors, dtype=np.float32))
    (folder / "groups.json").write_text(json.dumps(groups))
    checkpoint.unlink()
    print("ready", len(groups), flush=True)


if __name__ == "__main__":
    build()
