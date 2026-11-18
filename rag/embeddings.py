"""Quantized multilingual E5 with explicit mean pooling and L2 normalization."""
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time
import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer
from rag.config import MODEL

class RemoteEmbedder:
    """Teacher-provided 1024-dimensional embedding service, separate EMB_KEY."""
    def __init__(self, workers=4):
        self.key = os.getenv('EMB_KEY')
        if not self.key:
            raise ValueError('Set your personal EMB_KEY in .env')
        self.workers = workers
        self.local = threading.local()

    def _one(self, text):
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
            self.local.session.headers.update({'Authorization': 'Bearer ' + self.key})
        for attempt in range(6):
            try:
                response = self.local.session.post('https://embeddings.parsers360.ru:10443/embed',
                    json={'text': text}, timeout=(10, 30))
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                    print(f'Embedding retry: HTTP {response.status_code}, attempt {attempt + 1}', flush=True)
                    delay = max(65, float(response.headers.get('Retry-After', '65'))) if response.status_code == 429 else min(40, 2 ** (attempt + 1))
                    time.sleep(delay)
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f'Embedding service HTTP {response.status_code}')
                data = response.json()
                vector = np.asarray(data['embedding'], dtype=np.float32)
                if vector.shape != (1024,) or not np.isfinite(vector).all():
                    raise ValueError('Embedding service returned an invalid vector')
                norm = np.linalg.norm(vector)
                if norm <= 0:
                    raise ValueError('Embedding service returned a zero vector')
                return vector / norm
            except (requests.Timeout, requests.ConnectionError):
                print(f'Embedding network retry, attempt {attempt + 1}', flush=True)
                if attempt == 5:
                    raise RuntimeError('Embedding network error after retries') from None
                time.sleep(min(30, 2 ** attempt))

    def encode(self, texts, query=False, batch_size=16):
        if len(texts) == 1:
            return np.asarray([self._one(texts[0])], dtype=np.float32)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return np.asarray(list(pool.map(self._one, texts)), dtype=np.float32)

class Embedder:
    def __init__(self, directory=MODEL, threads=4):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'onnx/model_quantized.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=1, pad_token='<pad>')
        self.inputs = {value.name for value in self.session.get_inputs()}

    def encode(self, texts, query=False, batch_size=16):
        outputs = []
        prefix = 'query: ' if query else 'passage: '
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch([prefix + text for text in texts[start:start + batch_size]])
            feeds = {'input_ids': np.array([item.ids for item in encoded], dtype=np.int64),
                     'attention_mask': np.array([item.attention_mask for item in encoded], dtype=np.int64),
                     'token_type_ids': np.array([item.type_ids for item in encoded], dtype=np.int64)}
            hidden = self.session.run(None, {key: value for key, value in feeds.items() if key in self.inputs})[0]
            mask = feeds['attention_mask'][..., None]
            vectors = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            outputs.append(vectors.astype(np.float32))
        return np.concatenate(outputs) if outputs else np.empty((0, 384), dtype=np.float32)
