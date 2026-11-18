"""Build overlapping page-local chunks and the embedding matrix (no QA labels)."""
import argparse
import hashlib
import json
import time
import numpy as np
from rag.config import INDEX, MODEL
from rag.embeddings import Embedder, RemoteEmbedder

def chunks_for_page(page, max_chars=1500, overlap=250):
    text = page['text']
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield {'doc_id': page['doc_id'], 'page': page['page'], 'text': chunk,
                   'organization': page['organization'], 'offset': start}
        if end == len(text):
            break
        start = max(start + 1, end - overlap)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--backend', choices=['remote', 'local'], default='local')
    parser.add_argument('--reuse', action='store_true', help='Reuse unchanged chunks from a completed local index')
    args = parser.parse_args()
    started = time.perf_counter()
    source = INDEX / 'pages.jsonl'
    pages = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    chunks = [chunk for page in pages for chunk in chunks_for_page(page)]
    old_vectors = None
    old_lookup = {}
    def chunk_key(chunk):
        return (chunk['doc_id'], chunk['page'], chunk['organization'], chunk['text'])
    if args.reuse and (INDEX / 'manifest.json').exists():
        old_manifest = json.loads((INDEX / 'manifest.json').read_text())
        if args.backend != 'local' or old_manifest.get('backend') != 'local' or old_manifest['revision'] != (MODEL / 'revision.txt').read_text().strip():
            raise ValueError('Reuse requires the same fixed local embedding model')
        old_chunks = [json.loads(line) for line in (INDEX / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        old_lookup = {chunk_key(chunk): index for index, chunk in enumerate(old_chunks)}
        old_vectors = np.load(INDEX / 'vectors.npy', mmap_mode='r')
    staged_chunks = INDEX / 'chunks.build.jsonl'
    staged_chunks.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks), encoding='utf-8')
    chunk_hash = hashlib.sha256(staged_chunks.read_bytes()).hexdigest()
    remote = args.backend == 'remote'
    dimension = 1024 if remote else 384
    temporary_vectors = INDEX / ('remote_vectors.tmp.npy' if remote else 'vectors.tmp.npy')
    progress_path = INDEX / ('remote_embedding_progress.json' if remote else 'embedding_progress.json')
    offset = 0
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['chunks_sha256'] != chunk_hash:
            raise ValueError('Chunks changed: cannot resume embeddings')
        offset = progress['completed']
    encoder = RemoteEmbedder(workers=4) if remote else Embedder(threads=args.threads)
    output = np.lib.format.open_memmap(temporary_vectors, mode='r+' if offset else 'w+', dtype=np.float32, shape=(len(chunks), dimension))
    for start in range(offset, len(chunks), 128):
        batch = chunks[start:start + 128]
        missing = []
        for index, chunk in enumerate(batch):
            old_index = old_lookup.get(chunk_key(chunk))
            if old_index is None:
                missing.append(index)
            else:
                output[start + index] = old_vectors[old_index]
        if missing:
            texts = [f"{batch[i]['organization']} | {batch[i]['doc_id']} | {batch[i]['text']}" for i in missing]
            output[[start + i for i in missing]] = encoder.encode(texts)
        output.flush()
        progress_temp = progress_path.with_suffix('.tmp')
        progress_temp.write_text(json.dumps({'completed': start + len(batch), 'chunks_sha256': chunk_hash}), encoding='utf-8')
        progress_temp.replace(progress_path)
        print(f'Embedded {start + len(batch)}/{len(chunks)}', flush=True)
    if not np.isfinite(output).all() or not np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-4):
        raise ValueError('Invalid embeddings: incomplete or non-normalized vectors')
    del output
    if old_vectors is not None:
        del old_vectors
    temporary_vectors.replace(INDEX / 'vectors.npy')
    staged_chunks.replace(INDEX / 'chunks.jsonl')
    manifest = {'pages_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'chunks': len(chunks), 'dimension': dimension, 'backend': args.backend,
                'model': 'embeddings.parsers360.ru:10443/embed' if remote else 'Xenova/multilingual-e5-small',
                'revision': 'service-managed' if remote else (MODEL / 'revision.txt').read_text().strip(),
                'chunk_chars': 1500, 'overlap_chars': 250, 'seconds': time.perf_counter() - started}
    (INDEX / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    main()
