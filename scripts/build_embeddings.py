"""Построить векторы чанков с возможностью продолжения."""

import argparse
import json
from pathlib import Path
import sqlite3
import time

from filelock import FileLock

from rag.embeddings import Client, DIM, cache_open, cached, chunks, text_hash


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def progress(rows,ready):
    hashes={text_hash(r['text']) for r in rows}
    missing=len(hashes-set(ready))
    return {'chunks':len(rows),'unique_texts':len(hashes),'ready':len(hashes&set(ready)),
            'remaining':missing,'complete':missing==0,'theoretical_minutes_at_60rpm':round(missing/60,1)}


def build(args,root):
    rows=chunks(args.index)
    initial_stat=(args.index.stat().st_size,args.index.stat().st_mtime_ns)
    with FileLock(args.cache.with_suffix('.lock'),timeout=0):
        connection=cache_open(args.cache)
        try:
            ready=cached(connection,[text_hash(r['text']) for r in rows])
            initial=progress(rows,ready)
            missing=[];seen=set()
            for row in rows:
                key=text_hash(row['text'])
                if key in ready or key in seen:continue
                seen.add(key);missing.append((row,key))
            if args.limit is not None:missing=missing[:args.limit]
            status={**initial,'state':'running','new_vectors_this_run':0}
            atomic_json(args.status,status)
            if not missing:
                status['state']='complete' if initial['complete'] else 'partial'
                atomic_json(args.status,status)
                print(json.dumps(status,ensure_ascii=False));return
            started=time.monotonic()
            client=Client(root);profile=client.limits()
            if any(len(row['text'])>(profile.get('maxChars') or 20000) for row,key in missing):
                raise ValueError('Some selected chunks exceed the service text limit; rebuild chunks first.')
            connection.execute('INSERT OR REPLACE INTO config VALUES (?,?)',('limits',json.dumps(profile)));connection.commit()
            status['service_limits']=profile
            for row,key in missing:
                if initial_stat!=(args.index.stat().st_size,args.index.stat().st_mtime_ns):
                    raise RuntimeError('Text index changed during embedding. Resume against the rebuilt index.')
                vector=client.embed(row['text'],profile.get('maxChars') or 20000)
                connection.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?)',(key,DIM,vector.tobytes()))
                connection.commit()
                ready[key]=vector
                status={**status,**progress(rows,ready),'new_vectors_this_run':status['new_vectors_this_run']+1,
                        'last_doc_id':row['doc_id'],'last_pdf_page':row['pdf_page']}
                status['elapsed_seconds']=round(time.monotonic()-started,1)
                status['estimated_minutes_remaining']=round(status['remaining']*max(client.interval,status['elapsed_seconds']/status['new_vectors_this_run'])/60,1)
                atomic_json(args.status,status)
                print(json.dumps(status,ensure_ascii=False),flush=True)
            status['state']='complete' if status['complete'] else 'partial'
            atomic_json(args.status,status)
        except BaseException as error:
            state=locals().get('status',{})
            atomic_json(args.status,{**state,'state':'interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                                    'error_type':type(error).__name__})
            raise
        finally:connection.close()


def status(args):
    rows=chunks(args.index)
    if not args.cache.exists():return progress(rows,{})
    connection=sqlite3.connect(args.cache.resolve().as_uri()+'?mode=ro',uri=True)
    try:return progress(rows,cached(connection,[text_hash(r['text']) for r in rows]))
    finally:connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='build', choices=['build', 'status'])
    parser.add_argument('--index', type=Path, default=Path('index/corpus.sqlite'))
    parser.add_argument('--cache', type=Path, default=Path('index/embedding_cache.sqlite'))
    parser.add_argument('--status-file', dest='status', type=Path, default=Path('index/embedding_status.json'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.command == 'status':
        print(json.dumps(status(args), ensure_ascii=False, indent=2))
        return
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    build(args, Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
