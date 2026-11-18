"""Клиент эмбеддера и хранение векторов в SQLite."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
import sqlite3
import time

BASE = 'https://embeddings.parsers360.ru:10443'
DIM = 1024


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class Client:
    def __init__(self, root):
        import requests
        from dotenv import load_dotenv
        load_dotenv(root/'.env')
        key = os.environ.get('EMB_KEY','').strip()
        if not key or any(ord(c)<32 for c in key):
            raise ValueError('Set a valid single-line EMB_KEY in the local .env file.')
        self.session = requests.Session()
        self.session.headers['Authorization'] = 'Bearer '+key
        self.next_request = 0.0
        self.interval = 1.05

    def wait(self, seconds):
        deadline=time.monotonic()+max(0,seconds)
        while time.monotonic()<deadline:
            time.sleep(min(10,deadline-time.monotonic()))

    def request(self, method, path, **kwargs):
        import requests
        for attempt in range(6):
            self.wait(self.next_request-time.monotonic())
            self.next_request=time.monotonic()+self.interval
            try:
                response=self.session.request(method,BASE+path,timeout=(10,40),allow_redirects=False,**kwargs)
            except requests.RequestException as error:
                if attempt==5:
                    raise RuntimeError(f'Embedding network request failed: {type(error).__name__}') from None
                self.wait(min(2**attempt,16))
                continue
            if response.status_code in (401,403):
                raise RuntimeError('Embedding access denied. Check EMB_KEY; response body suppressed.')
            if response.status_code==429 or response.status_code>=500:
                if attempt==5:
                    raise RuntimeError(f'Embedding service HTTP {response.status_code}; retry limit reached.')
                retry=response.headers.get('Retry-After','')
                try:
                    delay=float(retry)
                except ValueError:
                    try:delay=(parsedate_to_datetime(retry)-datetime.now(timezone.utc)).total_seconds()
                    except (TypeError,ValueError,OverflowError):delay=min(2**attempt,16)
                print(json.dumps({'event':'retry','http_status':response.status_code,'wait_seconds':max(1,delay)}),flush=True)
                if not math.isfinite(delay):delay=min(2**attempt,16)
                self.wait(max(1,delay))
                continue
            if response.status_code!=200:
                raise RuntimeError(f'Embedding service HTTP {response.status_code}; response body suppressed.')
            try:return response.json()
            except ValueError:raise RuntimeError('Embedding service returned invalid JSON.') from None
        raise RuntimeError('Embedding request could not complete.')

    def limits(self):
        data=self.request('GET','/me')
        fields=['ratePerMinute','burst','maxConcurrent','maxChars','dim']
        profile={name:data.get(name) for name in fields}
        if profile['dim']!=DIM:
            raise RuntimeError(f'Unexpected embedding dimension: {profile["dim"]}')
        rate=profile.get('ratePerMinute')
        if isinstance(rate,(int,float)) and rate>0:
            self.interval=max(1.05,60/rate+0.05)
        return profile

    def embed(self,text,max_chars=20000):
        import numpy as np
        if not text.strip() or len(text)>max_chars:
            raise ValueError(f'Embedding text length must be 1..{max_chars} characters.')
        data=self.request('POST','/embed',json={'text':text})
        try:vector=np.asarray(data.get('embedding',[]),dtype=np.float32)
        except (TypeError,ValueError):raise RuntimeError('Embedding service returned non-numeric values.') from None
        if data.get('dim')!=DIM or vector.shape!=(DIM,) or not np.isfinite(vector).all():
            raise RuntimeError('Invalid embedding dimensions or values.')
        norm=float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm<1e-8:
            raise RuntimeError('Embedding vector has invalid norm.')
        return vector/norm


def chunks(index):
    connection=sqlite3.connect(index.resolve().as_uri()+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    try:
        return [dict(r) for r in connection.execute('SELECT chunk_id,doc_id,pdf_page,kind,text FROM chunks ORDER BY rowid')]
    finally:connection.close()


def cache_open(path):
    connection=sqlite3.connect(path,timeout=15)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=FULL')
    connection.execute('CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT)')
    connection.execute('CREATE TABLE IF NOT EXISTS vectors(text_sha256 TEXT PRIMARY KEY,dim INTEGER NOT NULL,vector BLOB NOT NULL)')
    signature=json.dumps({'endpoint':BASE+'/embed','dim':DIM,'model_version':'not_reported_by_service'},sort_keys=True)
    old=connection.execute('SELECT value FROM config WHERE key=?',('signature',)).fetchone()
    if old and old[0]!=signature:raise RuntimeError('Embedding cache uses a different provider/configuration.')
    connection.execute('INSERT OR IGNORE INTO config VALUES (?,?)',('signature',signature));connection.commit()
    return connection


def cached(connection, hashes):
    import numpy as np
    wanted=set(hashes);found={}
    for key,dim,blob in connection.execute('SELECT text_sha256,dim,vector FROM vectors'):
        if key not in wanted or dim!=DIM or len(blob)!=DIM*4:continue
        vector=np.frombuffer(blob,dtype=np.float32)
        norm=float(np.linalg.norm(vector))
        if np.isfinite(vector).all() and abs(norm-1)<1e-3:found[key]=vector
    return found
