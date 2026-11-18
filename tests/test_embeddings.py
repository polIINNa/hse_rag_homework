"""Проверки сохранения векторов и повторных запросов без доступа к сети."""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts import build_embeddings as m


class FakeClient:
    calls=0
    fail_at=None
    def __init__(self,root):self.interval=0
    def limits(self):return {'dim':1024,'maxChars':20000,'ratePerMinute':60}
    def embed(self,text,max_chars=20000):
        type(self).calls+=1
        if self.fail_at==type(self).calls:raise RuntimeError('simulated network failure')
        v=np.zeros(1024,dtype=np.float32);v[len(text)%1024]=1
        return v


class EmbeddingChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.args=argparse.Namespace(index=root/'corpus.sqlite',cache=root/'cache.sqlite',status=root/'status.json',limit=None)
        c=sqlite3.connect(self.args.index)
        c.execute('CREATE TABLE chunks(chunk_id TEXT,doc_id TEXT,pdf_page INTEGER,kind TEXT,text TEXT)')
        c.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)', [('a','demo',1,'text','first'),('b','demo',2,'text','second'),('c','demo',3,'text','third')]);c.commit();c.close()
        FakeClient.calls=0;FakeClient.fail_at=None
    def tearDown(self):self.temp.cleanup()
    def test_resume_and_changed_text(self):
        self.args.limit=1
        with patch.object(m,'Client',FakeClient):m.build(self.args,Path(self.temp.name))
        self.assertEqual(m.status(self.args)['ready'],1)
        self.args.limit=None
        with patch.object(m,'Client',FakeClient):m.build(self.args,Path(self.temp.name))
        self.assertEqual(FakeClient.calls,3)
        with patch.object(m,'Client',FakeClient):m.build(self.args,Path(self.temp.name))
        self.assertEqual(FakeClient.calls,3)
        c=sqlite3.connect(self.args.index);c.execute("UPDATE chunks SET text='changed' WHERE chunk_id='b'");c.commit();c.close()
        with patch.object(m,'Client',FakeClient):m.build(self.args,Path(self.temp.name))
        self.assertEqual(FakeClient.calls,4)
    def test_failure_keeps_committed_vectors(self):
        FakeClient.fail_at=2
        with patch.object(m,'Client',FakeClient), self.assertRaises(RuntimeError):m.build(self.args,Path(self.temp.name))
        self.assertEqual(m.status(self.args)['ready'],1)
        FakeClient.fail_at=None
        with patch.object(m,'Client',FakeClient):m.build(self.args,Path(self.temp.name))
        self.assertTrue(m.status(self.args)['complete'])
        self.assertEqual(FakeClient.calls,4)
    def test_retry_after(self):
        class Response:
            def __init__(self,code):self.status_code=code;self.headers={'Retry-After':'3'}
            def json(self):return {'ok':True}
        client=m.Client.__new__(m.Client);client.next_request=0;client.interval=0
        waits=[];client.wait=lambda seconds:waits.append(seconds)
        class Session:
            replies=iter([Response(429),Response(200)])
            def request(self,*args,**kwargs):return next(self.replies)
        client.session=Session()
        self.assertEqual(client.request('GET','/me'),{'ok':True})
        self.assertIn(3.0,waits)


if __name__=='__main__':unittest.main()
