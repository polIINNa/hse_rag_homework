import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from filelock import FileLock
from pypdf import PdfWriter


ROOT = Path(__file__).resolve().parents[1]


class CorpusCheckpointTests(unittest.TestCase):
    def test_resume_after_changes_and_invalid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / 'data/reports'
            reports.mkdir(parents=True)
            document = {'doc_id': 'demo', 'organization': 'Test', 'path': 'demo.pdf', 'pages': 1}
            (root / 'data/documents.jsonl').write_text(json.dumps(document) + '\n')
            output = root / 'prepared'
            page = output / 'pages/demo/0001.json'
            command = [
                sys.executable, '-B', '-m', 'scripts.prepare_corpus',
                '--root', str(root), '--output', str(output), '--stage', 'native',
            ]

            def write_pdf(version, encrypted=False):
                writer = PdfWriter()
                writer.add_blank_page(width=612, height=792)
                writer.add_metadata({'/Version': version})
                if encrypted:
                    writer.encrypt(user_password='', owner_password='test-owner', algorithm='AES-256')
                with (reports / 'demo.pdf').open('wb') as file:
                    writer.write(file)

            def prepare(check=True):
                return subprocess.run(command, cwd=ROOT, check=check, capture_output=True, text=True)

            write_pdf('A')
            prepare()
            initial = json.loads(page.read_text())
            stamp = page.stat().st_mtime_ns
            self.assertIn('"processed": 0', prepare().stdout)
            self.assertEqual(page.stat().st_mtime_ns, stamp)

            initial['ocr'] = {'status': 'complete', 'fingerprint': 'old', 'text': 'STALE'}
            initial['tables'] = {'status': 'complete', 'items': []}
            page.write_text(json.dumps(initial))
            write_pdf('B')
            prepare()
            changed = json.loads(page.read_text())
            self.assertNotEqual(changed['native']['fingerprint'], initial['native']['fingerprint'])
            self.assertEqual(changed['ocr']['status'], 'pending')
            self.assertEqual(changed['tables']['status'], 'pending')
            self.assertEqual(changed['text'], '')

            page.write_text('{broken')
            prepare()
            self.assertEqual(json.loads(page.read_text())['native']['status'], 'complete')

            write_pdf('C', encrypted=True)
            prepare()
            self.assertEqual(json.loads(page.read_text())['native']['status'], 'complete')

            with FileLock(output / '.prepare.lock'):
                blocked = prepare(check=False)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('Timeout', blocked.stderr)
