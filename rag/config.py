from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=False)
DATA = ROOT / 'data'
INDEX = ROOT / 'index'
MODEL = ROOT / 'models' / 'multilingual-e5-small'
TESSDATA = ROOT / 'models' / 'tessdata'

def env_bool(name, default=True):
    return os.getenv(name, str(default)).lower() in {'1', 'true', 'yes'}
