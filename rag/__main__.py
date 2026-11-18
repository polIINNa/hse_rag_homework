import argparse
import json
from rag.pipeline import RAG

parser = argparse.ArgumentParser(description='Ask a question about the financial reports')
parser.add_argument('question')
args = parser.parse_args()
print(json.dumps(RAG().rag(args.question), ensure_ascii=False, indent=2))
