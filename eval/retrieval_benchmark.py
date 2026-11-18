"""Train-only retrieval diagnostic. Never tunes on the test labels."""
import json
from statistics import mean
from rag.config import DATA, ROOT
from rag.retrieval import Retriever, context_from_pages
from eval.eval_functions import score_context

def main():
    items = [json.loads(line) for line in (DATA / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    retriever = Retriever()
    results = []
    for i, item in enumerate(items):
        pages = retriever.search(item['question'], limit=14)
        metrics = {str(k): score_context(item['gold_context'], context_from_pages(pages[:k])) for k in (3, 4, 6, 10, 14)}
        results.append({'question_id': item['question_id'], 'metrics': metrics})
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/{len(items)}', flush=True)
    summary = {str(k): {metric: mean(row['metrics'][str(k)][metric] for row in results)
                       for metric in ('context_recall', 'context_precision')} for k in (3, 4, 6, 10, 14)}
    path = ROOT / 'results/train_retrieval.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'summary': summary, 'results': results}, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
