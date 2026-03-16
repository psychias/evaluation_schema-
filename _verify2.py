import json, glob

sources = {}
total_models = 0
total_eval_results = 0
for f in sorted(glob.glob('data/**/*.json', recursive=True)):
    parts = f.split('/')
    source = parts[1]
    with open(f) as fh:
        d = json.load(fh)
        n_results = len(d.get('evaluation_results', []))
    if source not in sources:
        sources[source] = {'models': 0, 'eval_results': 0}
    sources[source]['models'] += 1
    sources[source]['eval_results'] += n_results
    total_models += 1
    total_eval_results += n_results

print(f'Total JSON files (models): {total_models}')
print(f'Total evaluation results: {total_eval_results}')
print()
for s in sorted(sources.keys()):
    print(f'{s}: {sources[s]["models"]} models, {sources[s]["eval_results"]} eval results')

paper_sources = [s for s in sources if s.startswith('papers_')]
paper_model_counts = [sources[s]['models'] for s in paper_sources]
print(f'\nPaper sources: {len(paper_sources)}')
print(f'Paper model counts: {paper_model_counts}')
print(f'Paper mean models: {sum(paper_model_counts)/len(paper_model_counts):.1f}')
