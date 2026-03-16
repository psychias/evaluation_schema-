#!/usr/bin/env python3
import json, glob, os

papers = {}
for d in sorted(glob.glob('data/papers_*')):
    name = os.path.basename(d)
    arxiv_id = name.replace('papers_', '')
    jsons = sorted(glob.glob(os.path.join(d, '**', '*.json'), recursive=True))
    if not jsons:
        continue
    count = len(jsons)
    r = json.load(open(jsons[0]))
    title = r.get('source_metadata', {}).get('source_name', '?')
    papers[arxiv_id] = (title, count)

for k, (t, c) in sorted(papers.items()):
    short_t = t[:50] + '...' if len(t) > 50 else t
    print(f'{k} | {short_t} | {c}')
print(f'Total papers: {len(papers)}')
print(f'Total records: {sum(c for _, c in papers.values())}')
