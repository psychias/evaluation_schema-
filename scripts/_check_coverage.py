#!/usr/bin/env python3
import csv

with open('analysis_output/coverage_stats.csv') as f:
    rows = list(csv.DictReader(f))

total = len(rows)
pt_docs = [r for r in rows if float(r['pct_prompt_template']) > 0]
nshot_docs = [r for r in rows if float(r['pct_n_shot']) > 0]
temp_docs = [r for r in rows if float(r['pct_temperature']) > 0]

print(f"Total sources: {total}")
print(f"prompt_template > 0%: {len(pt_docs)}/{total} = {100*len(pt_docs)/total:.1f}%")
print(f"n_shot > 0%: {len(nshot_docs)}/{total} = {100*len(nshot_docs)/total:.1f}%")
print(f"temperature > 0%: {len(temp_docs)}/{total} = {100*len(temp_docs)/total:.1f}%")
print()
print("Sources with prompt_template > 0%:")
for r in pt_docs:
    print(f"  {r['source']}: {r['pct_prompt_template']}%")
print()
nshot_absent = [r['source'] for r in rows if float(r['pct_n_shot']) == 0]
print(f"Sources with n_shot = 0% ({len(nshot_absent)}):")
for n in nshot_absent:
    print(f"  {n}")

# Paper mean
paper_rows = [r for r in rows if r['source'].startswith('papers_')]
n_papers = len(paper_rows)
mean_nshot = sum(float(r['pct_n_shot']) for r in paper_rows) / n_papers
mean_harness = sum(float(r['pct_harness']) for r in paper_rows) / n_papers
mean_pt = sum(float(r['pct_prompt_template']) for r in paper_rows) / n_papers
mean_temp = sum(float(r['pct_temperature']) for r in paper_rows) / n_papers
mean_models = sum(int(r['n_records']) for r in paper_rows) / n_papers
print(f"\nPaper mean (n={n_papers}): #M={mean_models:.1f}, n-shot={mean_nshot:.0f}%, harness={mean_harness:.0f}%, pt={mean_pt:.0f}%, temp={mean_temp:.0f}%")

# Overall mean across all sources
all_nshot = sum(float(r['pct_n_shot']) for r in rows) / total
all_harness = sum(float(r['pct_harness']) for r in rows) / total
all_pt = sum(float(r['pct_prompt_template']) for r in rows) / total
all_temp = sum(float(r['pct_temperature']) for r in rows) / total
print(f"Overall mean: n-shot={all_nshot:.1f}%, harness={all_harness:.1f}%, pt={all_pt:.1f}%, temp={all_temp:.1f}%")

# median delta per benchmark from collision_pairs.csv
import statistics
with open('analysis_output/collision_pairs.csv') as f:
    pairs = list(csv.DictReader(f))
benchmarks = sorted(set(r['benchmark'] for r in pairs))
print(f"\nCollision pairs per benchmark (median delta):")
zero_median = 0
for b in benchmarks:
    bp = [r for r in pairs if r['benchmark'] == b]
    deltas = [abs(float(r['delta'])) for r in bp]
    med = statistics.median(deltas)
    if med == 0:
        zero_median += 1
    print(f"  {b}: n={len(bp)}, med|d|={med:.3f}")
print(f"Benchmarks with median delta = 0: {zero_median}/{len(benchmarks)}")
