import csv
from collections import Counter
from statistics import median

# Count collision pairs
with open('analysis_output/collision_pairs.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f'Total collision pairs: {len(rows)}')
    
    benchmarks = set(r['benchmark'] for r in rows)
    print(f'Unique benchmarks in collisions: {len(benchmarks)} -> {sorted(benchmarks)}')
    
    source_pairs = set((r['source_a'], r['source_b']) for r in rows)
    print(f'Unique source pairs: {len(source_pairs)}')
    
    bench_counts = Counter(r['benchmark'] for r in rows)
    print(f'Per benchmark: {dict(bench_counts)}')
    
    for b in sorted(benchmarks):
        deltas = [abs(float(r['delta'])) for r in rows if r['benchmark'] == b]
        raw_deltas = [float(r['delta']) for r in rows if r['benchmark'] == b]
        print(f'  {b}: n={len(deltas)}, median |delta|={median(deltas):.4f}, median delta={median(raw_deltas):.4f}, range=[{min(raw_deltas):.3f}, {max(raw_deltas):.3f}]')
    
    big_delta = [r for r in rows if abs(float(r['delta'])) > 0.01]
    print(f'Pairs with |delta| > 0.01: {len(big_delta)}')
    big_delta_benchmarks = Counter(r['benchmark'] for r in big_delta)
    print(f'  Per benchmark: {dict(big_delta_benchmarks)}')

    # Benchmarks with median delta exactly 0
    for b in sorted(benchmarks):
        raw_deltas = [float(r['delta']) for r in rows if r['benchmark'] == b]
        med = median(raw_deltas)
        print(f'  {b}: median delta = {med}')

# Coverage stats
print('\n--- COVERAGE STATS ---')
with open('analysis_output/coverage_stats.csv') as f:
    reader = csv.DictReader(f)
    cov_rows = list(reader)
    print(f'Number of sources: {len(cov_rows)}')
    total_records = sum(int(r['n_records']) for r in cov_rows)
    print(f'Total records: {total_records}')
    
    # n_shot coverage: count sources with pct_n_shot > 0
    n_shot_sources = sum(1 for r in cov_rows if float(r['pct_n_shot']) > 0)
    print(f'Sources with n_shot > 0: {n_shot_sources} of {len(cov_rows)}')
    print(f'  Pct: {n_shot_sources/len(cov_rows)*100:.1f}%')
    
    # prompt template
    pt_sources = sum(1 for r in cov_rows if float(r['pct_prompt_template']) > 0)
    print(f'Sources with prompt_template > 0: {pt_sources} of {len(cov_rows)}')
    print(f'  Pct: {pt_sources/len(cov_rows)*100:.1f}%')
    
    # temperature
    temp_sources = sum(1 for r in cov_rows if float(r['pct_temperature']) > 0)
    print(f'Sources with temperature > 0: {temp_sources} of {len(cov_rows)}')
    print(f'  Pct: {temp_sources/len(cov_rows)*100:.1f}%')
    
    # Sources without n_shot
    no_nshot = [r['source'] for r in cov_rows if float(r['pct_n_shot']) == 0]
    print(f'Sources with 0% n_shot: {no_nshot}')
    
    # Sources without prompt template
    no_pt = [r['source'] for r in cov_rows if float(r['pct_prompt_template']) == 0]
    print(f'Sources with 0% prompt_template: {no_pt} ({len(no_pt)} sources)')
    
    # Overall means
    n_shot_mean = sum(float(r['pct_n_shot']) for r in cov_rows) / len(cov_rows)
    harness_mean = sum(float(r['pct_harness']) for r in cov_rows) / len(cov_rows)
    pt_mean = sum(float(r['pct_prompt_template']) for r in cov_rows) / len(cov_rows)
    temp_mean = sum(float(r['pct_temperature']) for r in cov_rows) / len(cov_rows)
    print(f'Overall mean n_shot: {n_shot_mean:.1f}%')
    print(f'Overall mean harness: {harness_mean:.1f}%')
    print(f'Overall mean prompt_template: {pt_mean:.1f}%')
    print(f'Overall mean temperature: {temp_mean:.1f}%')
    
    # Model counts per source
    for r in cov_rows:
        print(f'  {r["source"]}: {r["n_records"]} records')

# Rank instability
print('\n--- RANK INSTABILITY ---')
with open('analysis_output/rank_instability.csv') as f:
    reader = csv.DictReader(f)
    rank_rows = list(reader)
    print(f'Total rank pairs: {len(rank_rows)}')
    
    # Min tau
    taus = [float(r['tau_b']) for r in rank_rows]
    print(f'Min tau: {min(taus)}, Max tau: {max(taus)}')
    
    # GSM8K specific taus
    gsm_rows = [r for r in rank_rows if r['benchmark'] == 'GSM8K']
    gsm_taus = [float(r['tau_b']) for r in gsm_rows]
    print(f'GSM8K taus: {sorted(gsm_taus)}')
    print(f'GSM8K tau range: {min(gsm_taus):.4f} to {max(gsm_taus):.4f}')

# Variance decomp
print('\n--- VARIANCE DECOMP ---')
with open('analysis_output/variance_decomp.csv') as f:
    reader = csv.DictReader(f)
    var_rows = list(reader)
    for r in var_rows:
        print(f'  {r["predictor"]}: coef={r["coef"]}, p={r["pvalue"]}, partial_r2={r["partial_r2"]}')

# Per benchmark OLS
print('\n--- PER BENCHMARK OLS ---')
with open('analysis_output/per_benchmark_ols.csv') as f:
    reader = csv.DictReader(f)
    ols_rows = list(reader)
    for r in ols_rows:
        print(f'  {r["benchmark"]}/{r["predictor"]}: n={r["n"]}, partial_r2={r["partial_r2"]}, f2={r["f2_cohen"]}, p={r["p_value"]}, full_R2={r["full_model_r2"]}, beta={r["beta"]}')

# Rank flips
print('\n--- RANK FLIPS ---')
with open('analysis_output/collision_pairs.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    # For rank flips, need to check how that's computed...
    # "48.9% of all collision pairs" - this is about pairwise ordering reversals

# Power simulation
print('\n--- POWER SIMULATION ---')
with open('analysis_output/power_simulation.csv') as f:
    reader = csv.DictReader(f)
    pow_rows = list(reader)
    # k=19 power
    for r in pow_rows:
        if r['k'] == '19':
            print(f'  k=19: mean_power={r["mean_power"]}')
    # k=20
    for r in pow_rows:
        if r['k'] == '20':
            print(f'  k=20: mean_power={r["mean_power"]}')
    # Max power
    max_pow = max(pow_rows, key=lambda r: float(r['mean_power']))
    print(f'  Max power: k={max_pow["k"]}, mean_power={max_pow["mean_power"]}')
    # k_80
    for r in pow_rows:
        print(f'  k={r["k"]}: k_80={r["k_80"]}')
