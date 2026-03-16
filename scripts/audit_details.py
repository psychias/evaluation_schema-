"""Get detailed numbers for paper update."""
import csv
import json
import glob
from collections import Counter
from statistics import median

# Collision pair details per benchmark
with open("analysis_output/collision_pairs.csv") as f:
    crows = list(csv.DictReader(f))

print("=== PER-BENCHMARK COLLISION DETAIL ===")
benchmarks = sorted(set(r["benchmark"] for r in crows))
for b in benchmarks:
    pairs = [r for r in crows if r["benchmark"] == b]
    deltas = [float(r["delta"]) for r in pairs]
    abs_deltas = [abs(d) for d in deltas]
    med = median(deltas)
    med_abs = median(abs_deltas)
    mn = min(deltas)
    mx = max(deltas)
    print(f"  {b}: n={len(pairs)}, median_delta={med:.3f}, median_|delta|={med_abs:.4f}, range=[{mn:.3f}, {mx:.3f}]")

# Rank flips
print("\n=== RANK FLIP ANALYSIS ===")
total_flips = sum(1 for r in crows if float(r["delta"]) != 0)
large_delta = [r for r in crows if abs(float(r["delta"])) > 0.01]
print(f"  Total pairs: {len(crows)}")
print(f"  With delta != 0: {total_flips}")
print(f"  Flip rate: {100*total_flips/len(crows):.1f}%")
print(f"  Pairs with |delta| > 0.01: {len(large_delta)}")
for b in benchmarks:
    big = [r for r in large_delta if r["benchmark"] == b]
    if big:
        print(f"    {b}: {len(big)} pairs with |delta|>0.01")

# Check rank instability for leaderboard pairs
print("\n=== RANK INSTABILITY: LEADERBOARD PAIRS ===")
with open("analysis_output/rank_instability.csv") as f:
    rrows = list(csv.DictReader(f))
leaderboards = {"hfopenllm_v2", "alpacaeval2", "chatbot_arena", "mt_bench", "wildbench", "bigcodebench"}
lb_pairs = [r for r in rrows if r["source_a"] in leaderboards or r["source_b"] in leaderboards]
print(f"  Leaderboard-involving pairs: {len(lb_pairs)}")
for r in lb_pairs:
    print(f"    {r['source_a']} vs {r['source_b']}: benchmark={r['benchmark']}, tau={r['tau']}, n={r['n_models']}")

# Check paper-paper pairs with lowest tau
print("\n=== LOWEST TAU PAPER PAIRS ===")
pp_pairs = sorted(rrows, key=lambda r: float(r["tau"]))[:10]
for r in pp_pairs:
    print(f"  {r['source_a']} vs {r['source_b']}: bench={r['benchmark']}, tau={r['tau']}, n={r['n_models']}")

# Global variance decomp
print("\n=== GLOBAL VARIANCE DECOMP ===")
with open("analysis_output/variance_decomp.csv") as f:
    vrows = list(csv.DictReader(f))
for r in vrows:
    print(f"  {r['predictor']}: coef={r['coef']}, p={r['pvalue']}, R2={r['partial_r2']}")

# Per-benchmark OLS
print("\n=== PER-BENCHMARK OLS ===")
with open("analysis_output/per_benchmark_ols.csv") as f:
    orows = list(csv.DictReader(f))
for r in orows:
    print(f"  {r['benchmark']:12s} {r['predictor']:25s} n={r['n']:3s} R2={r['partial_r2']:8s} f2={r['f2_cohen']:8s} p={r['p_value']:8s} beta={r['beta']}")

# Largest benchmark in collisions for violin mention
bc = Counter(r["benchmark"] for r in crows)
largest = bc.most_common(1)[0]
print(f"\n=== LARGEST BENCHMARK IN COLLISIONS ===")
print(f"  {largest[0]}: n={largest[1]}")
