#!/usr/bin/env python3
"""Compute collision pair breakdown for paper update."""
import csv
from collections import Counter

with open("analysis_output/collision_pairs.csv") as f:
    rows = list(csv.DictReader(f))

print(f"Total collision pairs: {len(rows)}")
print()

bench_counts = Counter(r["benchmark"] for r in rows)
print("Per benchmark:")
for b, c in sorted(bench_counts.items(), key=lambda x: -x[1]):
    subs = [abs(float(r["delta"])) for r in rows if r["benchmark"] == b]
    sig = sum(1 for d in subs if d > 0.01)
    med = sorted(subs)[len(subs) // 2]
    print(f"  {b:20s} n={c:3d}  med|d|={med:.3f}  |d|>0.01: {sig}")

print()
sig_total = sum(1 for r in rows if abs(float(r["delta"])) > 0.01)
print(f"Total with |delta| > 0.01: {sig_total}")
print(f"Unique benchmarks: {len(bench_counts)}")

source_pairs = set()
for r in rows:
    pair = tuple(sorted([r["source_a"], r["source_b"]]))
    source_pairs.add(pair)
print(f"Unique source pairs: {len(source_pairs)}")

models = set(r["model_id"] for r in rows)
print(f"Unique models in collisions: {len(models)}")
for m in sorted(models):
    c = sum(1 for r in rows if r["model_id"] == m)
    print(f"  {m}: {c} pairs")
