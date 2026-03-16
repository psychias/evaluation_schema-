#!/usr/bin/env python3
"""Compute BH-FDR correction for all per-benchmark OLS p-values."""
import csv

with open("analysis_output/per_benchmark_ols.csv") as f:
    rows = list(csv.DictReader(f))

pvals = []
for r in rows:
    pvals.append({
        "bench": r["benchmark"],
        "pred": r["predictor"],
        "n": int(r["n"]),
        "r2p": float(r["partial_r2"]),
        "f2": float(r["f2_cohen"]),
        "p": float(r["p_value"]),
        "beta": float(r["beta"]),
        "full_r2": float(r["full_model_r2"]),
    })

pvals.sort(key=lambda x: x["p"])
m = len(pvals)

# Step-up BH-FDR
qvals = [0.0] * m
qvals[m - 1] = pvals[m - 1]["p"]
for i in range(m - 2, -1, -1):
    qvals[i] = min(pvals[i]["p"] * m / (i + 1), qvals[i + 1])

print(f"Total tests: {m}")
print()
for i, pv in enumerate(pvals):
    sig = " ***" if qvals[i] < 0.001 else (" **" if qvals[i] < 0.01 else (" *" if qvals[i] < 0.05 else ""))
    print(f"  {pv['bench']:15s} {pv['pred']:28s} n={pv['n']:2d} R2p={pv['r2p']:.4f} f2={pv['f2']:.4f} p={pv['p']:.4f} q_BH={qvals[i]:.4f}{sig}")

# Show key results
print()
print("=== Key nominal results (p < 0.10) ===")
for i, pv in enumerate(pvals):
    if pv["p"] < 0.10:
        print(f"  {pv['bench']:15s} {pv['pred']:28s} p={pv['p']:.4f} q_adj={qvals[i]:.4f}")
