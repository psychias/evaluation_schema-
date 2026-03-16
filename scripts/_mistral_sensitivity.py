#!/usr/bin/env python3
"""
Mistral-7B sensitivity analysis: run OLS with and without Mistral-7B-v0.1
to check robustness of harness R² estimates.

Reports:
  - Full dataset results (all 154 pairs)
  - Excluding Mistral-7B-v0.1 (154 - 37 = 117 pairs)
  - Comparison of partial R² and p-values
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def run_ols(df_bench: pd.DataFrame, label: str) -> dict:
    """Run OLS for |delta| ~ harness_differs + n_shot_diff + prompt_template_differs."""
    n = len(df_bench)
    if n < 5:
        return {"n": n, "label": label, "skip": True}

    y = df_bench["abs_delta"].values

    X_cols = []
    X_data = []

    # harness_differs
    h_diff = (df_bench["harness_a"] != df_bench["harness_b"]).astype(float).values
    X_cols.append("harness_differs")
    X_data.append(h_diff)

    # n_shot_diff
    try:
        ns_a = pd.to_numeric(df_bench["n_shot_a"], errors="coerce").fillna(0)
        ns_b = pd.to_numeric(df_bench["n_shot_b"], errors="coerce").fillna(0)
        n_diff = (ns_a - ns_b).abs().values
        X_cols.append("n_shot_diff")
        X_data.append(n_diff)
    except Exception:
        pass

    # prompt_template_differs
    pt_diff = (df_bench["prompt_template_a"] != df_bench["prompt_template_b"]).astype(float).values
    X_cols.append("prompt_tmpl_differs")
    X_data.append(pt_diff)

    X = np.column_stack(X_data)
    X = sm.add_constant(X)

    try:
        model_full = sm.OLS(y, X).fit()
        ss_res_full = np.sum(model_full.resid ** 2)
        ss_total = np.sum((y - y.mean()) ** 2)
    except Exception:
        return {"n": n, "label": label, "skip": True}

    results = {"n": n, "label": label, "skip": False, "R2_full": model_full.rsquared}

    # Drop-one partial R² for each predictor
    for j, col_name in enumerate(X_cols):
        drop_idx = j + 1  # +1 because column 0 is constant
        X_reduced = np.delete(X, drop_idx, axis=1)
        try:
            model_reduced = sm.OLS(y, X_reduced).fit()
            ss_res_reduced = np.sum(model_reduced.resid ** 2)
            partial_r2 = (ss_res_reduced - ss_res_full) / ss_total
            # F-test for dropped predictor
            df_num = 1
            df_den = n - X.shape[1]
            if df_den > 0 and ss_res_full > 0:
                f_stat = (ss_res_reduced - ss_res_full) / (ss_res_full / df_den)
                from scipy import stats
                p_val = 1 - stats.f.cdf(f_stat, df_num, df_den)
            else:
                f_stat, p_val = 0, 1.0
        except Exception:
            partial_r2, f_stat, p_val = 0, 0, 1.0

        results[f"{col_name}_R2p"] = partial_r2
        results[f"{col_name}_p"] = p_val

    return results


def main():
    csv_path = Path(_ROOT) / "analysis_output" / "collision_pairs.csv"
    df = pd.read_csv(csv_path)
    df["abs_delta"] = df["delta"].abs()

    mistral_mask = df["model_id"].str.contains("Mistral-7B-v0.1", case=False, na=False)
    n_mistral = mistral_mask.sum()
    print(f"Total collision pairs: {len(df)}")
    print(f"Mistral-7B-v0.1 pairs: {n_mistral}")
    print(f"Non-Mistral pairs: {len(df) - n_mistral}")
    print()

    benchmarks = sorted(df["benchmark"].unique())

    print(f"{'Benchmark':<18} {'Subset':<18} {'n':>4}  {'harness R²p':>12}  {'harness p':>10}  {'n_shot R²p':>11}  {'pt_diff R²p':>12}")
    print("-" * 100)

    for bench in benchmarks:
        df_bench_full = df[df["benchmark"] == bench]
        df_bench_no_m = df[(df["benchmark"] == bench) & ~mistral_mask]

        for subset_label, df_sub in [("Full", df_bench_full), ("No Mistral-7B", df_bench_no_m)]:
            r = run_ols(df_sub, subset_label)
            if r.get("skip"):
                print(f"{bench:<18} {subset_label:<18} {r['n']:>4}  {'(skipped: n<5)':>12}")
                continue

            hr2 = r.get("harness_differs_R2p", 0)
            hp = r.get("harness_differs_p", 1)
            nr2 = r.get("n_shot_diff_R2p", 0)
            pr2 = r.get("prompt_tmpl_differs_R2p", 0)
            print(f"{bench:<18} {subset_label:<18} {r['n']:>4}  {hr2:>12.4f}  {hp:>10.4f}  {nr2:>11.4f}  {pr2:>12.4f}")

        print()


if __name__ == "__main__":
    main()
