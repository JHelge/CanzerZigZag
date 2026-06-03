#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items", nargs="+", required=True, help="Items of form DATASET=/path/to/final/run_dir")
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()
    rows = []
    for item in args.items:
        if "=" not in item:
            raise ValueError(f"Item must be DATASET=/path: {item}")
        dataset, run_dir = item.split("=", 1)
        path = os.path.join(run_dir, "baseline_comparison_summary.csv")
        if not os.path.exists(path):
            print(f"[skip missing] {path}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "dataset", dataset)
        rows.append(df)
    if not rows:
        raise RuntimeError("No baseline summary files found.")
    out = pd.concat(rows, ignore_index=True)
    preferred = [
        "dataset", "method",
        "success_frac_proba_ge_threshold", "fallback_frac",
        "mean_tumor_proba", "mean_tumor_logit",
        "mean_id_dist_expr", "mean_resid_id_dist_expr",
        "resid_expr_actual_over_perm_mean", "resid_expr_p_perm_mean_le_actual",
        "resid_latent_actual_over_perm_mean", "raw_expr_actual_over_perm_mean",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
