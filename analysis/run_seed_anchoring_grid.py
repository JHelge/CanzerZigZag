#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run selected-candidate seed anchoring analysis for all grid runs.

This script loops over folders like:
  r1_t1_s0.0
  r1_t10_s0.0
  r10_t75_s0.0
  ...

For each run, it calls selected_seed_anchoring.py and then aggregates all
selected_seed_anchoring_summary.csv files into one grid-level table.

Use this after the r x t grid has finished.
"""

import os
import re
import glob
import argparse
import subprocess
import pandas as pd


def parse_r_t_from_run_dir(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.match(r"^r(\d+)_t(\d+)_s(.+)$", name)
    if m is None:
        return None
    return {
        "r": int(m.group(1)),
        "t": int(m.group(2)),
        "s": m.group(3),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid_dir",
        required=True,
        help="Directory containing r*_t*_s0.0 run folders, e.g. .../multi_healthy2tumor",
    )
    parser.add_argument(
        "--h5ad_seed",
        required=True,
        help="Aligned TEST h5ad used as seed source for this dataset.",
    )
    parser.add_argument(
        "--vae_ckpt",
        required=True,
    )
    parser.add_argument(
        "--seed_anchoring_script",
        default="/prj/ml-ident-canc/original_codes/selected_seed_anchoring.py",
    )
    parser.add_argument(
        "--repo_dir",
        default="/prj/ml-ident-canc/original_codes/scDiffusion",
    )
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--selection_rule", default="proba_ge_0.7_min_identity")
    parser.add_argument("--label_col", default="status")
    parser.add_argument("--healthy_label", default="healthy")
    parser.add_argument("--tumor_label", default="tumor")
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--n_random_tumor", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if selected_seed_anchoring_summary.csv already exists.",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional output CSV. Default: grid_dir/selected_seed_anchoring_grid_summary.csv",
    )

    args = parser.parse_args()

    run_dirs = sorted(glob.glob(os.path.join(args.grid_dir, "r*_t*_s*")))

    parsed = []
    for rd in run_dirs:
        p = parse_r_t_from_run_dir(rd)
        if p is not None:
            parsed.append((rd, p))

    if len(parsed) == 0:
        raise RuntimeError(f"No run dirs found in {args.grid_dir}")

    print(f"[grid] found {len(parsed)} run folders")

    for run_dir, info in parsed:
        summary_path = os.path.join(run_dir, "selected_seed_anchoring_summary.csv")

        if os.path.exists(summary_path) and not args.force:
            print(f"[skip existing] r={info['r']} t={info['t']} | {summary_path}")
            continue

        # Require selected candidate files.
        eval_path = os.path.join(run_dir, "selected_candidates_eval.csv")
        expr_path = os.path.join(run_dir, "selected_candidates_expr.h5ad")

        if not os.path.exists(eval_path):
            print(f"[skip missing selected eval] {run_dir}")
            continue

        if not os.path.exists(expr_path):
            print(f"[skip missing selected expr] {run_dir}")
            continue

        cmd = [
            "python3", args.seed_anchoring_script,
            "--run_dir", run_dir,
            "--h5ad_seed", args.h5ad_seed,
            "--vae_ckpt", args.vae_ckpt,
            "--latent_dim", str(args.latent_dim),
            "--batch", str(args.batch),
            "--repo_dir", args.repo_dir,
            "--selection_rule", args.selection_rule,
            "--label_col", args.label_col,
            "--healthy_label", args.healthy_label,
            "--tumor_label", args.tumor_label,
            "--n_perm", str(args.n_perm),
            "--n_random_tumor", str(args.n_random_tumor),
            "--seed", str(args.seed),
        ]

        print("\n[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # Aggregate summaries.
    rows = []

    for run_dir, info in parsed:
        summary_path = os.path.join(run_dir, "selected_seed_anchoring_summary.csv")

        if not os.path.exists(summary_path):
            continue

        df = pd.read_csv(summary_path)
        if len(df) == 0:
            continue

        row = df.iloc[0].to_dict()
        row["run_dir"] = run_dir
        row["r"] = info["r"]
        row["t"] = info["t"]
        row["s"] = info["s"]

        # Also pull core selected-candidate performance if available.
        eval_path = os.path.join(run_dir, "selected_candidates_eval.csv")
        if os.path.exists(eval_path):
            sel = pd.read_csv(eval_path)
            if "selection_rule" in sel.columns:
                sel = sel[sel["selection_rule"].astype(str) == args.selection_rule].copy()

            if len(sel) > 0:
                row["grid_n_selected"] = len(sel)

                if "tumor_proba" in sel.columns:
                    row["grid_frac_proba_ge_0p7"] = float((sel["tumor_proba"] >= 0.7).mean())
                    row["grid_mean_tumor_proba"] = float(sel["tumor_proba"].mean())
                    row["grid_median_tumor_proba"] = float(sel["tumor_proba"].median())

                if "tumor_logit" in sel.columns:
                    row["grid_mean_tumor_logit"] = float(sel["tumor_logit"].mean())
                    row["grid_median_tumor_logit"] = float(sel["tumor_logit"].median())

                if "id_dist_expr" in sel.columns:
                    row["grid_mean_id_dist_expr"] = float(sel["id_dist_expr"].mean())
                    row["grid_median_id_dist_expr"] = float(sel["id_dist_expr"].median())

                if "progress01" in sel.columns:
                    row["grid_mean_progress01"] = float(sel["progress01"].mean())
                    row["grid_median_progress01"] = float(sel["progress01"].median())

                if "used_fallback" in sel.columns:
                    row["grid_fallback_frac"] = float(sel["used_fallback"].astype(bool).mean())
                elif "fallback" in sel.columns:
                    row["grid_fallback_frac"] = float(sel["fallback"].astype(bool).mean())

        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No selected_seed_anchoring_summary.csv files found after running.")

    out = pd.DataFrame(rows)

    # Ranking helper:
    # Primary: high success, low fallback, low actual_over_perm, low id distance, high tumor proba.
    sort_cols = []
    ascending = []

    for c, asc in [
        ("grid_frac_proba_ge_0p7", False),
        ("grid_fallback_frac", True),
        ("actual_over_perm_mean_dist", True),
        ("p_perm_mean_le_actual", True),
        ("grid_mean_id_dist_expr", True),
        ("grid_mean_tumor_proba", False),
    ]:
        if c in out.columns:
            sort_cols.append(c)
            ascending.append(asc)

    if sort_cols:
        out_ranked = out.sort_values(sort_cols, ascending=ascending)
    else:
        out_ranked = out.sort_values(["r", "t"])

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(args.grid_dir, "selected_seed_anchoring_grid_summary.csv")

    out_ranked.to_csv(out_csv, index=False)

    print("\n[DONE]")
    print("Wrote:", out_csv)
    print("\nTop 10 ranked runs:")
    cols_show = [
        "r", "t",
        "grid_frac_proba_ge_0p7",
        "grid_fallback_frac",
        "grid_mean_tumor_proba",
        "grid_mean_id_dist_expr",
        "actual_over_perm_mean_dist",
        "p_perm_mean_le_actual",
        "actual_over_random_real_tumor_dist",
        "mean_own_rank_among_selected_seeds",
        "frac_own_seed_top3_selected_seeds",
    ]
    cols_show = [c for c in cols_show if c in out_ranked.columns]
    print(out_ranked[cols_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()