#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zigzag_claim_aligned_report.py

Reviewer-oriented reporting layer for CancerZigZag.

This script does NOT create another leaderboard.

It turns the existing claim-aligned analyses into reviewer-defensible tables:

A) Does ZigZag work?
   - tumor-like candidate discovery
   - selected candidate tumor scores

B) Is it seed-dependent?
   - true seed vs permuted seed distances
   - own-seed retrieval
   - within-seed vs between-seed cloud structure

C) Is it non-trivial?
   - not reducible to linear seed->tumor-mode controls
   - trajectory nonlinearity vs single-cycle editing

D) Is it biologically plausible?
   - DE/pathway concordance with held-out real tumor-vs-healthy reference

Required prior analyses
-----------------------
1) zigzag_fair_falsification_eval.py
   RUN_DIR/zigzag_falsification_eval/

2) zigzag_linear_control_analysis.py
   RUN_DIR/linear_control_analysis/

Optional
--------
You can pass multiple datasets:

  --dataset CRC:/path/to/crc_run
  --dataset BRCA:/path/to/brca_run
  --dataset LC:/path/to/lc_run
  --dataset RCC:/path/to/rcc_run

Outputs
-------
OUTDIR/claim_aligned_method_table.csv
OUTDIR/zigzag_claim_evidence_table.csv
OUTDIR/control_interpretation_table.csv
OUTDIR/reviewer_response_points.md
OUTDIR/figure_ready_long_table.csv
OUTDIR/claim_aligned_meta.json

Core philosophy
---------------
Controls are interpreted as controls, not as equivalent methods:
- r=1 tests whether multi-round ZigZag differs from single-cycle editing.
- Gaussian tests whether random local perturbation explains the result.
- tumor centroid tests whether global tumor-axis displacement explains it.
- tumor cluster centroid tests whether linear movement to explicit tumor modes explains it.

No result is called "reviewer proof" automatically. The script labels evidence as:
- supported
- mixed
- not supported
based on transparent thresholds that you may adjust.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def safe_read(path, required=False):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(path)
    return pd.DataFrame()


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def safe_get(row, key, default=np.nan):
    return safe_float(row[key]) if key in row.index else default


def status_bool(value, good_if="high", threshold=0.0):
    v = safe_float(value)
    if not np.isfinite(v):
        return None
    if good_if == "high":
        return bool(v >= threshold)
    return bool(v <= threshold)


def call_status(ok, mixed=None):
    if ok is True:
        return "supported"
    if ok is False:
        if mixed:
            return "mixed"
        return "not_supported"
    return "not_available"


def method_role(method):
    roles = {
        "zigzag_pool": "main_method",
        "single_cycle_pool": "direct_diffusion_ablation",
        "gaussian_matched_perturbation": "random_local_perturbation_control",
        "tumor_centroid_interpolation": "global_linear_tumor_axis_control",
        "tumor_cluster_centroid_interpolation": "explicit_tumor_mode_linear_control",
    }
    return roles.get(method, "other")


def method_question(method):
    questions = {
        "zigzag_pool": "Does multi-round tumor-trained diffusion editing discover tumor-like, seed-dependent candidates?",
        "single_cycle_pool": "Can a single noise-denoise editing step explain the effect?",
        "gaussian_matched_perturbation": "Can random local latent perturbation explain the effect?",
        "tumor_centroid_interpolation": "Can global linear movement toward the tumor centroid explain the effect?",
        "tumor_cluster_centroid_interpolation": "Can direct linear movement toward explicit tumor modes explain the effect?",
    }
    return questions.get(method, "Additional method/control.")


def format_num(x, digits=3):
    x = safe_float(x)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


# ============================================================
# Load and assemble one dataset
# ============================================================

def load_dataset_tables(dataset, run_dir):
    run_dir = Path(run_dir)

    fdir = run_dir / "zigzag_falsification_eval"
    ldir = run_dir / "linear_control_analysis"

    tables = {
        "selected_summary": safe_read(fdir / "01_selected_candidate_summary.csv", required=True),
        "seed_selected": safe_read(fdir / "02_seed_null_selected.csv", required=True),
        "clouds": safe_read(fdir / "03_seed_null_candidate_clouds.csv", required=True),
        "biology": safe_read(fdir / "06_biology_de_pathway.csv", required=True),
        "flags": safe_read(fdir / "07_interpretation_flags.csv", required=False),

        "line_selected": safe_read(ldir / "linear_explainability_summary_selected.csv", required=True),
        "success_diversity": safe_read(ldir / "success_diversity_summary.csv", required=False),
        "trajectory": safe_read(ldir / "trajectory_nonlinearity_summary.csv", required=False),
        "linear_flags": safe_read(ldir / "linear_control_interpretation_flags.csv", required=False),
    }

    for name, df in tables.items():
        if len(df):
            df.insert(0, "dataset", dataset)

    return tables


def assemble_method_table(dataset, run_dir):
    t = load_dataset_tables(dataset, run_dir)

    base = t["selected_summary"].copy()
    if "method" not in base.columns:
        raise RuntimeError(f"selected_summary for {dataset} has no method column")

    # Add seed selected null: residual expression only.
    seed_resid = t["seed_selected"]
    if len(seed_resid) and "space" in seed_resid.columns:
        seed_resid = seed_resid[seed_resid["space"].astype(str) == "resid_expr"].copy()
        keep = [
            "dataset", "method",
            "actual_over_perm_mean",
            "p_perm_mean_le_actual",
            "own_rank_mean",
            "own_top1",
            "own_top3",
        ]
        seed_resid = seed_resid[[c for c in keep if c in seed_resid.columns]]
        seed_resid = seed_resid.rename(columns={
            "actual_over_perm_mean": "selected_resid_seed_actual_over_perm",
            "p_perm_mean_le_actual": "selected_resid_seed_perm_p",
            "own_rank_mean": "selected_resid_own_rank_mean",
            "own_top1": "selected_resid_own_top1",
            "own_top3": "selected_resid_own_top3",
        })
        base = base.merge(seed_resid, on=["dataset", "method"], how="left")

    # Add cloud null: residual expression only.
    clouds = t["clouds"]
    if len(clouds) and "space" in clouds.columns:
        clouds = clouds[clouds["space"].astype(str) == "resid_expr"].copy()
        keep = [
            "dataset", "method",
            "within_over_between_mean",
            "within_over_between_median",
        ]
        clouds = clouds[[c for c in keep if c in clouds.columns]]
        clouds = clouds.rename(columns={
            "within_over_between_mean": "cloud_resid_within_over_between_mean",
            "within_over_between_median": "cloud_resid_within_over_between_median",
        })
        base = base.merge(clouds, on=["dataset", "method"], how="left")

    # Biology.
    bio = t["biology"]
    if len(bio):
        keep = [
            "dataset", "method",
            "de_pearson_all",
            "de_spearman_all",
            "de_top50_abs_overlap",
            "de_top100_abs_overlap",
            "de_top200_abs_overlap",
            "pathway_pearson",
            "pathway_spearman",
            "pathway_direction_agree_frac",
        ]
        bio = bio[[c for c in keep if c in bio.columns]]
        base = base.merge(bio, on=["dataset", "method"], how="left")

    # Linear explainability selected.
    line = t["line_selected"]
    if len(line):
        keep = [
            "dataset", "method",
            "mean_best_line_resid_norm",
            "median_best_line_resid_norm",
            "frac_linear_explainable_0p05",
            "frac_linear_explainable_0p10",
            "frac_linear_explainable_0p20",
            "mean_global_centroid_line_resid_norm",
        ]
        line = line[[c for c in keep if c in line.columns]]
        base = base.merge(line, on=["dataset", "method"], how="left")

    # Success diversity.
    div = t["success_diversity"]
    if len(div):
        keep = [
            "dataset", "method",
            "mean_n_success",
            "mean_success_frac",
            "mean_n_success_nearest_tumor_modes",
            "mean_success_mode_entropy",
            "mean_success_pairwise_latent_distance",
            "mean_success_best_line_resid_norm",
        ]
        div = div[[c for c in keep if c in div.columns]]
        base = base.merge(div, on=["dataset", "method"], how="left")

    # Trajectory.
    traj = t["trajectory"]
    if len(traj):
        keep = [
            "dataset", "method",
            "n_selected_with_trajectory",
            "mean_path_over_chord",
            "median_path_over_chord",
            "mean_max_line_deviation_norm",
            "mean_chord_length",
            "mean_path_length",
        ]
        traj = traj[[c for c in keep if c in traj.columns]]
        base = base.merge(traj, on=["dataset", "method"], how="left")

    base["method_role"] = base["method"].map(method_role)
    base["control_question"] = base["method"].map(method_question)

    return base


# ============================================================
# Evidence interpretation
# ============================================================

def evidence_for_zigzag(method_table, args):
    rows = []

    for dataset, df in method_table.groupby("dataset", sort=True):
        z = df[df["method"] == "zigzag_pool"]
        if len(z) == 0:
            rows.append({
                "dataset": dataset,
                "claim": "overall",
                "status": "not_available",
                "evidence": "No zigzag_pool row found.",
            })
            continue

        z = z.iloc[0]

        # A. Tumor-like discovery
        success = safe_get(z, "success_frac")
        fallback = safe_get(z, "fallback_frac")
        tumor_proba = safe_get(z, "mean_tumor_proba")
        discovery_ok = (
            np.isfinite(success) and success >= args.min_success_frac
            and (not np.isfinite(fallback) or fallback <= args.max_fallback_frac)
        )
        rows.append({
            "dataset": dataset,
            "claim": "A_tumor_like_candidate_discovery",
            "status": call_status(discovery_ok),
            "evidence": (
                f"success_frac={format_num(success)}, fallback_frac={format_num(fallback)}, "
                f"mean_tumor_proba={format_num(tumor_proba)}"
            ),
        })

        # B. Seed dependence selected
        seed_ratio = safe_get(z, "selected_resid_seed_actual_over_perm")
        seed_p = safe_get(z, "selected_resid_seed_perm_p")
        own_top3 = safe_get(z, "selected_resid_own_top3")
        seed_ok = (
            np.isfinite(seed_ratio) and seed_ratio < args.max_seed_perm_ratio
            and (not np.isfinite(seed_p) or seed_p <= args.max_perm_p)
        )
        rows.append({
            "dataset": dataset,
            "claim": "B_selected_candidates_retain_residual_seed_dependence",
            "status": call_status(seed_ok),
            "evidence": (
                f"resid actual/permuted={format_num(seed_ratio)}, "
                f"perm_p={format_num(seed_p)}, own_top3={format_num(own_top3)}"
            ),
        })

        # C. Seed-conditioned clouds
        cloud_ratio = safe_get(z, "cloud_resid_within_over_between_mean")
        cloud_ok = np.isfinite(cloud_ratio) and cloud_ratio < args.max_cloud_ratio
        rows.append({
            "dataset": dataset,
            "claim": "C_candidate_clouds_are_seed_conditioned",
            "status": call_status(cloud_ok),
            "evidence": f"resid cloud within/between={format_num(cloud_ratio)}",
        })

        # D. Not reducible to linear cluster centroid selected
        line_res = safe_get(z, "mean_best_line_resid_norm")
        linear_frac = safe_get(z, "frac_linear_explainable_0p10")
        nonlinear_ok = (
            np.isfinite(line_res) and line_res >= args.min_line_resid_norm
            and (not np.isfinite(linear_frac) or linear_frac <= args.max_linear_explainable_frac)
        )
        rows.append({
            "dataset": dataset,
            "claim": "D_not_reducible_to_seed_to_tumor_cluster_lines",
            "status": call_status(nonlinear_ok),
            "evidence": (
                f"mean line residual norm={format_num(line_res)}, "
                f"frac linear-explainable <=0.10={format_num(linear_frac)}"
            ),
        })

        # E. Trajectory nonlinearity
        path_ratio = safe_get(z, "mean_path_over_chord")
        traj_dev = safe_get(z, "mean_max_line_deviation_norm")
        traj_ok = np.isfinite(path_ratio) and path_ratio >= args.min_path_over_chord
        rows.append({
            "dataset": dataset,
            "claim": "E_multi_round_trajectory_is_non_linear",
            "status": call_status(traj_ok, mixed=True),
            "evidence": (
                f"path/chord={format_num(path_ratio)}, "
                f"max line deviation norm={format_num(traj_dev)}"
            ),
        })

        # F. Biology
        de = safe_get(z, "de_pearson_all")
        pw = safe_get(z, "pathway_direction_agree_frac")
        bio_ok = (
            (np.isfinite(de) and de >= args.min_de_pearson)
            or (np.isfinite(pw) and pw >= args.min_pathway_agreement)
        )
        rows.append({
            "dataset": dataset,
            "claim": "F_biological_program_concordance",
            "status": call_status(bio_ok, mixed=True),
            "evidence": (
                f"DE Pearson={format_num(de)}, "
                f"pathway direction agreement={format_num(pw)}"
            ),
        })

        # Direct control warning: does any control match all major axes?
        controls = df[df["method"] != "zigzag_pool"].copy()
        warning = "No direct control matched ZigZag across all checked axes."
        status = "supported"
        if len(controls):
            z_success = safe_get(z, "success_frac")
            z_seed = safe_get(z, "selected_resid_seed_actual_over_perm")
            z_bio = safe_get(z, "de_pearson_all")
            z_line = safe_get(z, "mean_best_line_resid_norm")

            matched = []
            for _, c in controls.iterrows():
                c_success = safe_get(c, "success_frac")
                c_seed = safe_get(c, "selected_resid_seed_actual_over_perm")
                c_bio = safe_get(c, "de_pearson_all")
                c_line = safe_get(c, "mean_best_line_resid_norm")

                # A control is worrying if it is at least as successful,
                # at least as seed-dependent, at least as biologically concordant,
                # and no more linear-explained than ZigZag.
                if (
                    np.isfinite(c_success) and np.isfinite(z_success) and c_success >= z_success
                    and np.isfinite(c_seed) and np.isfinite(z_seed) and c_seed <= z_seed
                    and np.isfinite(c_bio) and np.isfinite(z_bio) and c_bio >= z_bio
                    and np.isfinite(c_line) and np.isfinite(z_line) and c_line >= z_line
                ):
                    matched.append(str(c["method"]))

            if matched:
                status = "mixed"
                warning = "Controls matching or exceeding ZigZag across major axes: " + ", ".join(matched)

        rows.append({
            "dataset": dataset,
            "claim": "G_controls_do_not_fully_explain_zigzag",
            "status": status,
            "evidence": warning,
        })

    return pd.DataFrame(rows)


def control_interpretation_table(method_table):
    rows = []
    for dataset, df in method_table.groupby("dataset", sort=True):
        for _, r in df.iterrows():
            method = r["method"]
            role = method_role(method)

            if method == "zigzag_pool":
                interpretation = (
                    "Main method. Evaluate by tumor discovery, residual seed dependence, "
                    "non-reducibility to linear controls, trajectory structure, and biology."
                )
            elif method == "single_cycle_pool":
                interpretation = (
                    "Direct ablation. If it matches ZigZag, the need for multi-round ZigZagging is weakened. "
                    "If ZigZag has more nonlinear trajectories or stronger cloud structure, multi-round adds exploration."
                )
            elif method == "gaussian_matched_perturbation":
                interpretation = (
                    "Random perturbation control. Strong tumor scores here suggest the scorer/selection can be reached "
                    "by local noise; biology and seed-cloud structure become key."
                )
            elif method == "tumor_centroid_interpolation":
                interpretation = (
                    "Global linear tumor-axis control. High tumor scores are expected and do not imply generative equivalence."
                )
            elif method == "tumor_cluster_centroid_interpolation":
                interpretation = (
                    "Strong linear tumor-mode control. It uses explicit tumor-cluster centroids, so it should not be framed "
                    "as an equivalent generative baseline. Use it to test whether ZigZag is reducible to linear tumor-mode movement."
                )
            else:
                interpretation = "Additional method/control."

            rows.append({
                "dataset": dataset,
                "method": method,
                "role": role,
                "control_question": method_question(method),
                "interpretation": interpretation,
            })

    return pd.DataFrame(rows)


def make_markdown_report(evidence, control_table, method_table, out_path):
    lines = []
    lines.append("# CancerZigZag claim-aligned evaluation report\n")
    lines.append("This report is designed for manuscript/reviewer framing. It avoids interpreting controls as a simple leaderboard.\n")

    lines.append("## Core framing\n")
    lines.append(
        "CancerZigZag is evaluated as a seed-initialized stochastic candidate-discovery framework. "
        "The relevant questions are whether it discovers tumor-like candidates, retains residual seed dependence, "
        "and is not reducible to single-cycle editing or linear movement toward tumor modes.\n"
    )

    lines.append("## Evidence by dataset\n")
    for dataset, df in evidence.groupby("dataset", sort=True):
        lines.append(f"### {dataset}\n")
        for _, r in df.iterrows():
            lines.append(f"- **{r['claim']}**: `{r['status']}` — {r['evidence']}")
        lines.append("")

    lines.append("## How to describe controls\n")
    seen = set()
    for _, r in control_table.iterrows():
        key = r["method"]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"### {r['method']}\n")
        lines.append(f"Role: `{r['role']}`\n")
        lines.append(f"Question: {r['control_question']}\n")
        lines.append(f"Interpretation: {r['interpretation']}\n")

    lines.append("## Manuscript-ready wording\n")
    lines.append(
        "We did not treat real tumor retrieval or direct tumor-mode interpolation as equivalent generative baselines, "
        "because they do not solve the seed-initialized stochastic candidate-discovery task. Instead, we used controls "
        "as falsification tests. Single-cycle editing tests whether multi-round ZigZagging reduces to one denoising pass; "
        "Gaussian perturbation tests whether local random variation is sufficient; and tumor-centroid or tumor-cluster "
        "interpolation tests whether observed tumor-likeness is explainable by direct linear movement toward tumor modes. "
        "CancerZigZag is therefore assessed by tumor-like candidate discovery, residual seed dependence relative to "
        "permutation nulls, candidate-cloud structure, trajectory nonlinearity, and biological concordance.\n"
    )

    lines.append("## Caveat\n")
    lines.append(
        "This framework is more reviewer-defensible than a leaderboard, but it is not automatically reviewer-proof. "
        "If a simple control matches ZigZag across tumor discovery, seed dependence, nonlinearity, and biology, "
        "the strong ZigZag claim should be reduced.\n"
    )

    Path(out_path).write_text("\n".join(lines))


# ============================================================
# Main
# ============================================================

def parse_dataset_arg(s):
    if ":" not in s:
        raise argparse.ArgumentTypeError("Dataset must be NAME:/path/to/run_dir")
    name, path = s.split(":", 1)
    return name, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        type=parse_dataset_arg,
        help="One or more NAME:/path/to/run_dir entries."
    )
    p.add_argument("--outdir", required=True)

    # Evidence thresholds. These are transparent, not magic.
    p.add_argument("--min_success_frac", type=float, default=0.8)
    p.add_argument("--max_fallback_frac", type=float, default=0.3)
    p.add_argument("--max_seed_perm_ratio", type=float, default=0.95)
    p.add_argument("--max_perm_p", type=float, default=0.05)
    p.add_argument("--max_cloud_ratio", type=float, default=0.95)
    p.add_argument("--min_line_resid_norm", type=float, default=0.2)
    p.add_argument("--max_linear_explainable_frac", type=float, default=0.5)
    p.add_argument("--min_path_over_chord", type=float, default=1.2)
    p.add_argument("--min_de_pearson", type=float, default=0.25)
    p.add_argument("--min_pathway_agreement", type=float, default=0.6)

    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    method_tables = []
    for name, run_dir in args.dataset:
        print(f"[load] {name}: {run_dir}")
        mt = assemble_method_table(name, run_dir)
        method_tables.append(mt)

    method_table = pd.concat(method_tables, ignore_index=True)
    evidence = evidence_for_zigzag(method_table, args)
    control_table = control_interpretation_table(method_table)

    # Long table for plotting.
    fig_long = method_table.copy()

    paths = {
        "method_table": outdir / "claim_aligned_method_table.csv",
        "evidence": outdir / "zigzag_claim_evidence_table.csv",
        "control_table": outdir / "control_interpretation_table.csv",
        "fig_long": outdir / "figure_ready_long_table.csv",
        "report": outdir / "reviewer_response_points.md",
        "meta": outdir / "claim_aligned_meta.json",
    }

    method_table.to_csv(paths["method_table"], index=False)
    evidence.to_csv(paths["evidence"], index=False)
    control_table.to_csv(paths["control_table"], index=False)
    fig_long.to_csv(paths["fig_long"], index=False)
    make_markdown_report(evidence, control_table, method_table, paths["report"])

    meta = {
        "purpose": "claim-aligned reviewer-oriented report, not a leaderboard",
        "datasets": [{"name": n, "run_dir": p} for n, p in args.dataset],
        "thresholds": {
            "min_success_frac": args.min_success_frac,
            "max_fallback_frac": args.max_fallback_frac,
            "max_seed_perm_ratio": args.max_seed_perm_ratio,
            "max_perm_p": args.max_perm_p,
            "max_cloud_ratio": args.max_cloud_ratio,
            "min_line_resid_norm": args.min_line_resid_norm,
            "max_linear_explainable_frac": args.max_linear_explainable_frac,
            "min_path_over_chord": args.min_path_over_chord,
            "min_de_pearson": args.min_de_pearson,
            "min_pathway_agreement": args.min_pathway_agreement,
        },
        "not_reviewer_proof": (
            "No analysis is automatically reviewer-proof. This makes the comparison conceptually defensible "
            "by matching controls to falsification questions."
        ),
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    print("\n================ DONE ================")
    for k, v in paths.items():
        print(f"{k}: {v}")

    print("\n[evidence preview]")
    print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
