#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_baseline_pools_with_pipeline.py

Purpose
-------
Evaluate previously generated seed-initialized baseline candidate pools with the
SAME downstream ZigZag posthoc/evaluation pipeline.

This script does not invent a new tumor score or a new selection rule.

It converts baseline_pools/baseline_pools_latent.npz into one pseudo-run
directory per method:

    RUN_DIR/baseline_eval_runs/<method>/

Each pseudo-run receives:
    - trajectories_latent_all_reps.npz
      with Z_traj shaped as (n_candidates, 2, latent_dim):
          round 0 = Z_seed
          final   = Z_final
    - baseline_method_meta.csv
    - symlinks/copies of reference/evaluation files from the original RUN_DIR

Then it optionally runs:
    python posthoc.py --outdir <pseudo_run_dir>

This is the safest way to avoid unfair baseline comparisons:
    same generated-candidate format
    same posthoc evaluation code
    same tumor_proba classifier if posthoc uses the original pipeline resources
    same selection rule if posthoc implements selected-candidate evaluation

Expected workflow
-----------------
1) Generate pools:
   baseline_candidate_discovery.py  -> RUN_DIR/baseline_pools/

2) Evaluate pools through existing posthoc:
   evaluate_baseline_pools_with_pipeline.py --run_dir RUN_DIR --run_posthoc

3) Collect outputs:
   evaluate_baseline_pools_with_pipeline.py --run_dir RUN_DIR --collect_only

Outputs
-------
RUN_DIR/baseline_eval_runs/<method>/...
RUN_DIR/baseline_eval_runs/baseline_method_eval_summary.csv
RUN_DIR/baseline_eval_runs/baseline_selected_candidates_all_methods.csv

Notes
-----
If your existing posthoc.py expects specific run metadata files, this script
symlinks/copies common files from RUN_DIR. If posthoc still errors, send the
error and we adjust the list of required files.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_FILES_TO_LINK = [
    # Required reference files for posthoc. These may live in RUN_DIR or parent dirs.
    "real_h_eval.h5ad",
    "real_t_eval.h5ad",

    # Reference/config files only. Do NOT link result/output files from original ZigZag run.
    "args.json",
    "config.json",
    "gene_order.tsv",
    "gene_order.txt",
    "var_names.tsv",
    "var_names.txt",
    "eval_refs.npz",
    "eval_reference.npz",
    "reference_eval.npz",
    "progress_axis.npz",
    "pca_eval.npz",
    "seed_eval_refs.npz",
    "train_eval_refs.npz",
    "test_eval_refs.npz",
    "eval_data.npz",
    "eval_context.npz",
]
COMMON_GLOBS_TO_LINK = []

def safe_link_or_copy(src: Path, dst: Path, mode: str = "symlink"):
    if not src.exists():
        return False
    if dst.exists() or dst.is_symlink():
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(str(src), str(dst))
    return True


def read_pools(run_dir: Path):
    pool_dir = run_dir / "baseline_pools"
    npz_path = pool_dir / "baseline_pools_latent.npz"
    meta_path = pool_dir / "baseline_pools_meta.csv"
    manifest_path = pool_dir / "baseline_pools_manifest.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run baseline_candidate_discovery.py first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run baseline_candidate_discovery.py first.")

    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)

    required = ["Z_seed", "Z_final", "method", "seed_order", "cell_idx", "rep_id"]
    for k in required:
        if k not in z and k not in meta.columns:
            raise KeyError(f"Missing {k} in baseline pool npz/meta")

    Z_seed = z["Z_seed"].astype(np.float32)
    Z_final = z["Z_final"].astype(np.float32)

    if len(meta) != Z_seed.shape[0]:
        raise ValueError(f"meta rows ({len(meta)}) != Z_seed rows ({Z_seed.shape[0]})")
    if Z_final.shape != Z_seed.shape:
        raise ValueError(f"Z_final shape {Z_final.shape} != Z_seed shape {Z_seed.shape}")

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    return Z_seed, Z_final, meta, manifest


def link_common_files(original_run_dir: Path, method_dir: Path, link_mode: str):
    """
    Link only reference/config files, never original result files.

    Search order:
      1. original final run dir, e.g. .../status/multi_healthy2tumor/r10_t75_s0.0
      2. parent, e.g. .../status/multi_healthy2tumor
      3. grandparent, e.g. .../status
      4. great-grandparent, e.g. .../<OUT_ROOT>
    This is needed because real_h_eval.h5ad/real_t_eval.h5ad may live in .../status/.
    """
    linked = []

    forbidden_outputs = {
        "selected_candidates_eval.csv",
        "selected_candidates_expr.h5ad",
        "decoded_subset.h5ad",
        "per_seed_best_of_k.csv",
        "run_eval.json",
        "summary.csv",
        "summary_all_runs.csv",
        "best_of_k_summary.json",
        "selected_candidates_summary.json",
        "selected_candidates_de_summary_proba_ge_0p7_min_identity.json",
        "selected_trajectory_summary_proba_ge_0p7_min_identity.json",
        "residual_seed_identity_summary.json",
        "direction_consistency_summary.json",
        "bootstrap_ci_decoded.json",
        "posthoc_baseline_eval.log",
    }

    search_dirs = [
        original_run_dir,
        original_run_dir.parent,
        original_run_dir.parent.parent,
        original_run_dir.parent.parent.parent,
    ]

    for name in COMMON_FILES_TO_LINK:
        if name in forbidden_outputs:
            continue

        for base in search_dirs:
            src = base / name
            dst = method_dir / name
            if safe_link_or_copy(src, dst, link_mode):
                linked.append(f"{name} <- {src}")
                break

    return linked

def find_file_in_parents(original_run_dir: Path, filename: str):
    for base in [
        original_run_dir,
        original_run_dir.parent,
        original_run_dir.parent.parent,
        original_run_dir.parent.parent.parent,
    ]:
        p = base / filename
        if p.exists():
            return p
    return None


def make_pseudo_summary_all_runs(original_run_dir: Path, method_dir: Path, method: str, meta_method: pd.DataFrame):
    """
    posthoc.py expects summary_all_runs.csv as a marker that the pipeline ran once.
    We must NOT symlink the original summary_all_runs.csv because that would make
    all baseline methods inherit original ZigZag results.

    This function creates a method-specific pseudo summary. If an original
    summary_all_runs.csv exists, use its columns as a template but overwrite
    path/method information and remove obvious metric values. If no template
    exists, create a minimal one-row file.
    """
    template_path = find_file_in_parents(original_run_dir, "summary_all_runs.csv")

    if template_path is not None:
        try:
            tmpl = pd.read_csv(template_path)
            if len(tmpl) > 0:
                row = tmpl.iloc[[0]].copy()
            else:
                row = pd.DataFrame([{}])

            # Overwrite path-like columns.
            for col in row.columns:
                low = col.lower()
                if low in {"outdir", "out_dir", "run_dir", "path"} or "outdir" in low or "run_dir" in low:
                    row.loc[:, col] = str(method_dir)

            # Mark baseline method where possible.
            for col in row.columns:
                low = col.lower()
                if low in {"method", "baseline_method"}:
                    row.loc[:, col] = method

            # Remove obvious evaluation metric values from copied template.
            metric_tokens = [
                "tumor", "proba", "logit", "progress", "id_dist", "knn",
                "success", "fallback", "mean", "median", "score", "ratio",
                "pearson", "spearman", "pathway", "hallmark", "de_",
            ]
            keep_tokens = [
                "outdir", "out_dir", "run_dir", "path", "mode", "direction",
                "label", "seed", "guidance", "eta", "multi_pass", "round",
                "t_grid", "r_grid", "n_reps", "batch", "latent", "method",
            ]
            for col in row.columns:
                low = col.lower()
                if any(k in low for k in keep_tokens):
                    continue
                if any(t in low for t in metric_tokens):
                    row.loc[:, col] = np.nan

        except Exception as e:
            print(f"[warning] could not use summary_all_runs template {template_path}: {e}")
            row = pd.DataFrame([{}])
    else:
        row = pd.DataFrame([{}])

    # Ensure minimal useful columns.
    row.loc[:, "baseline_method"] = method
    row.loc[:, "outdir"] = str(method_dir)
    row.loc[:, "run_dir"] = str(method_dir)
    row.loc[:, "n_candidates"] = int(len(meta_method))
    row.loc[:, "n_seeds"] = int(meta_method["seed_order"].nunique())

    # Try to provide r/t-ish fields for posthoc code that expects them.
    row.loc[:, "mode"] = "baseline_pool"
    row.loc[:, "direction"] = "healthy2tumor"

    out = method_dir / "summary_all_runs.csv"
    row.to_csv(out, index=False)
    return out


def write_method_run(original_run_dir: Path, out_root: Path, method: str, Z_seed, Z_final, meta_method, link_mode: str):
    method_dir = out_root / method

    # Critical: remove stale pseudo-run directory.
    # Otherwise old symlinked selected_candidates_eval.csv or previous posthoc outputs
    # can make all baselines look identical.
    if method_dir.exists():
        shutil.rmtree(method_dir)
    method_dir.mkdir(parents=True, exist_ok=True)

    Z_traj = np.stack([Z_seed, Z_final], axis=1).astype(np.float32)

    # Make rep_id unique within seed if needed; keep original in meta too.
    cell_idx = meta_method["cell_idx"].astype(int).values
    rep_id = meta_method["rep_id"].astype(int).values
    seed_order = meta_method["seed_order"].astype(int).values

    np.savez_compressed(
        method_dir / "trajectories_latent_all_reps.npz",
        Z_traj=Z_traj,
        cell_idx=cell_idx,
        rep_id=rep_id,
        seed_order=seed_order,
        method=np.array([method] * len(meta_method), dtype=object),
    )

    meta_method.to_csv(method_dir / "baseline_method_meta.csv", index=False)

    # Create a fresh method-specific summary_all_runs.csv required by posthoc.
    # This must be created, not symlinked from the original run.
    summary_path = make_pseudo_summary_all_runs(original_run_dir, method_dir, method, meta_method)

    linked = link_common_files(original_run_dir, method_dir, link_mode)

    manifest = {
        "method": method,
        "n_candidates": int(len(meta_method)),
        "n_seeds": int(meta_method["seed_order"].nunique()),
        "source_run_dir": str(original_run_dir),
        "pseudo_run_dir": str(method_dir),
        "linked_files": linked,
        "pseudo_summary_all_runs": str(summary_path),
        "note": "Pseudo-run created from baseline pool. Evaluate with same posthoc pipeline as ZigZag.",
    }
    with open(method_dir / "baseline_method_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return method_dir


def run_posthoc_for_method(method_dir: Path, repo_dir: Path, posthoc_script: str, extra_args: list):
    script = Path(posthoc_script)
    if not script.is_absolute():
        script = repo_dir / posthoc_script

    if not script.exists():
        raise FileNotFoundError(f"Cannot find posthoc script: {script}")

    cmd = [sys.executable, "-u", str(script), "--outdir", str(method_dir)] + extra_args
    print("[run]", " ".join(cmd), flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

    res = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_path = method_dir / "posthoc_baseline_eval.log"
    log_path.write_text(res.stdout)

    if res.returncode != 0:
        print(res.stdout)
        print(f"[warning] posthoc failed for {method_dir}. See {log_path}")
        return log_path

    return log_path


def collect_outputs(out_root: Path, selection_rule: str):
    rows = []
    selected_frames = []

    for method_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        method = method_dir.name

        # collect selected_candidates_eval.csv if present
        sel_path = method_dir / "selected_candidates_eval.csv"
        if sel_path.exists():
            try:
                df = pd.read_csv(sel_path)
                df.insert(0, "baseline_method", method)
                if selection_rule and "selection_rule" in df.columns:
                    # keep all in full file, but summary can use selection rule
                    pass
                selected_frames.append(df)
            except Exception as e:
                print(f"[warning] could not read {sel_path}: {e}")

        # summarize useful csv/json files
        summary = {"baseline_method": method, "method_dir": str(method_dir)}

        for fname in [
            "run_eval.json",
            "selected_trajectory_final_sanity_proba_ge_0p7_min_identity.csv",
                    "baseline_method_meta.csv",
        ]:
            p = method_dir / fname
            summary[f"has_{fname}"] = bool(p.exists())

        # If selected_candidates_eval.csv exists, compute simple summary.
        if sel_path.exists():
            try:
                df = pd.read_csv(sel_path)
                if selection_rule and "selection_rule" in df.columns:
                    dfr = df[df["selection_rule"].astype(str) == selection_rule].copy()
                    if len(dfr) == 0:
                        dfr = df.copy()
                else:
                    dfr = df.copy()

                summary["n_selected_rows"] = int(len(dfr))

                for c in ["tumor_proba", "tumor_logit", "progress01", "id_dist_expr", "resid_id_dist_expr"]:
                    if c in dfr.columns:
                        summary[f"mean_{c}"] = float(pd.to_numeric(dfr[c], errors="coerce").mean())
                        summary[f"median_{c}"] = float(pd.to_numeric(dfr[c], errors="coerce").median())

                if "tumor_proba" in dfr.columns:
                    proba = pd.to_numeric(dfr["tumor_proba"], errors="coerce")
                    summary["frac_tumor_proba_ge_0p7"] = float((proba >= 0.7).mean())

                if "used_fallback" in dfr.columns:
                    summary["fallback_frac"] = float(pd.to_numeric(dfr["used_fallback"], errors="coerce").mean())
                elif "fallback" in dfr.columns:
                    summary["fallback_frac"] = float(pd.to_numeric(dfr["fallback"], errors="coerce").mean())

            except Exception as e:
                summary["selected_summary_error"] = str(e)

        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    summary_path = out_root / "baseline_method_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if selected_frames:
        all_sel = pd.concat(selected_frames, ignore_index=True)
        all_sel_path = out_root / "baseline_selected_candidates_all_methods.csv"
        all_sel.to_csv(all_sel_path, index=False)
    else:
        all_sel_path = None

    return summary_path, all_sel_path


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--run_dir", required=True,
                   help="Original final ZigZag run dir containing baseline_pools/")
    p.add_argument("--repo_dir", default="/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat")
    p.add_argument("--out_root", default=None,
                   help="Default: RUN_DIR/baseline_eval_runs")

    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods to convert/evaluate. Default: all methods in baseline_pools_meta.csv")

    p.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--run_posthoc", action="store_true")
    p.add_argument("--collect_only", action="store_true")
    p.add_argument("--posthoc_script", default="posthoc.py")
    p.add_argument("--posthoc_extra_args", nargs=argparse.REMAINDER, default=[])

    p.add_argument("--selection_rule", default="proba_ge_0.7_min_identity")

    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else run_dir / "baseline_eval_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        summary_path, all_sel_path = collect_outputs(out_root, args.selection_rule)
        print("Collected summary:", summary_path)
        if all_sel_path:
            print("Collected selected:", all_sel_path)
        return

    Z_seed_all, Z_final_all, meta, manifest = read_pools(run_dir)

    methods = args.methods or sorted(meta["method"].astype(str).unique().tolist())

    created_dirs = []
    for method in methods:
        mask = meta["method"].astype(str).values == method
        if not mask.any():
            print(f"[warning] method not found in pool: {method}")
            continue

        meta_m = meta.loc[mask].copy().reset_index(drop=True)
        Zs_m = Z_seed_all[mask]
        Zf_m = Z_final_all[mask]

        print(f"[create pseudo-run] {method}: n={len(meta_m)}, seeds={meta_m['seed_order'].nunique()}")
        method_dir = write_method_run(run_dir, out_root, method, Zs_m, Zf_m, meta_m, args.link_mode)
        created_dirs.append(method_dir)

    if args.run_posthoc:
        for method_dir in created_dirs:
            run_posthoc_for_method(method_dir, repo_dir, args.posthoc_script, args.posthoc_extra_args)

    summary_path, all_sel_path = collect_outputs(out_root, args.selection_rule)

    print("\n================ DONE ================")
    print("Pseudo-run root:", out_root)
    print("Summary:", summary_path)
    if all_sel_path:
        print("All selected:", all_sel_path)

    print("\nNext files to inspect:")
    print("  ", summary_path)
    if all_sel_path:
        print("  ", all_sel_path)


if __name__ == "__main__":
    main()
