#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
residual_seed_identity_grid.py

Full/fair seed-identity analysis for ZigZag grids.

This script addresses the main problem with naive seed-identity metrics:
if a candidate becomes strongly tumor-like, the global healthy->tumor shift
dominates Euclidean distances. Therefore this script reports both:

1) raw seed anchoring
2) residual seed anchoring after removing the real healthy->tumor axis
3) candidate-cloud seed structure overall
4) candidate-cloud seed structure within matched tumor_proba bins
5) posthoc residual-identity selection rules

Inputs per run folder
---------------------
Required:
  - trajectories_latent_all_reps.npz
  - selected_candidates_eval.csv
  - decoded_progress_identity.csv

The pipeline writes decoded_progress_identity.csv from the all-candidate
decoded_metrics table and selected_candidates_eval.csv from predefined
per-seed selection rules.

Main outputs
------------
Per run:
  - residual_seed_identity_selected_candidates.csv
  - residual_seed_identity_cloud_bins.csv
  - residual_seed_identity_posthoc_selection.csv
  - residual_seed_identity_summary.csv
  - residual_seed_identity_summary.json

Grid:
  - residual_seed_identity_grid_summary.csv

Key interpretation
------------------
selected_resid_expr_actual_over_perm_mean < 1:
  selected candidates remain closer to their own trajectory seed than
  random seed assignments after removing the tumor axis.

cloud_bin_resid_expr_within_over_between < 1 in high tumor_proba bins:
  tumor-like candidate clouds remain seed-conditioned at matched tumor-likeness.

posthoc_resid_expr_proba_ge_0.7_min_resid_identity:
  asks whether the same generated candidate pool contains tumor-like candidates
  with better residual seed anchoring than the original rule.

Notes
-----
- This is an evaluation/posthoc analysis script. It does not regenerate samples.
- It assumes healthy2tumor runs.
"""

import os
import re
import sys
import json
import glob
import argparse
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import torch


# ============================================================
# Imports from your repository
# ============================================================

def import_project_functions(repo_dir: Optional[str]):
    if repo_dir:
        sys.path.insert(0, repo_dir)

    try:
        from zigzag.common import (
            load_VAE,
            robust_load_vae,
            decode_latents_in_batches,
            project_sparsity_gene_wise,
            to_numpy_dense,
            load_latents_from_h5ad,
        )
        return {
            "load_VAE": load_VAE,
            "robust_load_vae": robust_load_vae,
            "decode_latents_in_batches": decode_latents_in_batches,
            "project_sparsity_gene_wise": project_sparsity_gene_wise,
            "to_numpy_dense": to_numpy_dense,
            "load_latents_from_h5ad": load_latents_from_h5ad,
        }
    except Exception as e:
        raise ImportError(
            "Could not import required functions from zigzag.common. "
            "Set --repo_dir to the folder containing the zigzag package. "
            f"Original error: {e}"
        )


# ============================================================
# Utilities
# ============================================================

def parse_run_dir(path: str):
    name = os.path.basename(path.rstrip("/"))
    m = re.match(r"^r(\d+)_t(\d+)_s(.+)$", name)
    if m is None:
        return None
    return {"r": int(m.group(1)), "t": int(m.group(2)), "s": m.group(3)}


def to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def pairwise_dist(A: np.ndarray, B: np.ndarray, batch: int = 512) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    B_norm = np.sum(B * B, axis=1)[None, :]
    for start in range(0, A.shape[0], batch):
        sl = slice(start, min(A.shape[0], start + batch))
        X = A[sl]
        X_norm = np.sum(X * X, axis=1)[:, None]
        D2 = X_norm + B_norm - 2.0 * (X @ B.T)
        D2 = np.maximum(D2, 0.0)
        out[sl] = np.sqrt(D2).astype(np.float32)
    return out


def rank_own(dist_row: np.ndarray, own_pos: int) -> int:
    own = dist_row[own_pos]
    return int(1 + np.sum(dist_row < own))


def unit_axis(mu_src: np.ndarray, mu_tgt: np.ndarray) -> np.ndarray:
    v = np.asarray(mu_tgt, dtype=np.float32) - np.asarray(mu_src, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Cannot define tumor axis: source and target means are identical.")
    return (v / n).astype(np.float32)


def residualize(X: np.ndarray, axis: np.ndarray, center: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    Xc = X - center[None, :]
    proj = (Xc @ axis)[:, None] * axis[None, :]
    return (Xc - proj).astype(np.float32)


def summarize_actual_vs_perm(
    X_final: np.ndarray,
    X_seed: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    batch: int,
    prefix: str,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    n = X_final.shape[0]
    actual = np.linalg.norm(X_final - X_seed, axis=1)
    actual_mean = float(np.mean(actual))
    actual_median = float(np.median(actual))

    D = pairwise_dist(X_final, X_seed, batch=batch)
    ranks = np.array([rank_own(D[i], i) for i in range(n)], dtype=int)

    rows = []
    mean_perm = []
    median_perm = []
    for b in range(n_perm):
        perm = rng.permutation(n)
        if n > 1:
            tries = 0
            while np.any(perm == np.arange(n)) and tries < 100:
                perm = rng.permutation(n)
                tries += 1
        d = np.linalg.norm(X_final - X_seed[perm], axis=1)
        mean_perm.append(float(np.mean(d)))
        median_perm.append(float(np.median(d)))
        rows.append({
            "perm_id": int(b),
            f"{prefix}_mean_perm_dist": float(np.mean(d)),
            f"{prefix}_median_perm_dist": float(np.median(d)),
        })

    mean_perm = np.asarray(mean_perm, dtype=np.float32)
    median_perm = np.asarray(median_perm, dtype=np.float32)

    out = {
        f"{prefix}_actual_mean": actual_mean,
        f"{prefix}_actual_median": actual_median,
        f"{prefix}_perm_mean_mean": float(np.mean(mean_perm)),
        f"{prefix}_perm_mean_median": float(np.median(mean_perm)),
        f"{prefix}_perm_median_mean": float(np.mean(median_perm)),
        f"{prefix}_perm_median_median": float(np.median(median_perm)),
        f"{prefix}_actual_over_perm_mean": float(actual_mean / (np.mean(mean_perm) + 1e-9)),
        f"{prefix}_actual_over_perm_median": float(actual_median / (np.median(median_perm) + 1e-9)),
        f"{prefix}_p_perm_mean_le_actual": float(np.mean(mean_perm <= actual_mean)),
        f"{prefix}_p_perm_median_le_actual": float(np.mean(median_perm <= actual_median)),
        f"{prefix}_own_rank_mean": float(np.mean(ranks)),
        f"{prefix}_own_rank_median": float(np.median(ranks)),
        f"{prefix}_own_seed_top1": float(np.mean(ranks <= 1)),
        f"{prefix}_own_seed_top3": float(np.mean(ranks <= 3)),
        f"{prefix}_own_seed_top5": float(np.mean(ranks <= 5)),
    }

    perm_df = pd.DataFrame(rows)
    return out, perm_df, ranks


def sampled_within_between(
    X: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int,
) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float32)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    g2i = {g: np.where(groups == g)[0] for g in unique}
    valid_within = [g for g in unique if len(g2i[g]) >= 2]

    within = []
    between = []

    for _ in range(n_pairs):
        if valid_within:
            g = rng.choice(valid_within)
            i, j = rng.choice(g2i[g], size=2, replace=False)
            within.append(float(np.linalg.norm(X[i] - X[j])))

        if len(unique) >= 2:
            g1, g2 = rng.choice(unique, size=2, replace=False)
            i = rng.choice(g2i[g1])
            j = rng.choice(g2i[g2])
            between.append(float(np.linalg.norm(X[i] - X[j])))

    within = np.asarray(within, dtype=np.float32)
    between = np.asarray(between, dtype=np.float32)

    return {
        "within_mean": float(np.mean(within)) if len(within) else np.nan,
        "within_median": float(np.median(within)) if len(within) else np.nan,
        "between_mean": float(np.mean(between)) if len(between) else np.nan,
        "between_median": float(np.median(between)) if len(between) else np.nan,
        "within_over_between_mean": float(np.mean(within) / (np.mean(between) + 1e-9)) if len(within) and len(between) else np.nan,
        "within_over_between_median": float(np.median(within) / (np.median(between) + 1e-9)) if len(within) and len(between) else np.nan,
        "n_within_pairs": int(len(within)),
        "n_between_pairs": int(len(between)),
    }


def own_rank_against_unique_seeds(
    X_final: np.ndarray,
    X_seed_per_candidate: np.ndarray,
    seed_ids: np.ndarray,
    batch: int,
) -> Dict[str, float]:
    unique_seed_ids = np.array(list(dict.fromkeys(seed_ids.astype(int).tolist())), dtype=int)
    seed_ref = []
    for sid in unique_seed_ids:
        pos = np.where(seed_ids == sid)[0][0]
        seed_ref.append(X_seed_per_candidate[pos])
    seed_ref = np.vstack(seed_ref).astype(np.float32)

    sid2pos = {int(s): i for i, s in enumerate(unique_seed_ids)}
    own_pos = np.array([sid2pos[int(s)] for s in seed_ids], dtype=int)

    D = pairwise_dist(X_final, seed_ref, batch=batch)
    ranks = np.array([rank_own(D[i], own_pos[i]) for i in range(D.shape[0])], dtype=int)

    return {
        "own_rank_mean": float(np.mean(ranks)),
        "own_rank_median": float(np.median(ranks)),
        "own_seed_top1": float(np.mean(ranks <= 1)),
        "own_seed_top3": float(np.mean(ranks <= 3)),
        "own_seed_top5": float(np.mean(ranks <= 5)),
    }


# ============================================================
# Data loading
# ============================================================

def load_run_tables(run_dir: str, selection_rule: str):
    sel_path = os.path.join(run_dir, "selected_candidates_eval.csv")
    all_path = os.path.join(run_dir, "decoded_progress_identity.csv")
    traj_path = os.path.join(run_dir, "trajectories_latent_all_reps.npz")

    if not os.path.exists(sel_path):
        raise FileNotFoundError(sel_path)
    if not os.path.exists(all_path):
        raise FileNotFoundError(all_path)
    if not os.path.exists(traj_path):
        raise FileNotFoundError(traj_path)

    sel = pd.read_csv(sel_path)
    if "selection_rule" in sel.columns:
        sel = sel[sel["selection_rule"].astype(str) == selection_rule].copy()
    if sel.empty:
        raise RuntimeError(f"No selected rows for selection_rule={selection_rule}")

    allm = pd.read_csv(all_path)

    for df in [sel, allm]:
        df["cell_idx"] = df["cell_idx"].astype(int)
        df["rep_id"] = df["rep_id"].astype(int)
        df["_key"] = df["cell_idx"].astype(str) + "__" + df["rep_id"].astype(str)

    zdat = np.load(traj_path)
    Z_traj = zdat["Z_traj"].astype(np.float32)
    cell_idx = zdat["cell_idx"].astype(int)
    rep_id = zdat["rep_id"].astype(int)
    keys = np.array([f"{c}__{r}" for c, r in zip(cell_idx, rep_id)], dtype=object)

    key_to_pos = {str(k): i for i, k in enumerate(keys)}

    return sel, allm, Z_traj, cell_idx, rep_id, keys, key_to_pos


def subset_by_keys(df: pd.DataFrame, key_to_pos: Dict[str, int]):
    pos = []
    missing = []
    for k in df["_key"].astype(str).values:
        p = key_to_pos.get(k, None)
        if p is None:
            missing.append(k)
        else:
            pos.append(p)
    if missing:
        raise RuntimeError(f"Could not match {len(missing)} rows to trajectories. First missing={missing[:5]}")
    return np.asarray(pos, dtype=int)


# ============================================================
# Reference axes
# ============================================================

def fit_reference_axes(args, funcs, vae, device, A_test, genes):
    """
    Fit real healthy->tumor axes in latent and expression space.
    The expression axis can be raw decoded or target-sparsity projected.
    """
    label = A_test.obs[args.label_col].astype(str).values
    h_idx = np.where(label == args.source_label)[0]
    t_idx = np.where(label == args.target_label)[0]
    if len(h_idx) == 0 or len(t_idx) == 0:
        raise RuntimeError(f"Need both source={args.source_label} and target={args.target_label} in h5ad_test.")

    rng = np.random.default_rng(args.seed)
    h_sub = rng.choice(h_idx, size=min(args.max_ref_cells, len(h_idx)), replace=False)
    t_sub = rng.choice(t_idx, size=min(args.max_ref_cells, len(t_idx)), replace=False)

    print(f"[axis] encoding h={len(h_sub)} t={len(t_sub)} reference cells")

    # Encode all test cells once using existing loader.
    Z_all, _ = funcs["load_latents_from_h5ad"](
        h5ad_path=args.h5ad_test,
        vae_ckpt=args.vae_ckpt,
        hidden_dim=args.latent_dim,
        normalize_total=True,
        target_sum=1e4,
        log1p=True,
        layer=None,
        encode_batch=4096,
        return_obs=False,
        return_var_names=False,
    )

    Z_h = Z_all[h_sub].astype(np.float32)
    Z_t = Z_all[t_sub].astype(np.float32)

    latent_center = ((Z_h.mean(axis=0) + Z_t.mean(axis=0)) / 2.0).astype(np.float32)
    latent_axis = unit_axis(Z_h.mean(axis=0), Z_t.mean(axis=0))

    # Decode refs for expression axis.
    X_h = funcs["decode_latents_in_batches"](vae, Z_h, device, batch_size=args.batch).astype(np.float32)
    X_t = funcs["decode_latents_in_batches"](vae, Z_t, device, batch_size=args.batch).astype(np.float32)

    if args.sparsity_project and args.axis_projection_mode == "target_for_both":
        target_ref = A_test[t_idx].copy()
        tmp = ad.AnnData(X=np.vstack([X_h, X_t]), var=pd.DataFrame(index=pd.Index(genes)))
        tmp = funcs["project_sparsity_gene_wise"](
            tmp,
            target_ref,
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )
        X = funcs["to_numpy_dense"](tmp.X).astype(np.float32)
        X_h = X[:len(Z_h)]
        X_t = X[len(Z_h):]

    expr_center = ((X_h.mean(axis=0) + X_t.mean(axis=0)) / 2.0).astype(np.float32)
    expr_axis = unit_axis(X_h.mean(axis=0), X_t.mean(axis=0))

    return {
        "Z_all": Z_all.astype(np.float32),
        "latent_axis": latent_axis,
        "latent_center": latent_center,
        "expr_axis": expr_axis,
        "expr_center": expr_center,
        "h_idx": h_idx,
        "t_idx": t_idx,
    }


# ============================================================
# Decode/project candidates
# ============================================================

def decode_project_pair(
    Z_seed: np.ndarray,
    Z_final: np.ndarray,
    funcs,
    vae,
    device,
    genes: List[str],
    A_test,
    args,
):
    Z = np.vstack([Z_seed, Z_final]).astype(np.float32)
    X = funcs["decode_latents_in_batches"](vae, Z, device, batch_size=args.batch).astype(np.float32)

    if args.sparsity_project:
        label = A_test.obs[args.label_col].astype(str).values
        t_idx = np.where(label == args.target_label)[0]
        target_ref = A_test[t_idx].copy()
        tmp = ad.AnnData(X=X, var=pd.DataFrame(index=pd.Index(genes)))
        tmp = funcs["project_sparsity_gene_wise"](
            tmp,
            target_ref,
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )
        X = funcs["to_numpy_dense"](tmp.X).astype(np.float32)

    n = Z_seed.shape[0]
    return X[:n], X[n:]


# ============================================================
# Analysis
# ============================================================

def analyze_run(run_dir, A_test, genes, axes, funcs, vae, device, args):
    info = parse_run_dir(run_dir)
    if info is None:
        return None

    try:
        sel, allm, Z_traj, cell_idx, rep_id, keys, key_to_pos = load_run_tables(run_dir, args.selection_rule)
    except Exception as e:
        print(f"[skip] {run_dir}: {e}")
        return None

    # Restrict allm to rows that are present in trajectories and keep same order as trajectories.
    all_pos = subset_by_keys(allm, key_to_pos)
    allm = allm.copy()
    allm["_traj_pos"] = all_pos

    # In your full decode_frac=1 runs, all rows should be present. Use allm order for metrics.
    Z_seed_all = Z_traj[all_pos, 0, :].astype(np.float32)
    Z_final_all = Z_traj[all_pos, -1, :].astype(np.float32)
    seed_ids_all = allm["cell_idx"].astype(int).values
    tumor_proba_all = allm["tumor_proba"].astype(float).values if "tumor_proba" in allm.columns else np.full(len(allm), np.nan)
    tumor_logit_all = allm["tumor_logit"].astype(float).values if "tumor_logit" in allm.columns else np.full(len(allm), np.nan)

    sel_pos_in_allm = []
    allm_key_to_row = {k: i for i, k in enumerate(allm["_key"].astype(str).values)}
    for k in sel["_key"].astype(str).values:
        if k not in allm_key_to_row:
            raise RuntimeError(f"Selected key not present in all candidates: {k}")
        sel_pos_in_allm.append(allm_key_to_row[k])
    sel_pos_in_allm = np.asarray(sel_pos_in_allm, dtype=int)

    # Decode all candidates if feasible.
    if len(allm) > args.max_decode_candidates:
        raise RuntimeError(
            f"Run {run_dir} has {len(allm)} candidates > --max_decode_candidates={args.max_decode_candidates}. "
            "Increase max or reduce candidates."
        )

    X_seed_all, X_final_all = decode_project_pair(
        Z_seed_all, Z_final_all, funcs, vae, device, genes, A_test, args
    )

    # Raw and residual spaces.
    Z_seed_all_res = residualize(Z_seed_all, axes["latent_axis"], axes["latent_center"])
    Z_final_all_res = residualize(Z_final_all, axes["latent_axis"], axes["latent_center"])

    X_seed_all_res = residualize(X_seed_all, axes["expr_axis"], axes["expr_center"])
    X_final_all_res = residualize(X_final_all, axes["expr_axis"], axes["expr_center"])

    rng = np.random.default_rng(args.seed + info["r"] * 1009 + info["t"] * 917)

    # ------------------------------
    # Selected-candidate metrics
    # ------------------------------
    selected_spaces = {
        "selected_raw_latent": (Z_final_all[sel_pos_in_allm], Z_seed_all[sel_pos_in_allm]),
        "selected_resid_latent": (Z_final_all_res[sel_pos_in_allm], Z_seed_all_res[sel_pos_in_allm]),
        "selected_raw_expr": (X_final_all[sel_pos_in_allm], X_seed_all[sel_pos_in_allm]),
        "selected_resid_expr": (X_final_all_res[sel_pos_in_allm], X_seed_all_res[sel_pos_in_allm]),
    }

    summary = {
        "run_dir": run_dir,
        "r": info["r"],
        "t": info["t"],
        "s": info["s"],
        "selection_rule": args.selection_rule,
        "n_all_candidates": int(len(allm)),
        "n_selected": int(len(sel)),
        "n_unique_seeds": int(pd.Series(seed_ids_all).nunique()),
        "grid_frac_proba_ge_0p7": float((sel["tumor_proba"] >= 0.7).mean()) if "tumor_proba" in sel.columns else np.nan,
        "grid_mean_tumor_proba": float(sel["tumor_proba"].mean()) if "tumor_proba" in sel.columns else np.nan,
        "grid_median_tumor_proba": float(sel["tumor_proba"].median()) if "tumor_proba" in sel.columns else np.nan,
        "grid_mean_tumor_logit": float(sel["tumor_logit"].mean()) if "tumor_logit" in sel.columns else np.nan,
        "grid_mean_id_dist_expr_original": float(sel["id_dist_expr"].mean()) if "id_dist_expr" in sel.columns else np.nan,
        "grid_fallback_frac": float(sel["fallback_used"].astype(bool).mean()) if "fallback_used" in sel.columns else np.nan,
    }

    per_selected = sel.copy()
    permutation_frames = []

    for prefix, (XF, XS) in selected_spaces.items():
        m, perm_df, ranks = summarize_actual_vs_perm(
            XF, XS, rng=rng, n_perm=args.n_perm, batch=args.batch, prefix=prefix
        )
        summary.update(m)
        per_selected[f"{prefix}_actual_dist"] = np.linalg.norm(XF - XS, axis=1)
        per_selected[f"{prefix}_own_rank"] = ranks
        perm_df["space"] = prefix
        permutation_frames.append(perm_df)

    per_selected.to_csv(os.path.join(run_dir, "residual_seed_identity_selected_candidates.csv"), index=False)
    if permutation_frames:
        pd.concat(permutation_frames, ignore_index=True).to_csv(
            os.path.join(run_dir, "residual_seed_identity_permutation_null.csv"),
            index=False,
        )

    # ------------------------------
    # Candidate cloud structure
    # ------------------------------
    cloud_rows = []
    bins = args.tumor_bins
    bin_labels = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (tumor_proba_all >= lo) & (tumor_proba_all <= hi)
        else:
            mask = (tumor_proba_all >= lo) & (tumor_proba_all < hi)
        bin_labels.append((lo, hi, mask))

    # Overall + bins
    cloud_definitions = [("all", np.ones(len(allm), dtype=bool))]
    for lo, hi, mask in bin_labels:
        cloud_definitions.append((f"proba_{lo:g}_{hi:g}", mask))

    cloud_spaces = {
        "raw_latent": (Z_final_all, Z_seed_all),
        "resid_latent": (Z_final_all_res, Z_seed_all_res),
        "raw_expr": (X_final_all, X_seed_all),
        "resid_expr": (X_final_all_res, X_seed_all_res),
    }

    for bin_name, mask in cloud_definitions:
        if int(mask.sum()) < args.min_bin_candidates:
            continue

        for space_name, (XF_all, XS_all) in cloud_spaces.items():
            XF = XF_all[mask]
            XS = XS_all[mask]
            sid = seed_ids_all[mask]

            rng_cloud = np.random.default_rng(args.seed + info["r"] * 31 + info["t"] * 37 + len(cloud_rows))

            wb = sampled_within_between(
                XF, sid,
                rng=rng_cloud,
                n_pairs=args.cloud_pairs,
            )
            rank = own_rank_against_unique_seeds(
                XF, XS, sid,
                batch=args.batch,
            )
            row = {
                "run_dir": run_dir,
                "r": info["r"],
                "t": info["t"],
                "bin": bin_name,
                "space": space_name,
                "n_candidates": int(mask.sum()),
                "n_seeds": int(pd.Series(sid).nunique()),
                "mean_tumor_proba": float(np.nanmean(tumor_proba_all[mask])),
                "median_tumor_proba": float(np.nanmedian(tumor_proba_all[mask])),
                **{f"cloud_{k}": v for k, v in wb.items()},
                **{f"cloud_{k}": v for k, v in rank.items()},
            }
            cloud_rows.append(row)

            # Put the most important overall/high-bin metrics into summary.
            if bin_name == "all":
                for k, v in row.items():
                    if k.startswith("cloud_"):
                        summary[f"cloud_all_{space_name}_{k.replace('cloud_', '')}"] = v
            if bin_name == "proba_0.7_0.8" or bin_name == "proba_0.7_0.9":
                for k, v in row.items():
                    if k.startswith("cloud_"):
                        summary[f"cloud_{bin_name}_{space_name}_{k.replace('cloud_', '')}"] = v

    cloud_df = pd.DataFrame(cloud_rows)
    cloud_df.to_csv(os.path.join(run_dir, "residual_seed_identity_cloud_bins.csv"), index=False)

    # ------------------------------
    # Posthoc selection using residual identity
    # ------------------------------
    posthoc_rows = []
    thresholds = args.selection_thresholds

    # Precompute per-candidate own distances.
    dist_raw_expr = np.linalg.norm(X_final_all - X_seed_all, axis=1)
    dist_resid_expr = np.linalg.norm(X_final_all_res - X_seed_all_res, axis=1)
    dist_raw_latent = np.linalg.norm(Z_final_all - Z_seed_all, axis=1)
    dist_resid_latent = np.linalg.norm(Z_final_all_res - Z_seed_all_res, axis=1)

    allm_ext = allm.copy()
    allm_ext["_row_pos"] = np.arange(len(allm_ext))
    allm_ext["dist_raw_expr_round0"] = dist_raw_expr
    allm_ext["dist_resid_expr_round0"] = dist_resid_expr
    allm_ext["dist_raw_latent_round0"] = dist_raw_latent
    allm_ext["dist_resid_latent_round0"] = dist_resid_latent

    def select_by(seed_group, thr, metric):
        ok = seed_group[seed_group["tumor_proba"] >= thr]
        fallback = False
        if len(ok) == 0:
            ok = seed_group
            fallback = True
            idx = ok["tumor_logit"].idxmax()
        else:
            idx = ok[metric].idxmin()
        row = seed_group.loc[idx].copy()
        row["posthoc_threshold"] = thr
        row["posthoc_metric"] = metric
        row["posthoc_fallback"] = int(fallback)
        return row

    selected_posthoc_all = []
    for thr in thresholds:
        for metric in [
            "dist_raw_expr_round0",
            "dist_resid_expr_round0",
            "dist_raw_latent_round0",
            "dist_resid_latent_round0",
        ]:
            rows = []
            for sid, g in allm_ext.groupby("cell_idx", sort=False):
                rows.append(select_by(g, thr, metric))
            ph = pd.DataFrame(rows)
            selected_posthoc_all.append(ph)

            posthoc_rows.append({
                "run_dir": run_dir,
                "r": info["r"],
                "t": info["t"],
                "threshold": float(thr),
                "metric": metric,
                "n_selected": int(len(ph)),
                "frac_proba_ge_threshold": float((ph["tumor_proba"] >= thr).mean()),
                "fallback_frac": float(ph["posthoc_fallback"].mean()),
                "mean_tumor_proba": float(ph["tumor_proba"].mean()),
                "median_tumor_proba": float(ph["tumor_proba"].median()),
                "mean_tumor_logit": float(ph["tumor_logit"].mean()),
                "mean_dist_raw_expr": float(ph["dist_raw_expr_round0"].mean()),
                "mean_dist_resid_expr": float(ph["dist_resid_expr_round0"].mean()),
                "mean_dist_raw_latent": float(ph["dist_raw_latent_round0"].mean()),
                "mean_dist_resid_latent": float(ph["dist_resid_latent_round0"].mean()),
            })

    posthoc_df = pd.DataFrame(posthoc_rows)
    posthoc_df.to_csv(os.path.join(run_dir, "residual_seed_identity_posthoc_selection.csv"), index=False)

    # Add key posthoc threshold 0.7/min residual expr if present.
    key = posthoc_df[
        (np.isclose(posthoc_df["threshold"], 0.7)) &
        (posthoc_df["metric"] == "dist_resid_expr_round0")
    ]
    if len(key):
        k = key.iloc[0].to_dict()
        for kk, vv in k.items():
            if kk not in {"run_dir", "r", "t", "threshold", "metric"}:
                summary[f"posthoc_proba_ge_0p7_min_resid_expr_{kk}"] = vv

    # Write summary.
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(run_dir, "residual_seed_identity_summary.csv"), index=False)
    with open(os.path.join(run_dir, "residual_seed_identity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] r={info['r']} t={info['t']} "
        f"success={summary.get('grid_frac_proba_ge_0p7', np.nan):.2f} "
        f"resid_expr_actual/perm={summary.get('selected_resid_expr_actual_over_perm_mean', np.nan):.3f} "
        f"cloud_all_resid_expr={summary.get('cloud_all_resid_expr_within_over_between_mean', np.nan):.3f}"
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--grid_dir", required=True)
    p.add_argument("--h5ad_test", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--repo_dir", default="/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat")

    p.add_argument("--selection_rule", default="proba_ge_0.7_min_identity")
    p.add_argument("--label_col", default="status")
    p.add_argument("--source_label", default="healthy")
    p.add_argument("--target_label", default="tumor")

    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)

    p.add_argument("--sparsity_project", action="store_true")
    p.add_argument("--sparsity_min_detect_rate", type=float, default=0.0)
    p.add_argument("--sparsity_max_detect_rate", type=float, default=1.0)
    p.add_argument("--axis_projection_mode", choices=["raw", "target_for_both"], default="target_for_both")

    p.add_argument("--max_ref_cells", type=int, default=5000)
    p.add_argument("--max_decode_candidates", type=int, default=2000)

    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--cloud_pairs", type=int, default=20000)
    p.add_argument("--min_bin_candidates", type=int, default=50)
    p.add_argument("--tumor_bins", nargs="+", type=float, default=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    p.add_argument("--selection_thresholds", nargs="+", type=float, default=[0.6, 0.7, 0.8])
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--force", action="store_true")
    p.add_argument("--out_csv", default=None)

    args = p.parse_args()

    funcs = import_project_functions(args.repo_dir)

    print("[load test]", args.h5ad_test)
    A_test = sc.read_h5ad(args.h5ad_test)
    genes = list(map(str, A_test.var_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    print("[load VAE]", args.vae_ckpt)
    vae = funcs["load_VAE"](args.vae_ckpt, num_gene=len(genes), hidden_dim=args.latent_dim).eval().to(device)
    funcs["robust_load_vae"](vae, args.vae_ckpt)

    axes = fit_reference_axes(args, funcs, vae, device, A_test, genes)

    run_dirs = sorted(glob.glob(os.path.join(args.grid_dir, "r*_t*_s*")))
    run_dirs = [rd for rd in run_dirs if parse_run_dir(rd) is not None]
    if not run_dirs:
        raise RuntimeError(f"No run dirs found in {args.grid_dir}")

    print(f"[grid] {len(run_dirs)} run dirs")

    rows = []
    for rd in run_dirs:
        out_summary = os.path.join(rd, "residual_seed_identity_summary.csv")
        if os.path.exists(out_summary) and not args.force:
            print("[read existing]", rd)
            rows.append(pd.read_csv(out_summary).iloc[0].to_dict())
            continue

        try:
            row = analyze_run(rd, A_test, genes, axes, funcs, vae, device, args)
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"[ERROR skip] {rd}: {e}")

    if not rows:
        raise RuntimeError("No run summaries generated.")

    grid = pd.DataFrame(rows)

    # Rank by: success, fallback, residual selected pairing, high-tumor-bin cloud, posthoc residual selection.
    sort_cols = []
    ascending = []
    for c, asc in [
        ("grid_frac_proba_ge_0p7", False),
        ("grid_fallback_frac", True),
        ("selected_resid_expr_actual_over_perm_mean", True),
        ("selected_resid_expr_p_perm_mean_le_actual", True),
        ("cloud_all_resid_expr_within_over_between_mean", True),
        ("posthoc_proba_ge_0p7_min_resid_expr_fallback_frac", True),
        ("posthoc_proba_ge_0p7_min_resid_expr_mean_dist_resid_expr", True),
        ("grid_mean_tumor_proba", False),
    ]:
        if c in grid.columns:
            sort_cols.append(c)
            ascending.append(asc)

    if sort_cols:
        grid = grid.sort_values(sort_cols, ascending=ascending)

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(args.grid_dir, "residual_seed_identity_grid_summary.csv")
    grid.to_csv(out_csv, index=False)

    print("\n================ DONE ================")
    print("Wrote:", out_csv)
    show_cols = [
        "r", "t",
        "grid_frac_proba_ge_0p7",
        "grid_fallback_frac",
        "grid_mean_tumor_proba",
        "grid_mean_id_dist_expr_original",
        "selected_resid_expr_actual_over_perm_mean",
        "selected_resid_expr_p_perm_mean_le_actual",
        "selected_resid_latent_actual_over_perm_mean",
        "cloud_all_resid_expr_within_over_between_mean",
        "cloud_all_resid_latent_within_over_between_mean",
        "posthoc_proba_ge_0p7_min_resid_expr_fallback_frac",
        "posthoc_proba_ge_0p7_min_resid_expr_mean_tumor_proba",
        "posthoc_proba_ge_0p7_min_resid_expr_mean_dist_resid_expr",
    ]
    show_cols = [c for c in show_cols if c in grid.columns]
    print(grid[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
