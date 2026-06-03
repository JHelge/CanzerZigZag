#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
selected_seed_anchoring_fair_grid.py

Fair, selected-candidate-specific and candidate-cloud seed anchoring analysis
for ZigZag r x t grid runs.

Why this version is fairer than the previous script
---------------------------------------------------
1. It does NOT compare selected candidates against separately decoded external seeds.
2. It uses the saved trajectory latent file:
      trajectories_latent_all_reps.npz
   and defines the seed for a selected candidate as the round-0 state of the
   exact same trajectory.
3. It decodes round-0 and final trajectory states together and optionally applies
   the same sparsity projection to both. Thus seed and final candidate are in the
   same expression space.
4. It evaluates:
   A) selected-candidate seed pairing
   B) permutation/null seed assignment
   C) latent-space anchoring
   D) full candidate-cloud seed structure
5. It writes one summary per run and one grid-level ranked summary.

Main outputs per grid
---------------------
fair_seed_anchoring_grid_summary.csv

Main outputs per run
--------------------
fair_seed_anchoring_summary.csv
fair_seed_anchoring_per_selected_candidate.csv
fair_seed_anchoring_permutation_null.csv
fair_seed_anchoring_cloud_summary.csv

Interpretation
--------------
actual_over_perm_mean_expr < 1:
    selected final candidates are closer to their own round-0 seeds than to
    randomly assigned round-0 seeds.

p_perm_mean_le_actual_expr close to 0:
    the observed own-seed pairing is better than permutation null.

cloud_within_over_between_latent < 1:
    candidates generated from the same seed form a more compact cloud than
    candidates from different seeds.

Notes
-----
- This script assumes healthy2tumor runs.
- It uses selected_candidates_eval.csv only to decide which candidate keys
  are selected by your rule.
- It uses trajectories_latent_all_reps.npz to get the exact trajectory-native
  round-0 seed and final candidate.
"""

import os
import re
import json
import glob
import argparse
import sys
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import torch


# ============================================================
# ------------------------- Imports ---------------------------
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
        )
        return load_VAE, robust_load_vae, decode_latents_in_batches, project_sparsity_gene_wise, to_numpy_dense
    except Exception as e:
        raise ImportError(
            "Could not import required functions from zigzag.common. "
            "Set --repo_dir to the repository root that contains zigzag/common.py. "
            f"Original error: {e}"
        )


# ============================================================
# ---------------------- Basic utilities ----------------------
# ============================================================

def parse_r_t_from_run_dir(path: str):
    name = os.path.basename(path.rstrip("/"))
    m = re.match(r"^r(\d+)_t(\d+)_s(.+)$", name)
    if m is None:
        return None
    return {"r": int(m.group(1)), "t": int(m.group(2)), "s": m.group(3)}


def to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def safe_rank_own(dist_row: np.ndarray, own_pos: int) -> int:
    own_d = dist_row[own_pos]
    return int(1 + np.sum(dist_row < own_d))


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


def sample_within_between_distances(
    X: np.ndarray,
    group: np.ndarray,
    n_pairs: int = 20000,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    group = np.asarray(group)

    groups = np.unique(group)
    group_to_idx = {g: np.where(group == g)[0] for g in groups}

    within = []
    between = []

    valid_groups = [g for g in groups if len(group_to_idx[g]) >= 2]

    for _ in range(n_pairs):
        if valid_groups:
            g = rng.choice(valid_groups)
            i, j = rng.choice(group_to_idx[g], size=2, replace=False)
            within.append(float(np.linalg.norm(X[i] - X[j])))

        g1, g2 = rng.choice(groups, size=2, replace=False)
        i = rng.choice(group_to_idx[g1])
        j = rng.choice(group_to_idx[g2])
        between.append(float(np.linalg.norm(X[i] - X[j])))

    within = np.asarray(within, dtype=np.float32)
    between = np.asarray(between, dtype=np.float32)

    return {
        "cloud_within_mean": float(np.mean(within)) if len(within) else np.nan,
        "cloud_within_median": float(np.median(within)) if len(within) else np.nan,
        "cloud_between_mean": float(np.mean(between)) if len(between) else np.nan,
        "cloud_between_median": float(np.median(between)) if len(between) else np.nan,
        "cloud_within_over_between_mean": float(np.mean(within) / (np.mean(between) + 1e-9)) if len(within) and len(between) else np.nan,
        "cloud_within_over_between_median": float(np.median(within) / (np.median(between) + 1e-9)) if len(within) and len(between) else np.nan,
        "cloud_n_within_pairs_sampled": int(len(within)),
        "cloud_n_between_pairs_sampled": int(len(between)),
    }


def load_selected_df(run_dir: str, selection_rule: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "selected_candidates_eval.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "selection_rule" in df.columns:
        df = df[df["selection_rule"].astype(str) == selection_rule].copy()

    if df.empty:
        raise RuntimeError(f"No selected candidates for rule={selection_rule} in {path}")

    for c in ["cell_idx", "rep_id"]:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column {c}")

    df["cell_idx"] = df["cell_idx"].astype(int)
    df["rep_id"] = df["rep_id"].astype(int)
    df["_key"] = df["cell_idx"].astype(str) + "__" + df["rep_id"].astype(str)
    return df


def load_traj(run_dir: str):
    path = os.path.join(run_dir, "trajectories_latent_all_reps.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    zdat = np.load(path)
    Z_traj = zdat["Z_traj"].astype(np.float32)
    cell_idx = zdat["cell_idx"].astype(int)
    rep_id = zdat["rep_id"].astype(int)
    keys = np.array([f"{c}__{r}" for c, r in zip(cell_idx, rep_id)], dtype=object)
    return Z_traj, cell_idx, rep_id, keys


def match_selected_trajectories(selected: pd.DataFrame, Z_traj, cell_idx, rep_id, keys):
    key_to_pos = {str(k): i for i, k in enumerate(keys)}
    order = []
    missing = []
    for k in selected["_key"].astype(str).values:
        pos = key_to_pos.get(k, None)
        if pos is None:
            missing.append(k)
        else:
            order.append(pos)

    if missing:
        raise RuntimeError(f"Could not match {len(missing)} selected candidates in trajectory npz. First missing: {missing[:5]}")

    order = np.asarray(order, dtype=int)
    return Z_traj[order], cell_idx[order], rep_id[order]


# ============================================================
# ------------------ Expression-space decoding ----------------
# ============================================================

def decode_and_project_states(
    Z_seed: np.ndarray,
    Z_final: np.ndarray,
    vae,
    device,
    genes: List[str],
    batch: int,
    sparsity_project: bool,
    A_ref_target,
    project_sparsity_gene_wise,
    to_numpy_dense,
    min_detect_rate: float,
    max_detect_rate: float,
):
    """
    Decode seed and final states together, then apply the same sparsity projection
    to both. This ensures exact comparability of seed and final expression.
    """
    Z = np.vstack([Z_seed, Z_final]).astype(np.float32)

    from zigzag.common import decode_latents_in_batches
    X = decode_latents_in_batches(
        vae,
        Z,
        device,
        batch_size=int(batch),
    ).astype(np.float32)

    if sparsity_project:
        tmp = ad.AnnData(
            X=X,
            var=pd.DataFrame(index=pd.Index(genes)),
        )
        tmp = project_sparsity_gene_wise(
            tmp,
            A_ref_target,
            min_detect_rate=float(min_detect_rate),
            max_detect_rate=float(max_detect_rate),
        )
        X = to_numpy_dense(tmp.X).astype(np.float32)

    n = Z_seed.shape[0]
    return X[:n], X[n:]


# ============================================================
# ---------------------- Run-level analysis -------------------
# ============================================================

def analyze_one_run(
    run_dir: str,
    A_test,
    genes: List[str],
    vae,
    device,
    project_sparsity_gene_wise,
    to_numpy_dense,
    args,
) -> Optional[Dict[str, object]]:

    info = parse_r_t_from_run_dir(run_dir)
    if info is None:
        return None

    try:
        selected = load_selected_df(run_dir, args.selection_rule)
        Z_traj, all_cell_idx, all_rep_id, all_keys = load_traj(run_dir)
    except Exception as e:
        print(f"[skip] {run_dir}: {e}")
        return None

    Z_sel_traj, sel_cell_idx, sel_rep_id = match_selected_trajectories(
        selected, Z_traj, all_cell_idx, all_rep_id, all_keys
    )

    Z_seed_sel = Z_sel_traj[:, 0, :].astype(np.float32)
    Z_final_sel = Z_sel_traj[:, -1, :].astype(np.float32)

    # Reference for sparsity projection: target tumor cells from test set.
    if args.sparsity_project:
        if args.label_col not in A_test.obs.columns:
            raise KeyError(f"Missing obs['{args.label_col}'] in h5ad_test")
        target_mask = A_test.obs[args.label_col].astype(str).values == str(args.target_label)
        if int(target_mask.sum()) == 0:
            raise RuntimeError(f"No target cells found with {args.label_col} == {args.target_label}")
        A_ref_target = A_test[target_mask].copy()
    else:
        A_ref_target = None

    # ------------------------------------------------------------
    # Selected-candidate expression anchoring: own vs permutation.
    # ------------------------------------------------------------
    X_seed_sel, X_final_sel = decode_and_project_states(
        Z_seed=Z_seed_sel,
        Z_final=Z_final_sel,
        vae=vae,
        device=device,
        genes=genes,
        batch=args.batch,
        sparsity_project=args.sparsity_project,
        A_ref_target=A_ref_target,
        project_sparsity_gene_wise=project_sparsity_gene_wise,
        to_numpy_dense=to_numpy_dense,
        min_detect_rate=args.sparsity_min_detect_rate,
        max_detect_rate=args.sparsity_max_detect_rate,
    )

    actual_expr = np.linalg.norm(X_final_sel - X_seed_sel, axis=1)
    actual_mean_expr = float(np.mean(actual_expr))
    actual_median_expr = float(np.median(actual_expr))

    # Distances final selected candidates to the selected seed set.
    D_expr = pairwise_dist(X_final_sel, X_seed_sel, batch=args.batch)
    own_rank_expr = np.array([safe_rank_own(D_expr[i], i) for i in range(D_expr.shape[0])], dtype=int)

    nearest_seed_expr = D_expr.min(axis=1)
    median_seed_expr = np.median(D_expr, axis=1)

    # Permutation null.
    rng = np.random.default_rng(args.seed + info["r"] * 1000 + info["t"])
    perm_rows = []
    perm_mean_expr = []
    perm_median_expr = []
    n = X_final_sel.shape[0]

    for b in range(args.n_perm):
        perm = rng.permutation(n)
        if n > 1:
            tries = 0
            while np.any(perm == np.arange(n)) and tries < 100:
                perm = rng.permutation(n)
                tries += 1

        d = np.linalg.norm(X_final_sel - X_seed_sel[perm], axis=1)
        perm_mean_expr.append(float(np.mean(d)))
        perm_median_expr.append(float(np.median(d)))
        perm_rows.append({
            "perm_id": b,
            "mean_permuted_dist_expr": float(np.mean(d)),
            "median_permuted_dist_expr": float(np.median(d)),
        })

    perm_mean_expr = np.asarray(perm_mean_expr, dtype=np.float32)
    perm_median_expr = np.asarray(perm_median_expr, dtype=np.float32)

    p_perm_mean_le_actual_expr = float(np.mean(perm_mean_expr <= actual_mean_expr))
    p_perm_median_le_actual_expr = float(np.mean(perm_median_expr <= actual_median_expr))

    # ------------------------------------------------------------
    # Latent-space selected anchoring.
    # ------------------------------------------------------------
    actual_lat = np.linalg.norm(Z_final_sel - Z_seed_sel, axis=1)
    D_lat = pairwise_dist(Z_final_sel, Z_seed_sel, batch=args.batch)
    own_rank_lat = np.array([safe_rank_own(D_lat[i], i) for i in range(D_lat.shape[0])], dtype=int)

    perm_mean_lat = []
    perm_median_lat = []
    for b in range(args.n_perm):
        perm = rng.permutation(n)
        if n > 1:
            tries = 0
            while np.any(perm == np.arange(n)) and tries < 100:
                perm = rng.permutation(n)
                tries += 1
        d = np.linalg.norm(Z_final_sel - Z_seed_sel[perm], axis=1)
        perm_mean_lat.append(float(np.mean(d)))
        perm_median_lat.append(float(np.median(d)))

    perm_mean_lat = np.asarray(perm_mean_lat, dtype=np.float32)
    perm_median_lat = np.asarray(perm_median_lat, dtype=np.float32)

    # ------------------------------------------------------------
    # Candidate-cloud structure in latent space.
    # Uses all generated candidates, not only selected candidates.
    # ------------------------------------------------------------
    Z_seed_all = Z_traj[:, 0, :].astype(np.float32)
    Z_final_all = Z_traj[:, -1, :].astype(np.float32)

    cloud_lat = sample_within_between_distances(
        X=Z_final_all,
        group=all_cell_idx,
        n_pairs=args.cloud_pairs,
        seed=args.seed + info["r"] * 17 + info["t"] * 19,
    )
    cloud_lat = {f"{k}_latent": v for k, v in cloud_lat.items()}

    # Candidate-cloud own seed rank among unique round-0 seeds.
    unique_seed_ids = np.array(list(dict.fromkeys(all_cell_idx.tolist())), dtype=int)
    unique_seed_latents = []
    for sid in unique_seed_ids:
        pos = np.where(all_cell_idx == sid)[0][0]
        unique_seed_latents.append(Z_seed_all[pos])
    unique_seed_latents = np.vstack(unique_seed_latents).astype(np.float32)

    D_cloud_seed_lat = pairwise_dist(Z_final_all, unique_seed_latents, batch=args.batch)
    sid_to_pos = {int(sid): i for i, sid in enumerate(unique_seed_ids)}
    own_pos_all = np.array([sid_to_pos[int(sid)] for sid in all_cell_idx], dtype=int)
    own_rank_all_lat = np.array(
        [safe_rank_own(D_cloud_seed_lat[i], own_pos_all[i]) for i in range(D_cloud_seed_lat.shape[0])],
        dtype=int
    )

    # ------------------------------------------------------------
    # Optional candidate-cloud structure in expression space.
    # For your grids this is usually 10 seeds x 100 reps = 1000 candidates,
    # so it is feasible.
    # ------------------------------------------------------------
    cloud_expr = {}
    if Z_final_all.shape[0] <= args.max_cloud_expr_decode:
        X_seed_all, X_final_all = decode_and_project_states(
            Z_seed=Z_seed_all,
            Z_final=Z_final_all,
            vae=vae,
            device=device,
            genes=genes,
            batch=args.batch,
            sparsity_project=args.sparsity_project,
            A_ref_target=A_ref_target,
            project_sparsity_gene_wise=project_sparsity_gene_wise,
            to_numpy_dense=to_numpy_dense,
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )

        cloud_expr = sample_within_between_distances(
            X=X_final_all,
            group=all_cell_idx,
            n_pairs=args.cloud_pairs,
            seed=args.seed + info["r"] * 23 + info["t"] * 29,
        )
        cloud_expr = {f"{k}_expr": v for k, v in cloud_expr.items()}

        unique_seed_expr = []
        for sid in unique_seed_ids:
            pos = np.where(all_cell_idx == sid)[0][0]
            unique_seed_expr.append(X_seed_all[pos])
        unique_seed_expr = np.vstack(unique_seed_expr).astype(np.float32)

        D_cloud_seed_expr = pairwise_dist(X_final_all, unique_seed_expr, batch=args.batch)
        own_rank_all_expr = np.array(
            [safe_rank_own(D_cloud_seed_expr[i], own_pos_all[i]) for i in range(D_cloud_seed_expr.shape[0])],
            dtype=int
        )
        cloud_expr.update({
            "cloud_mean_own_rank_expr": float(np.mean(own_rank_all_expr)),
            "cloud_median_own_rank_expr": float(np.median(own_rank_all_expr)),
            "cloud_frac_own_seed_top1_expr": float(np.mean(own_rank_all_expr <= 1)),
            "cloud_frac_own_seed_top3_expr": float(np.mean(own_rank_all_expr <= 3)),
        })

    # ------------------------------------------------------------
    # Per-selected-candidate output.
    # ------------------------------------------------------------
    per = selected.copy()
    per["fair_actual_dist_expr_round0_to_final"] = actual_expr
    per["fair_own_rank_expr_among_selected_seeds"] = own_rank_expr
    per["fair_own_rank_percentile_expr_among_selected_seeds"] = own_rank_expr / max(1, n)
    per["fair_nearest_seed_expr_dist"] = nearest_seed_expr
    per["fair_median_seed_expr_dist"] = median_seed_expr
    per["fair_actual_over_median_seed_expr_dist"] = actual_expr / (median_seed_expr + 1e-9)

    per["fair_actual_dist_latent_round0_to_final"] = actual_lat
    per["fair_own_rank_latent_among_selected_seeds"] = own_rank_lat
    per["fair_own_rank_percentile_latent_among_selected_seeds"] = own_rank_lat / max(1, n)

    out_per = os.path.join(run_dir, args.out_prefix + "_per_selected_candidate.csv")
    per.to_csv(out_per, index=False)

    perm_df = pd.DataFrame(perm_rows)
    out_perm = os.path.join(run_dir, args.out_prefix + "_permutation_null.csv")
    perm_df.to_csv(out_perm, index=False)

    cloud_summary = {
        "run_dir": run_dir,
        "r": info["r"],
        "t": info["t"],
        "n_all_candidates": int(Z_final_all.shape[0]),
        "n_unique_seeds": int(len(unique_seed_ids)),
        "cloud_mean_own_rank_latent": float(np.mean(own_rank_all_lat)),
        "cloud_median_own_rank_latent": float(np.median(own_rank_all_lat)),
        "cloud_frac_own_seed_top1_latent": float(np.mean(own_rank_all_lat <= 1)),
        "cloud_frac_own_seed_top3_latent": float(np.mean(own_rank_all_lat <= 3)),
        **cloud_lat,
        **cloud_expr,
    }
    out_cloud = os.path.join(run_dir, args.out_prefix + "_cloud_summary.csv")
    pd.DataFrame([cloud_summary]).to_csv(out_cloud, index=False)

    # ------------------------------------------------------------
    # Run-level summary.
    # ------------------------------------------------------------
    summary = {
        "run_dir": run_dir,
        "r": info["r"],
        "t": info["t"],
        "s": info["s"],
        "selection_rule": args.selection_rule,
        "n_selected": int(n),
        "n_all_candidates": int(Z_final_all.shape[0]),
        "n_unique_seeds": int(len(unique_seed_ids)),

        # Existing selected candidate performance
        "grid_frac_proba_ge_0p7": float((selected["tumor_proba"] >= 0.7).mean()) if "tumor_proba" in selected.columns else np.nan,
        "grid_mean_tumor_proba": float(selected["tumor_proba"].mean()) if "tumor_proba" in selected.columns else np.nan,
        "grid_median_tumor_proba": float(selected["tumor_proba"].median()) if "tumor_proba" in selected.columns else np.nan,
        "grid_mean_tumor_logit": float(selected["tumor_logit"].mean()) if "tumor_logit" in selected.columns else np.nan,
        "grid_median_tumor_logit": float(selected["tumor_logit"].median()) if "tumor_logit" in selected.columns else np.nan,
        "grid_mean_id_dist_expr_original": float(selected["id_dist_expr"].mean()) if "id_dist_expr" in selected.columns else np.nan,
        "grid_median_id_dist_expr_original": float(selected["id_dist_expr"].median()) if "id_dist_expr" in selected.columns else np.nan,
        "grid_mean_progress01": float(selected["progress01"].mean()) if "progress01" in selected.columns else np.nan,
        "grid_median_progress01": float(selected["progress01"].median()) if "progress01" in selected.columns else np.nan,
        "grid_fallback_frac": float(selected["used_fallback"].astype(bool).mean()) if "used_fallback" in selected.columns else (
            float(selected["fallback"].astype(bool).mean()) if "fallback" in selected.columns else np.nan
        ),

        # Fair selected expression anchoring
        "fair_mean_actual_dist_expr": actual_mean_expr,
        "fair_median_actual_dist_expr": actual_median_expr,
        "fair_perm_mean_dist_expr_mean": float(np.mean(perm_mean_expr)),
        "fair_perm_mean_dist_expr_median": float(np.median(perm_mean_expr)),
        "fair_actual_over_perm_mean_expr": float(actual_mean_expr / (np.mean(perm_mean_expr) + 1e-9)),
        "fair_actual_over_perm_median_expr": float(actual_median_expr / (np.median(perm_median_expr) + 1e-9)),
        "fair_p_perm_mean_le_actual_expr": p_perm_mean_le_actual_expr,
        "fair_p_perm_median_le_actual_expr": p_perm_median_le_actual_expr,
        "fair_mean_own_rank_expr_selected": float(np.mean(own_rank_expr)),
        "fair_median_own_rank_expr_selected": float(np.median(own_rank_expr)),
        "fair_frac_own_seed_top1_expr_selected": float(np.mean(own_rank_expr <= 1)),
        "fair_frac_own_seed_top3_expr_selected": float(np.mean(own_rank_expr <= 3)),

        # Fair selected latent anchoring
        "fair_mean_actual_dist_latent": float(np.mean(actual_lat)),
        "fair_median_actual_dist_latent": float(np.median(actual_lat)),
        "fair_perm_mean_dist_latent_mean": float(np.mean(perm_mean_lat)),
        "fair_perm_mean_dist_latent_median": float(np.median(perm_mean_lat)),
        "fair_actual_over_perm_mean_latent": float(np.mean(actual_lat) / (np.mean(perm_mean_lat) + 1e-9)),
        "fair_actual_over_perm_median_latent": float(np.median(actual_lat) / (np.median(perm_median_lat) + 1e-9)),
        "fair_p_perm_mean_le_actual_latent": float(np.mean(perm_mean_lat <= np.mean(actual_lat))),
        "fair_p_perm_median_le_actual_latent": float(np.mean(perm_median_lat <= np.median(actual_lat))),
        "fair_mean_own_rank_latent_selected": float(np.mean(own_rank_lat)),
        "fair_median_own_rank_latent_selected": float(np.median(own_rank_lat)),
        "fair_frac_own_seed_top1_latent_selected": float(np.mean(own_rank_lat <= 1)),
        "fair_frac_own_seed_top3_latent_selected": float(np.mean(own_rank_lat <= 3)),

        # Candidate cloud
        **cloud_summary,
    }

    out_summary_csv = os.path.join(run_dir, args.out_prefix + "_summary.csv")
    pd.DataFrame([summary]).to_csv(out_summary_csv, index=False)

    out_summary_json = os.path.join(run_dir, args.out_prefix + "_summary.json")
    with open(out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done run] r={info['r']} t={info['t']} "
        f"success={summary['grid_frac_proba_ge_0p7']:.3f} "
        f"expr_actual/perm={summary['fair_actual_over_perm_mean_expr']:.3f} "
        f"cloud_lat_within/between={summary.get('cloud_within_over_between_mean_latent', np.nan):.3f}"
    )

    return summary


# ============================================================
# ----------------------------- Main --------------------------
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--grid_dir", required=True)
    parser.add_argument("--h5ad_test", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--repo_dir", default="/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat")

    parser.add_argument("--selection_rule", default="proba_ge_0.7_min_identity")
    parser.add_argument("--label_col", default="status")
    parser.add_argument("--target_label", default="tumor")

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)

    parser.add_argument("--sparsity_project", action="store_true")
    parser.add_argument("--sparsity_min_detect_rate", type=float, default=0.0)
    parser.add_argument("--sparsity_max_detect_rate", type=float, default=1.0)

    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--cloud_pairs", type=int, default=20000)
    parser.add_argument("--max_cloud_expr_decode", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument("--out_prefix", default="fair_seed_anchoring")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    load_VAE, robust_load_vae, decode_latents_in_batches, project_sparsity_gene_wise, to_numpy_dense = import_project_functions(args.repo_dir)

    print("[load h5ad_test]", args.h5ad_test)
    A_test = sc.read_h5ad(args.h5ad_test)
    genes = list(map(str, A_test.var_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    print("[load VAE]", args.vae_ckpt)
    vae = load_VAE(args.vae_ckpt, num_gene=len(genes), hidden_dim=args.latent_dim).eval().to(device)
    robust_load_vae(vae, args.vae_ckpt)

    run_dirs = sorted(glob.glob(os.path.join(args.grid_dir, "r*_t*_s*")))
    run_dirs = [rd for rd in run_dirs if parse_r_t_from_run_dir(rd) is not None]

    if not run_dirs:
        raise RuntimeError(f"No r*_t*_s* run folders found in {args.grid_dir}")

    print(f"[grid] found {len(run_dirs)} run dirs")

    rows = []
    for rd in run_dirs:
        info = parse_r_t_from_run_dir(rd)
        summary_path = os.path.join(rd, args.out_prefix + "_summary.csv")

        if os.path.exists(summary_path) and not args.force:
            print(f"[read existing] r={info['r']} t={info['t']}")
            row = pd.read_csv(summary_path).iloc[0].to_dict()
            rows.append(row)
            continue

        row = analyze_one_run(
            run_dir=rd,
            A_test=A_test,
            genes=genes,
            vae=vae,
            device=device,
            project_sparsity_gene_wise=project_sparsity_gene_wise,
            to_numpy_dense=to_numpy_dense,
            args=args,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No runs analyzed.")

    grid = pd.DataFrame(rows)

    # Ranking: prioritizes tumor success, no fallback, selected-pair anchoring,
    # candidate-cloud structure, then original identity and tumor proba.
    sort_cols = []
    ascending = []
    for c, asc in [
        ("grid_frac_proba_ge_0p7", False),
        ("grid_fallback_frac", True),
        ("fair_actual_over_perm_mean_expr", True),
        ("fair_p_perm_mean_le_actual_expr", True),
        ("cloud_within_over_between_mean_latent", True),
        ("grid_mean_id_dist_expr_original", True),
        ("grid_mean_tumor_proba", False),
    ]:
        if c in grid.columns:
            sort_cols.append(c)
            ascending.append(asc)

    if sort_cols:
        grid = grid.sort_values(sort_cols, ascending=ascending)

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(args.grid_dir, args.out_prefix + "_grid_summary.csv")

    grid.to_csv(out_csv, index=False)

    print("\n==================== DONE ====================")
    print("Wrote:", out_csv)

    show_cols = [
        "r", "t",
        "grid_frac_proba_ge_0p7",
        "grid_fallback_frac",
        "grid_mean_tumor_proba",
        "grid_mean_id_dist_expr_original",
        "fair_actual_over_perm_mean_expr",
        "fair_p_perm_mean_le_actual_expr",
        "fair_mean_own_rank_expr_selected",
        "fair_frac_own_seed_top3_expr_selected",
        "cloud_within_over_between_mean_latent",
        "cloud_frac_own_seed_top3_latent",
        "cloud_within_over_between_mean_expr",
        "cloud_frac_own_seed_top3_expr",
    ]
    show_cols = [c for c in show_cols if c in grid.columns]
    print("\nTop ranked runs:")
    print(grid[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
