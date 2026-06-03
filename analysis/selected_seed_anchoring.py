#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selected-candidate seed anchoring analysis.

Goal
----
Test whether selected ZigZag candidates remain measurably anchored to their
originating healthy seed cells.

Main comparisons
----------------
1. Actual seed pairing:
     distance(selected_candidate_i, own_seed_i)

2. Permuted seed pairing:
     distance(selected_candidate_i, wrong_seed_j)

3. Rank of own seed among all selected source seeds.

4. Optional real tumor baseline:
     distance(own_seed_i, random real tumor cell)

Required input
--------------
- run_dir containing:
    selected_candidates_eval.csv
    selected_candidates_expr.h5ad

- h5ad_seed:
    aligned TEST h5ad used as seed source for this run

- VAE checkpoint:
    used to decode seed cells into same decoder space

Output
------
selected_seed_anchoring_per_candidate.csv
selected_seed_anchoring_summary.csv
selected_seed_anchoring_permutation_null.csv
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------
# Import your project functions
# ---------------------------------------------------------------------
# Run this script from your scDiffusion/original_codes environment.
# If imports fail, add your repository root via --repo_dir.
def import_project_functions(repo_dir=None):
    import sys

    if repo_dir is not None:
        sys.path.insert(0, repo_dir)

    try:
        from zigzag.common import (
            load_VAE,
            robust_load_vae,
            load_latents_from_h5ad,
            decode_latents_in_batches,
            to_numpy_dense,
        )
        return load_VAE, robust_load_vae, load_latents_from_h5ad, decode_latents_in_batches, to_numpy_dense
    except Exception as e:
        raise ImportError(
            "Could not import from zigzag.common. "
            "Pass --repo_dir pointing to the repository root that contains zigzag/common.py. "
            f"Original error: {e}"
        )


def to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def soft_rank_own(dist_row, own_pos):
    """
    1 = nearest seed.
    """
    own_d = dist_row[own_pos]
    return int(1 + np.sum(dist_row < own_d))


def pairwise_dist(A, B, batch=512):
    """
    Euclidean distance matrix A x B, memory-safe enough for moderate B.
    """
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--h5ad_seed", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--repo_dir", default=None)

    parser.add_argument("--selection_rule", default="proba_ge_0.7_min_identity")
    parser.add_argument("--label_col", default="status")
    parser.add_argument("--healthy_label", default="healthy")
    parser.add_argument("--tumor_label", default="tumor")

    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--n_random_tumor", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_prefix", default="selected_seed_anchoring")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    (
        load_VAE,
        robust_load_vae,
        load_latents_from_h5ad,
        decode_latents_in_batches,
        to_numpy_dense,
    ) = import_project_functions(args.repo_dir)

    eval_path = os.path.join(args.run_dir, "selected_candidates_eval.csv")
    expr_path = os.path.join(args.run_dir, "selected_candidates_expr.h5ad")

    if not os.path.exists(eval_path):
        raise FileNotFoundError(eval_path)

    if not os.path.exists(expr_path):
        raise FileNotFoundError(
            f"{expr_path} missing. "
            "You need selected_candidates_expr.h5ad for expression-space anchoring."
        )

    print("[load]", eval_path)
    sel_eval = pd.read_csv(eval_path)

    if "selection_rule" in sel_eval.columns:
        sel_eval = sel_eval[sel_eval["selection_rule"].astype(str) == args.selection_rule].copy()

    if sel_eval.empty:
        raise RuntimeError(f"No selected rows for selection_rule={args.selection_rule}")

    required = {"cell_idx", "rep_id"}
    missing = required - set(sel_eval.columns)
    if missing:
        raise ValueError(f"selected_candidates_eval.csv missing columns: {missing}")

    print("[load]", expr_path)
    A_sel = sc.read_h5ad(expr_path)

    # Filter selected expression h5ad to same rule if possible.
    if "selection_rule" in A_sel.obs.columns:
        A_sel = A_sel[A_sel.obs["selection_rule"].astype(str).values == args.selection_rule].copy()

    # Ensure matching order by cell_idx + rep_id if possible.
    obs_cols = set(A_sel.obs.columns)
    if {"cell_idx", "rep_id"}.issubset(obs_cols):
        key_eval = sel_eval[["cell_idx", "rep_id"]].astype(int).astype(str).agg("__".join, axis=1).values
        A_sel.obs["_key"] = (
            A_sel.obs[["cell_idx", "rep_id"]].astype(int).astype(str).agg("__".join, axis=1).values
        )

        key_to_pos = {k: i for i, k in enumerate(A_sel.obs["_key"].values)}
        order = [key_to_pos.get(k, None) for k in key_eval]

        if any(x is None for x in order):
            missing_n = sum(x is None for x in order)
            raise RuntimeError(f"Could not match {missing_n} selected candidates in selected_candidates_expr.h5ad")

        A_sel = A_sel[order].copy()
    else:
        if A_sel.n_obs != len(sel_eval):
            raise RuntimeError(
                "selected_candidates_expr.h5ad lacks cell_idx/rep_id and n_obs does not match selected eval rows."
            )

    X_sel = to_dense(A_sel.X).astype(np.float32)

    print("[selected]", X_sel.shape)
    print(sel_eval[["cell_idx", "rep_id", "tumor_proba", "tumor_logit", "id_dist_expr"]].head())

    # ------------------------------------------------------------------
    # Decode seed h5ad with same VAE.
    # ------------------------------------------------------------------
    print("[load seed h5ad]", args.h5ad_seed)
    A_seed = sc.read_h5ad(args.h5ad_seed)

    if args.label_col not in A_seed.obs.columns:
        raise KeyError(f"Missing obs['{args.label_col}'] in seed h5ad")

    # load_latents_from_h5ad should apply same normalize_total/log1p convention as pipeline.
    print("[encode seeds with VAE]")
    Z_all, _info = load_latents_from_h5ad(
        h5ad_path=args.h5ad_seed,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    n_genes = A_sel.n_vars
    vae = load_VAE(args.vae_ckpt, num_gene=n_genes, hidden_dim=args.latent_dim).eval().to(device)
    robust_load_vae(vae, args.vae_ckpt)

    print("[decode all seed/test cells]")
    X_all_dec = decode_latents_in_batches(
        vae,
        Z_all.astype(np.float32),
        device,
        batch_size=args.batch,
    ).astype(np.float32)

    # Candidate own seed rows
    cell_idx = sel_eval["cell_idx"].astype(int).values
    rep_id = sel_eval["rep_id"].astype(int).values

    if np.max(cell_idx) >= X_all_dec.shape[0]:
        raise IndexError(
            f"cell_idx max={np.max(cell_idx)} but seed h5ad has only {X_all_dec.shape[0]} cells. "
            "Check that --h5ad_seed is the same TEST file used for this run."
        )

    X_own_seed = X_all_dec[cell_idx]

    # Background selected source seeds: unique selected seed ids
    unique_seed_ids = np.array(list(dict.fromkeys(cell_idx.tolist())), dtype=int)
    X_selected_seed_set = X_all_dec[unique_seed_ids]

    # All healthy cells in test h5ad as broader seed background
    status = A_seed.obs[args.label_col].astype(str).values
    healthy_idx = np.where(status == args.healthy_label)[0]
    tumor_idx = np.where(status == args.tumor_label)[0]

    if len(healthy_idx) == 0:
        raise RuntimeError(f"No healthy cells found with {args.label_col} == {args.healthy_label}")

    X_healthy_all = X_all_dec[healthy_idx]

    print(f"[background] selected seeds={len(unique_seed_ids)} | all healthy={len(healthy_idx)} | tumor={len(tumor_idx)}")

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------
    actual_dist = np.linalg.norm(X_sel - X_own_seed, axis=1)

    # Distance to selected seed set
    D_selected_seed_set = pairwise_dist(X_sel, X_selected_seed_set, batch=args.batch)

    own_pos_in_selected_seed_set = np.array([
        int(np.where(unique_seed_ids == sid)[0][0])
        for sid in cell_idx
    ])

    own_rank_among_selected_seeds = np.array([
        soft_rank_own(D_selected_seed_set[i], own_pos_in_selected_seed_set[i])
        for i in range(X_sel.shape[0])
    ])

    nearest_selected_seed_dist = D_selected_seed_set.min(axis=1)
    median_selected_seed_dist = np.median(D_selected_seed_set, axis=1)

    # Distance to all healthy cells
    D_healthy_all = pairwise_dist(X_sel, X_healthy_all, batch=args.batch)

    # Map own global cell_idx to position within healthy_idx
    healthy_pos_map = {int(cid): j for j, cid in enumerate(healthy_idx)}
    own_pos_in_healthy = np.array([healthy_pos_map.get(int(sid), -1) for sid in cell_idx], dtype=int)

    own_rank_among_all_healthy = []
    for i in range(X_sel.shape[0]):
        pos = own_pos_in_healthy[i]
        if pos < 0:
            own_rank_among_all_healthy.append(np.nan)
        else:
            own_rank_among_all_healthy.append(soft_rank_own(D_healthy_all[i], pos))
    own_rank_among_all_healthy = np.asarray(own_rank_among_all_healthy, dtype=float)

    nearest_healthy_dist = D_healthy_all.min(axis=1)
    median_healthy_dist = np.median(D_healthy_all, axis=1)

    # ------------------------------------------------------------------
    # Permutation baseline: shuffle selected seed assignments.
    # This tests whether actual candidate→seed pairing is special.
    # ------------------------------------------------------------------
    perm_rows = []
    perm_mean_dists = []
    perm_median_dists = []

    n = X_sel.shape[0]

    for b in range(args.n_perm):
        perm = rng.permutation(n)

        # avoid identical pairing as much as possible for small n
        if n > 1:
            tries = 0
            while np.any(perm == np.arange(n)) and tries < 100:
                perm = rng.permutation(n)
                tries += 1

        X_perm_seed = X_own_seed[perm]
        d_perm = np.linalg.norm(X_sel - X_perm_seed, axis=1)

        perm_mean_dists.append(float(np.mean(d_perm)))
        perm_median_dists.append(float(np.median(d_perm)))

        perm_rows.append({
            "perm_id": b,
            "mean_permuted_dist": float(np.mean(d_perm)),
            "median_permuted_dist": float(np.median(d_perm)),
            "min_permuted_dist": float(np.min(d_perm)),
            "max_permuted_dist": float(np.max(d_perm)),
        })

    perm_df = pd.DataFrame(perm_rows)

    actual_mean = float(np.mean(actual_dist))
    actual_median = float(np.median(actual_dist))

    p_perm_mean_le_actual = float(np.mean(np.asarray(perm_mean_dists) <= actual_mean))
    p_perm_median_le_actual = float(np.mean(np.asarray(perm_median_dists) <= actual_median))

    # ------------------------------------------------------------------
    # Real tumor baseline: random real tumor cells vs own seed.
    # ------------------------------------------------------------------
    tumor_baseline = {}

    if len(tumor_idx) > 0:
        n_draw = min(int(args.n_random_tumor), len(tumor_idx))
        tumor_draw_idx = rng.choice(tumor_idx, size=n_draw, replace=False)
        X_tumor_draw = X_all_dec[tumor_draw_idx]

        # For each selected seed, compare distance from own seed to random tumor cells.
        random_tumor_dist_means = []
        random_tumor_dist_medians = []
        nearest_tumor_dist = []

        for i in range(n):
            d = np.linalg.norm(X_tumor_draw - X_own_seed[i], axis=1)
            random_tumor_dist_means.append(float(np.mean(d)))
            random_tumor_dist_medians.append(float(np.median(d)))
            nearest_tumor_dist.append(float(np.min(d)))

        random_tumor_dist_means = np.asarray(random_tumor_dist_means)
        random_tumor_dist_medians = np.asarray(random_tumor_dist_medians)
        nearest_tumor_dist = np.asarray(nearest_tumor_dist)

        tumor_baseline = {
            "mean_random_real_tumor_dist_to_seed": float(np.mean(random_tumor_dist_means)),
            "median_random_real_tumor_dist_to_seed": float(np.median(random_tumor_dist_medians)),
            "mean_nearest_real_tumor_dist_to_seed": float(np.mean(nearest_tumor_dist)),
            "zigzag_actual_over_random_real_tumor_mean": float(actual_mean / (np.mean(random_tumor_dist_means) + 1e-9)),
            "zigzag_actual_over_nearest_real_tumor_mean": float(actual_mean / (np.mean(nearest_tumor_dist) + 1e-9)),
        }
    else:
        random_tumor_dist_means = np.full(n, np.nan)
        random_tumor_dist_medians = np.full(n, np.nan)
        nearest_tumor_dist = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Per-candidate output
    # ------------------------------------------------------------------
    per_candidate = sel_eval.copy()
    per_candidate["actual_dist_to_own_seed_decoded"] = actual_dist
    per_candidate["own_rank_among_selected_seeds"] = own_rank_among_selected_seeds
    per_candidate["own_rank_percentile_selected_seeds"] = own_rank_among_selected_seeds / max(1, len(unique_seed_ids))
    per_candidate["nearest_selected_seed_dist"] = nearest_selected_seed_dist
    per_candidate["median_selected_seed_dist"] = median_selected_seed_dist
    per_candidate["actual_over_median_selected_seed_dist"] = actual_dist / (median_selected_seed_dist + 1e-9)

    per_candidate["own_rank_among_all_healthy"] = own_rank_among_all_healthy
    per_candidate["own_rank_percentile_all_healthy"] = own_rank_among_all_healthy / max(1, len(healthy_idx))
    per_candidate["nearest_healthy_dist"] = nearest_healthy_dist
    per_candidate["median_healthy_dist"] = median_healthy_dist
    per_candidate["actual_over_median_healthy_dist"] = actual_dist / (median_healthy_dist + 1e-9)

    per_candidate["mean_random_real_tumor_dist_to_seed"] = random_tumor_dist_means
    per_candidate["median_random_real_tumor_dist_to_seed"] = random_tumor_dist_medians
    per_candidate["nearest_real_tumor_dist_to_seed"] = nearest_tumor_dist
    per_candidate["actual_over_random_real_tumor_dist"] = actual_dist / (random_tumor_dist_means + 1e-9)
    per_candidate["actual_over_nearest_real_tumor_dist"] = actual_dist / (nearest_tumor_dist + 1e-9)

    out_per = os.path.join(args.run_dir, f"{args.out_prefix}_per_candidate.csv")
    per_candidate.to_csv(out_per, index=False)

    out_perm = os.path.join(args.run_dir, f"{args.out_prefix}_permutation_null.csv")
    perm_df.to_csv(out_perm, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "run_dir": args.run_dir,
        "h5ad_seed": args.h5ad_seed,
        "selection_rule": args.selection_rule,
        "n_selected": int(n),
        "n_unique_selected_seeds": int(len(unique_seed_ids)),
        "n_healthy_background": int(len(healthy_idx)),
        "n_tumor_background": int(len(tumor_idx)),

        "mean_tumor_proba": float(per_candidate["tumor_proba"].mean()) if "tumor_proba" in per_candidate.columns else np.nan,
        "median_tumor_proba": float(per_candidate["tumor_proba"].median()) if "tumor_proba" in per_candidate.columns else np.nan,
        "frac_tumor_proba_ge_0p7": float((per_candidate["tumor_proba"] >= 0.7).mean()) if "tumor_proba" in per_candidate.columns else np.nan,

        "mean_actual_dist_to_own_seed": actual_mean,
        "median_actual_dist_to_own_seed": actual_median,

        "mean_median_selected_seed_dist": float(np.mean(median_selected_seed_dist)),
        "mean_actual_over_median_selected_seed_dist": float(np.mean(per_candidate["actual_over_median_selected_seed_dist"])),

        "mean_own_rank_among_selected_seeds": float(np.mean(own_rank_among_selected_seeds)),
        "median_own_rank_among_selected_seeds": float(np.median(own_rank_among_selected_seeds)),
        "frac_own_seed_top1_selected_seeds": float(np.mean(own_rank_among_selected_seeds <= 1)),
        "frac_own_seed_top3_selected_seeds": float(np.mean(own_rank_among_selected_seeds <= 3)),

        "mean_own_rank_percentile_all_healthy": float(np.nanmean(per_candidate["own_rank_percentile_all_healthy"])),
        "median_own_rank_percentile_all_healthy": float(np.nanmedian(per_candidate["own_rank_percentile_all_healthy"])),

        "perm_mean_dist_mean": float(np.mean(perm_mean_dists)),
        "perm_mean_dist_median": float(np.median(perm_mean_dists)),
        "perm_median_dist_mean": float(np.mean(perm_median_dists)),
        "perm_median_dist_median": float(np.median(perm_median_dists)),
        "actual_over_perm_mean_dist": float(actual_mean / (np.mean(perm_mean_dists) + 1e-9)),
        "actual_over_perm_median_dist": float(actual_median / (np.median(perm_median_dists) + 1e-9)),
        "p_perm_mean_le_actual": p_perm_mean_le_actual,
        "p_perm_median_le_actual": p_perm_median_le_actual,
        **tumor_baseline,
    }

    out_summary_csv = os.path.join(args.run_dir, f"{args.out_prefix}_summary.csv")
    pd.DataFrame([summary]).to_csv(out_summary_csv, index=False)

    out_summary_json = os.path.join(args.run_dir, f"{args.out_prefix}_summary.json")
    with open(out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE]")
    print("Per candidate:", out_per)
    print("Permutation null:", out_perm)
    print("Summary CSV:", out_summary_csv)
    print("Summary JSON:", out_summary_json)

    print("\nKey results:")
    for k in [
        "n_selected",
        "frac_tumor_proba_ge_0p7",
        "mean_actual_dist_to_own_seed",
        "actual_over_perm_mean_dist",
        "p_perm_mean_le_actual",
        "mean_own_rank_among_selected_seeds",
        "frac_own_seed_top3_selected_seeds",
        "zigzag_actual_over_random_real_tumor_mean",
        "zigzag_actual_over_nearest_real_tumor_mean",
    ]:
        if k in summary:
            print(f"{k}: {summary[k]}")


if __name__ == "__main__":
    main()