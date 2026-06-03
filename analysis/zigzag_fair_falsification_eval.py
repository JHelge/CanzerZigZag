#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zigzag_fair_falsification_eval.py

A less stupid, claim-aligned evaluation for CancerZigZag.

This script does NOT ask:
    "Which method makes the most tumor-like sample?"

Instead it evaluates three falsifiable properties of ZigZag:

1) Tumor-like candidate discovery
   Does the method find tumor-like candidates under a fixed budget?

2) Seed dependence against null models
   Are selected candidates / candidate clouds closer to their true originating
   seed than to permuted/random seeds, especially after removing the global
   healthy->tumor axis?

3) Non-triviality against direct controls
   Is multi-round ZigZag meaningfully different from:
       - r=1 single-cycle diffusion editing
       - Gaussian seed perturbation
       - linear tumor-centroid / tumor-cluster interpolation
   This is assessed as an operating-regime analysis, not a "win every metric"
   leaderboard.

Important design choices
------------------------
- Random real tumor retrieval is intentionally not used as a primary baseline.
- kNN tumor-realism is reported only as a diagnostic, not as a hard filter.
- Centroid/cluster interpolation is treated as a linear-control, not as an
  equivalent generative method.
- The central seed claim is tested by true-seed vs permuted-seed null models.

Input
-----
Requires baseline pools created by your pool generator:

    RUN_DIR/baseline_pools/baseline_pools_latent.npz
    RUN_DIR/baseline_pools/baseline_pools_meta.csv

Outputs
-------
RUN_DIR/zigzag_falsification_eval/

    01_selected_candidate_summary.csv
    02_seed_null_selected.csv
    03_seed_null_candidate_clouds.csv
    04_matched_tumor_bins.csv
    05_discovery_curves.csv
    06_biology_de_pathway.csv
    07_interpretation_flags.csv
    candidate_scores.csv
    selected_per_seed.csv
    eval_meta.json

Recommended interpretation
--------------------------
A meaningful ZigZag result is not "ZigZag wins every metric".
A meaningful result is:

    - tumor-like candidates exist,
    - selected candidates retain residual seed dependence vs permutation,
    - candidate clouds retain seed-conditioned structure,
    - multi-round ZigZag differs from r=1 / Gaussian / linear controls,
    - biological DE/pathway shifts are plausible.

If simple controls match or exceed ZigZag on all of these axes, then the strong
ZigZag claim is not supported.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Imports
# ============================================================

def import_project_functions(repo_dir):
    candidates = []
    if repo_dir:
        candidates.append(os.path.abspath(repo_dir))

    script_dir = os.path.abspath(os.path.dirname(__file__))
    candidates.extend([
        os.path.abspath(os.getcwd()),
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "../..")),
        "/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat",
    ])

    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "zigzag")) and c not in sys.path:
            sys.path.insert(0, c)

    try:
        from zigzag.common import (
            load_VAE,
            robust_load_vae,
            load_latents_from_h5ad,
            decode_latents_in_batches,
            project_sparsity_gene_wise,
            to_numpy_dense,
        )
        return {
            "load_VAE": load_VAE,
            "robust_load_vae": robust_load_vae,
            "load_latents_from_h5ad": load_latents_from_h5ad,
            "decode_latents_in_batches": decode_latents_in_batches,
            "project_sparsity_gene_wise": project_sparsity_gene_wise,
            "to_numpy_dense": to_numpy_dense,
        }
    except Exception as e:
        print("[debug] import search paths:")
        for c in candidates:
            print("  ", c, "contains zigzag:", os.path.isdir(os.path.join(c, "zigzag")))
        raise ImportError(
            "Could not import zigzag.common. --repo_dir must point to folder containing zigzag/. "
            f"Original error: {e}"
        )


# ============================================================
# Helpers
# ============================================================

def dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def safe_median(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def safe_corr(a, b, method="pearson"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    a = a[ok]
    b = b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    if method == "spearman":
        a = pd.Series(a).rank().to_numpy()
        b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def top_overlap(a, b, n=100):
    ia = np.argsort(-np.abs(a))[:n]
    ib = np.argsort(-np.abs(b))[:n]
    return float(len(set(ia).intersection(set(ib))) / max(n, 1))


def unit_axis(a, b):
    v = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Cannot compute axis from identical centroids.")
    return (v / n).astype(np.float32)


def residualize(X, axis, center):
    X = np.asarray(X, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    Xc = X - center[None, :]
    return (Xc - ((Xc @ axis)[:, None] * axis[None, :])).astype(np.float32)


def pairwise_dist(A, B, batch=512):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    Bn = np.sum(B * B, axis=1)[None, :]
    for s in range(0, A.shape[0], batch):
        sl = slice(s, min(A.shape[0], s + batch))
        X = A[sl]
        Xn = np.sum(X * X, axis=1)[:, None]
        D2 = Xn + Bn - 2.0 * (X @ B.T)
        out[sl] = np.sqrt(np.maximum(D2, 0)).astype(np.float32)
    return out


def own_seed_rank_stats(X_final, X_seed, batch=512):
    D = pairwise_dist(X_final, X_seed, batch=batch)
    ranks = []
    for i in range(D.shape[0]):
        own = D[i, i]
        ranks.append(int(1 + np.sum(D[i] < own)))
    ranks = np.asarray(ranks, dtype=int)
    return {
        "own_rank_mean": safe_mean(ranks),
        "own_rank_median": safe_median(ranks),
        "own_top1": float(np.mean(ranks <= 1)),
        "own_top3": float(np.mean(ranks <= 3)),
        "own_top5": float(np.mean(ranks <= 5)),
    }


def permutation_distance_test(X_final, X_seed, rng, n_perm=1000):
    n = X_final.shape[0]
    actual = np.linalg.norm(X_final - X_seed, axis=1)
    actual_mean = float(np.mean(actual))
    actual_median = float(np.median(actual))

    if n < 2:
        return {
            "actual_mean": actual_mean,
            "actual_median": actual_median,
            "perm_mean": np.nan,
            "perm_median": np.nan,
            "actual_over_perm_mean": np.nan,
            "actual_over_perm_median": np.nan,
            "p_perm_mean_le_actual": np.nan,
            "p_perm_median_le_actual": np.nan,
        }

    perm_means, perm_medians = [], []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        tries = 0
        while np.any(perm == np.arange(n)) and tries < 100:
            perm = rng.permutation(n)
            tries += 1
        d = np.linalg.norm(X_final - X_seed[perm], axis=1)
        perm_means.append(float(np.mean(d)))
        perm_medians.append(float(np.median(d)))

    perm_means = np.asarray(perm_means)
    perm_medians = np.asarray(perm_medians)
    return {
        "actual_mean": actual_mean,
        "actual_median": actual_median,
        "perm_mean": float(np.mean(perm_means)),
        "perm_median": float(np.median(perm_medians)),
        "actual_over_perm_mean": float(actual_mean / (np.mean(perm_means) + 1e-9)),
        "actual_over_perm_median": float(actual_median / (np.median(perm_medians) + 1e-9)),
        "p_perm_mean_le_actual": float(np.mean(perm_means <= actual_mean)),
        "p_perm_median_le_actual": float(np.mean(perm_medians <= actual_median)),
    }


def sampled_cloud_stats(X, seed_order, rng, n_pairs=20000):
    X = np.asarray(X, dtype=np.float32)
    seed_order = np.asarray(seed_order, dtype=int)
    groups = {s: np.where(seed_order == s)[0] for s in np.unique(seed_order)}
    valid = [s for s, idx in groups.items() if len(idx) >= 2]
    seeds = list(groups.keys())

    within = []
    between = []
    for _ in range(n_pairs):
        if valid:
            s = rng.choice(valid)
            i, j = rng.choice(groups[s], size=2, replace=False)
            within.append(float(np.linalg.norm(X[i] - X[j])))
        if len(seeds) >= 2:
            s1, s2 = rng.choice(seeds, size=2, replace=False)
            i = rng.choice(groups[s1])
            j = rng.choice(groups[s2])
            between.append(float(np.linalg.norm(X[i] - X[j])))

    return {
        "within_mean": safe_mean(within),
        "between_mean": safe_mean(between),
        "within_over_between_mean": safe_mean(within) / (safe_mean(between) + 1e-9),
        "within_median": safe_median(within),
        "between_median": safe_median(between),
        "within_over_between_median": safe_median(within) / (safe_median(between) + 1e-9),
        "n_within_pairs": int(len(within)),
        "n_between_pairs": int(len(between)),
    }


# ============================================================
# Loading / decoding
# ============================================================

def load_h5ad_latents(args, funcs, h5ad_path):
    Z, _ = funcs["load_latents_from_h5ad"](
        h5ad_path=h5ad_path,
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
    return Z.astype(np.float32)


def decode_project(Z, funcs, vae, device, genes, target_ref, args):
    X = funcs["decode_latents_in_batches"](
        vae,
        Z.astype(np.float32),
        device,
        batch_size=args.batch,
    ).astype(np.float32)

    if args.sparsity_project:
        tmp = ad.AnnData(X=X, var=pd.DataFrame(index=pd.Index(genes)))
        tmp = funcs["project_sparsity_gene_wise"](
            tmp,
            target_ref,
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )
        X = funcs["to_numpy_dense"](tmp.X).astype(np.float32)

    return X


def load_pools(run_dir):
    pool_dir = Path(run_dir) / "baseline_pools"
    npz_path = pool_dir / "baseline_pools_latent.npz"
    meta_path = pool_dir / "baseline_pools_meta.csv"
    manifest_path = pool_dir / "baseline_pools_manifest.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run baseline pool generator first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run baseline pool generator first.")

    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)

    Z_seed = z["Z_seed"].astype(np.float32)
    Z_final = z["Z_final"].astype(np.float32)

    if "candidate_global_idx" not in meta.columns:
        meta.insert(0, "candidate_global_idx", np.arange(len(meta), dtype=int))

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    return Z_seed, Z_final, meta, manifest


# ============================================================
# Scoring
# ============================================================

def fit_classifier(X_h, X_t, n_pcs, seed):
    X = np.vstack([X_h, X_t]).astype(np.float32)
    y = np.array([0] * len(X_h) + [1] * len(X_t), dtype=int)

    n_pcs_eff = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=n_pcs_eff, random_state=seed),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
    )
    clf.fit(X, y)
    return clf


def fit_geometry(X_h, X_t, Z_h, Z_t):
    expr_axis = unit_axis(X_h.mean(axis=0), X_t.mean(axis=0))
    expr_center = ((X_h.mean(axis=0) + X_t.mean(axis=0)) / 2.0).astype(np.float32)

    latent_axis = unit_axis(Z_h.mean(axis=0), Z_t.mean(axis=0))
    latent_center = ((Z_h.mean(axis=0) + Z_t.mean(axis=0)) / 2.0).astype(np.float32)

    proj_h = (X_h - expr_center[None, :]) @ expr_axis
    proj_t = (X_t - expr_center[None, :]) @ expr_axis
    h_med = float(np.median(proj_h))
    t_med = float(np.median(proj_t))
    denom = t_med - h_med
    if abs(denom) < 1e-9:
        denom = 1.0

    return {
        "expr_axis": expr_axis,
        "expr_center": expr_center,
        "latent_axis": latent_axis,
        "latent_center": latent_center,
        "expr_h_median": h_med,
        "expr_t_median": t_med,
        "expr_denom": denom,
    }


def score_candidates(meta, Z_seed, Z_final, X_seed, X_final, clf, geom, nn_tumor, args):
    proba = clf.predict_proba(X_final)[:, 1].astype(float)
    logit = clf.decision_function(X_final).astype(float)

    proj = (X_final - geom["expr_center"][None, :]) @ geom["expr_axis"]
    progress = (proj - geom["expr_h_median"]) / (geom["expr_denom"] + 1e-9)

    Xs_res = residualize(X_seed, geom["expr_axis"], geom["expr_center"])
    Xf_res = residualize(X_final, geom["expr_axis"], geom["expr_center"])
    Zs_res = residualize(Z_seed, geom["latent_axis"], geom["latent_center"])
    Zf_res = residualize(Z_final, geom["latent_axis"], geom["latent_center"])

    d_knn, _ = nn_tumor.kneighbors(X_final, n_neighbors=min(args.knn_k, nn_tumor.n_neighbors))
    tumor_knn = d_knn.mean(axis=1)

    out = meta.copy()
    out["tumor_proba"] = proba
    out["tumor_logit"] = logit
    out["progress01_geom"] = progress.astype(float)
    out["id_dist_expr"] = np.linalg.norm(X_final - X_seed, axis=1).astype(float)
    out["resid_id_dist_expr"] = np.linalg.norm(Xf_res - Xs_res, axis=1).astype(float)
    out["id_dist_latent"] = np.linalg.norm(Z_final - Z_seed, axis=1).astype(float)
    out["resid_id_dist_latent"] = np.linalg.norm(Zf_res - Zs_res, axis=1).astype(float)
    out["tumor_knn_dist_expr"] = tumor_knn.astype(float)
    return out


# ============================================================
# Analyses
# ============================================================

def select_candidates(scores, threshold):
    rows = []
    for (method, seed), g in scores.groupby(["method", "seed_order"], sort=True):
        ok = g[g["tumor_proba"] >= threshold]
        if len(ok):
            idx = ok["id_dist_expr"].idxmin()
            fallback = 0
        else:
            idx = g["tumor_logit"].idxmax()
            fallback = 1
        row = g.loc[idx].copy()
        row["selected_success"] = int(row["tumor_proba"] >= threshold)
        row["used_fallback"] = int(fallback)
        row["selection_rule"] = f"proba_ge_{threshold:g}_min_id_else_max_logit"
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def selected_summary(selected):
    rows = []
    for method, g in selected.groupby("method", sort=True):
        rows.append({
            "method": method,
            "n_seeds": int(len(g)),
            "success_frac": safe_mean(g["selected_success"]),
            "fallback_frac": safe_mean(g["used_fallback"]),
            "mean_tumor_proba": safe_mean(g["tumor_proba"]),
            "median_tumor_proba": safe_median(g["tumor_proba"]),
            "mean_tumor_logit": safe_mean(g["tumor_logit"]),
            "mean_progress01_geom": safe_mean(g["progress01_geom"]),
            "mean_id_dist_expr": safe_mean(g["id_dist_expr"]),
            "mean_resid_id_dist_expr": safe_mean(g["resid_id_dist_expr"]),
            "mean_tumor_knn_dist_expr": safe_mean(g["tumor_knn_dist_expr"]),
        })
    return pd.DataFrame(rows)


def seed_null_selected(selected, Z_seed, Z_final, X_seed, X_final, geom, rng, args):
    rows = []
    selected = selected.copy()
    idx_col = selected["candidate_global_idx"].astype(int).values

    for method, g in selected.groupby("method", sort=True):
        idx = g["candidate_global_idx"].astype(int).values

        spaces = {
            "raw_expr": (X_final[idx], X_seed[idx]),
            "resid_expr": (
                residualize(X_final[idx], geom["expr_axis"], geom["expr_center"]),
                residualize(X_seed[idx], geom["expr_axis"], geom["expr_center"]),
            ),
            "raw_latent": (Z_final[idx], Z_seed[idx]),
            "resid_latent": (
                residualize(Z_final[idx], geom["latent_axis"], geom["latent_center"]),
                residualize(Z_seed[idx], geom["latent_axis"], geom["latent_center"]),
            ),
        }

        for space, (A_final, A_seed) in spaces.items():
            perm = permutation_distance_test(A_final, A_seed, rng, args.n_perm)
            rank = own_seed_rank_stats(A_final, A_seed, args.batch)
            row = {"method": method, "space": space}
            row.update({f"{k}": v for k, v in perm.items()})
            row.update(rank)
            rows.append(row)

    return pd.DataFrame(rows)


def seed_null_clouds(scores, Z_final, X_final, geom, rng, args):
    rows = []
    for method, g in scores.groupby("method", sort=True):
        idx = g["candidate_global_idx"].astype(int).values

        spaces = {
            "raw_expr": X_final[idx],
            "resid_expr": residualize(X_final[idx], geom["expr_axis"], geom["expr_center"]),
            "raw_latent": Z_final[idx],
            "resid_latent": residualize(Z_final[idx], geom["latent_axis"], geom["latent_center"]),
        }

        for space, A in spaces.items():
            st = sampled_cloud_stats(A, g["seed_order"].values, rng, args.cloud_pairs)
            row = {"method": method, "space": space}
            row.update(st)
            rows.append(row)

    return pd.DataFrame(rows)


def matched_tumor_bins(scores, bins):
    rows = []
    for method, g in scores.groupby("method", sort=True):
        for lo, hi in zip(bins[:-1], bins[1:]):
            if hi == bins[-1]:
                sub = g[(g["tumor_proba"] >= lo) & (g["tumor_proba"] <= hi)]
            else:
                sub = g[(g["tumor_proba"] >= lo) & (g["tumor_proba"] < hi)]
            if len(sub) == 0:
                continue
            rows.append({
                "method": method,
                "tumor_proba_bin": f"{lo:g}_{hi:g}",
                "n_candidates": int(len(sub)),
                "n_seeds": int(sub["seed_order"].nunique()),
                "mean_tumor_proba": safe_mean(sub["tumor_proba"]),
                "mean_id_dist_expr": safe_mean(sub["id_dist_expr"]),
                "mean_resid_id_dist_expr": safe_mean(sub["resid_id_dist_expr"]),
                "mean_tumor_knn_dist_expr": safe_mean(sub["tumor_knn_dist_expr"]),
            })
    return pd.DataFrame(rows)


def discovery_curves(scores, threshold, distance_cols, n_grid=40):
    rows = []
    for dist_col in distance_cols:
        vals = pd.to_numeric(scores[dist_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        grid = np.unique(np.quantile(vals, np.linspace(0, 1, n_grid)))

        for method, g in scores.groupby("method", sort=True):
            n_seeds = g["seed_order"].nunique()
            ok_base = g[g["tumor_proba"] >= threshold]
            for d in grid:
                ok = ok_base[ok_base[dist_col] <= d]
                rows.append({
                    "method": method,
                    "tumor_proba_threshold": threshold,
                    "distance_col": dist_col,
                    "distance_threshold": float(d),
                    "success_frac_within_distance": float(ok["seed_order"].nunique() / max(n_seeds, 1)),
                    "n_success_seeds": int(ok["seed_order"].nunique()),
                    "n_seeds": int(n_seeds),
                })
    return pd.DataFrame(rows)


def load_gmt(path, genes):
    if path is None or not os.path.exists(path):
        return {}
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            idx = [gene_to_idx[g] for g in parts[2:] if g in gene_to_idx]
            if len(idx) >= 5:
                sets[parts[0]] = np.asarray(idx, dtype=int)
    return sets


def biology_summary(selected, X_seed, X_final, X_h_ref, X_t_ref, genes, gmt_sets):
    real_de = X_t_ref.mean(axis=0) - X_h_ref.mean(axis=0)
    rows = []

    for method, g in selected.groupby("method", sort=True):
        idx = g["candidate_global_idx"].astype(int).values
        gen_de = X_final[idx].mean(axis=0) - X_seed[idx].mean(axis=0)

        row = {
            "method": method,
            "de_pearson_all": safe_corr(gen_de, real_de, "pearson"),
            "de_spearman_all": safe_corr(gen_de, real_de, "spearman"),
            "de_top50_abs_overlap": top_overlap(gen_de, real_de, 50),
            "de_top100_abs_overlap": top_overlap(gen_de, real_de, 100),
            "de_top200_abs_overlap": top_overlap(gen_de, real_de, 200),
        }

        if gmt_sets:
            real_pw, gen_pw = [], []
            for name, idxs in gmt_sets.items():
                real_pw.append(float(np.mean(real_de[idxs])))
                gen_pw.append(float(np.mean(gen_de[idxs])))
            real_pw = np.asarray(real_pw)
            gen_pw = np.asarray(gen_pw)

            nz = (np.abs(real_pw) > 1e-12) & (np.abs(gen_pw) > 1e-12)
            agree = np.sign(real_pw[nz]) == np.sign(gen_pw[nz])
            row.update({
                "pathway_n_sets": int(len(real_pw)),
                "pathway_pearson": safe_corr(gen_pw, real_pw, "pearson"),
                "pathway_spearman": safe_corr(gen_pw, real_pw, "spearman"),
                "pathway_direction_agree_frac": float(np.mean(agree)) if len(agree) else np.nan,
            })
        rows.append(row)

    return pd.DataFrame(rows)


def interpretation_flags(sel_sum, seed_null, cloud_null, bio, args):
    """
    Small diagnostic summary. This is not a statistical decision engine;
    it makes the interpretation transparent.
    """
    rows = []
    merged = sel_sum.merge(bio, on="method", how="left")

    resid_sel = seed_null[seed_null["space"] == "resid_expr"][["method", "actual_over_perm_mean", "p_perm_mean_le_actual", "own_top1", "own_top3"]]
    resid_cloud = cloud_null[cloud_null["space"] == "resid_expr"][["method", "within_over_between_mean"]]
    merged = merged.merge(resid_sel, on="method", how="left")
    merged = merged.merge(resid_cloud, on="method", how="left", suffixes=("", "_cloud"))

    for _, r in merged.iterrows():
        rows.append({
            "method": r["method"],
            "tumor_discovery_ok": bool(r.get("success_frac", 0) >= 0.8),
            "residual_seed_selected_ok": bool(r.get("actual_over_perm_mean", np.inf) < 1.0),
            "residual_cloud_seed_structure_ok": bool(r.get("within_over_between_mean", np.inf) < 1.0),
            "biology_direction_ok": bool(r.get("pathway_direction_agree_frac", 0) >= 0.6) if "pathway_direction_agree_frac" in r else None,
            "de_corr_ok": bool(r.get("de_pearson_all", 0) > 0.25) if "de_pearson_all" in r else None,
            "note": "Interpret jointly. ZigZag need not win all metrics; controls matching all axes would weaken the claim.",
        })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--run_dir", required=True)
    p.add_argument("--h5ad_train", required=True)
    p.add_argument("--h5ad_test", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--repo_dir", default="/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat")

    p.add_argument("--label_col", default="status")
    p.add_argument("--healthy_label", default="healthy")
    p.add_argument("--tumor_label", default="tumor")

    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_pcs", type=int, default=50)
    p.add_argument("--max_ref_cells", type=int, default=5000)
    p.add_argument("--success_proba", type=float, default=0.7)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--cloud_pairs", type=int, default=20000)
    p.add_argument("--knn_k", type=int, default=15)

    p.add_argument("--sparsity_project", action="store_true")
    p.add_argument("--sparsity_min_detect_rate", type=float, default=0.0)
    p.add_argument("--sparsity_max_detect_rate", type=float, default=1.0)

    p.add_argument("--pathway_gmt", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--out_prefix", default="zigzag_falsification")
    p.add_argument("--seed", type=int, default=17)

    # tolerate old slurm args
    p.add_argument("--single_cycle_run_dir", default=None)
    p.add_argument("--selection_rule", default=None)
    p.add_argument("--progress_threshold", default=None)
    p.add_argument("--matched_progress_tolerance", default=None)
    p.add_argument("--budget", default=None)
    p.add_argument("--interp_alphas", nargs="*", default=None)
    p.add_argument("--n_tumor_clusters", default=None)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.outdir is None:
        args.outdir = os.path.join(args.run_dir, "zigzag_falsification_eval")
    os.makedirs(args.outdir, exist_ok=True)

    funcs = import_project_functions(args.repo_dir)

    print("[load pools]")
    Z_seed, Z_final, meta, manifest = load_pools(args.run_dir)

    print("[load h5ad]")
    A_train = sc.read_h5ad(args.h5ad_train)
    A_test = sc.read_h5ad(args.h5ad_test)
    genes = list(map(str, A_test.var_names))

    y_train = A_train.obs[args.label_col].astype(str).values
    y_test = A_test.obs[args.label_col].astype(str).values

    train_h = np.where(y_train == args.healthy_label)[0]
    train_t = np.where(y_train == args.tumor_label)[0]
    test_h = np.where(y_test == args.healthy_label)[0]
    test_t = np.where(y_test == args.tumor_label)[0]

    if len(train_h) == 0 or len(train_t) == 0 or len(test_h) == 0 or len(test_t) == 0:
        raise RuntimeError("Need healthy and tumor cells in both train and test.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    print("[load VAE]")
    vae = funcs["load_VAE"](args.vae_ckpt, num_gene=len(genes), hidden_dim=args.latent_dim).eval().to(device)
    funcs["robust_load_vae"](vae, args.vae_ckpt)

    print("[encode refs]")
    Z_train = load_h5ad_latents(args, funcs, args.h5ad_train)
    Z_test = load_h5ad_latents(args, funcs, args.h5ad_test)

    h_tr = rng.choice(train_h, size=min(args.max_ref_cells, len(train_h)), replace=False)
    t_tr = rng.choice(train_t, size=min(args.max_ref_cells, len(train_t)), replace=False)
    h_te = rng.choice(test_h, size=min(args.max_ref_cells, len(test_h)), replace=False)
    t_te = rng.choice(test_t, size=min(args.max_ref_cells, len(test_t)), replace=False)

    target_ref = A_test[test_t].copy()

    print("[decode refs]")
    X_h_train = decode_project(Z_train[h_tr], funcs, vae, device, genes, target_ref, args)
    X_t_train = decode_project(Z_train[t_tr], funcs, vae, device, genes, target_ref, args)
    X_h_test = decode_project(Z_test[h_te], funcs, vae, device, genes, target_ref, args)
    X_t_test = decode_project(Z_test[t_te], funcs, vae, device, genes, target_ref, args)

    print("[fit common scorer]")
    clf = fit_classifier(X_h_train, X_t_train, args.eval_pcs, args.seed)
    geom = fit_geometry(X_h_train, X_t_train, Z_train[h_tr], Z_train[t_tr])

    print("[fit kNN diagnostic]")
    k_eff = min(args.knn_k, len(X_t_test))
    nn_tumor = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn_tumor.fit(X_t_test)

    print("[decode candidates]")
    X_seed = decode_project(Z_seed, funcs, vae, device, genes, target_ref, args)
    X_final = decode_project(Z_final, funcs, vae, device, genes, target_ref, args)

    print("[score candidates]")
    scores = score_candidates(meta, Z_seed, Z_final, X_seed, X_final, clf, geom, nn_tumor, args)

    print("[select candidates]")
    selected = select_candidates(scores, args.success_proba)

    print("[analysis 1 selected summary]")
    sel_sum = selected_summary(selected)

    print("[analysis 2 seed null selected]")
    seed_sel = seed_null_selected(selected, Z_seed, Z_final, X_seed, X_final, geom, rng, args)

    print("[analysis 3 seed null candidate clouds]")
    cloud = seed_null_clouds(scores, Z_final, X_final, geom, rng, args)

    print("[analysis 4 matched bins]")
    bins = matched_tumor_bins(scores, [0.0, 0.5, 0.7, 0.8, 0.9, 1.01])

    print("[analysis 5 discovery curves]")
    curves = discovery_curves(scores, args.success_proba, ["id_dist_expr", "resid_id_dist_expr"], n_grid=40)

    print("[analysis 6 biology]")
    gmt_sets = load_gmt(args.pathway_gmt, genes)
    bio = biology_summary(selected, X_seed, X_final, X_h_test, X_t_test, genes, gmt_sets)

    print("[analysis 7 interpretation flags]")
    flags = interpretation_flags(sel_sum, seed_sel, cloud, bio, args)

    prefix = args.out_prefix
    paths = {
        "candidate_scores": os.path.join(args.outdir, "candidate_scores.csv"),
        "selected": os.path.join(args.outdir, "selected_per_seed.csv"),
        "selected_summary": os.path.join(args.outdir, "01_selected_candidate_summary.csv"),
        "seed_null_selected": os.path.join(args.outdir, "02_seed_null_selected.csv"),
        "seed_null_clouds": os.path.join(args.outdir, "03_seed_null_candidate_clouds.csv"),
        "matched_bins": os.path.join(args.outdir, "04_matched_tumor_bins.csv"),
        "discovery_curves": os.path.join(args.outdir, "05_discovery_curves.csv"),
        "biology": os.path.join(args.outdir, "06_biology_de_pathway.csv"),
        "flags": os.path.join(args.outdir, "07_interpretation_flags.csv"),
        "meta": os.path.join(args.outdir, "eval_meta.json"),
    }

    scores.to_csv(paths["candidate_scores"], index=False)
    selected.to_csv(paths["selected"], index=False)
    sel_sum.to_csv(paths["selected_summary"], index=False)
    seed_sel.to_csv(paths["seed_null_selected"], index=False)
    cloud.to_csv(paths["seed_null_clouds"], index=False)
    bins.to_csv(paths["matched_bins"], index=False)
    curves.to_csv(paths["discovery_curves"], index=False)
    bio.to_csv(paths["biology"], index=False)
    flags.to_csv(paths["flags"], index=False)

    meta_json = {
        "evaluation_type": "claim-aligned falsification evaluation",
        "not_a_leaderboard": True,
        "main_question": "Does ZigZag show tumor-like discovery, residual seed dependence, and non-triviality vs direct controls?",
        "selection_rule": f"tumor_proba >= {args.success_proba}: min id_dist_expr else max tumor_logit",
        "classifier": "common PCA+logistic regression trained on decoded TRAIN healthy vs tumor refs; applied to all methods",
        "null_models": ["permuted seed assignment", "within-vs-between seed clouds"],
        "controls": sorted([m for m in scores["method"].unique() if m != "zigzag_pool"]),
        "manifest": manifest,
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta_json, f, indent=2)

    print("\n================ DONE ================")
    for k, v in paths.items():
        print(f"{k}: {v}")

    # Compact preview
    preview = sel_sum.merge(
        seed_sel[seed_sel["space"] == "resid_expr"][["method", "actual_over_perm_mean", "own_top1", "own_top3"]],
        on="method", how="left"
    ).merge(
        cloud[cloud["space"] == "resid_expr"][["method", "within_over_between_mean"]],
        on="method", how="left"
    ).merge(
        bio, on="method", how="left"
    )

    show = [
        "method", "success_frac", "fallback_frac", "mean_tumor_proba",
        "mean_id_dist_expr", "mean_resid_id_dist_expr",
        "actual_over_perm_mean", "within_over_between_mean",
        "de_pearson_all", "pathway_direction_agree_frac",
    ]
    show = [c for c in show if c in preview.columns]
    print("\n[compact preview]")
    print(preview[show].to_string(index=False))


if __name__ == "__main__":
    main()
