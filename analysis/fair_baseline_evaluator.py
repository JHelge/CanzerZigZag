#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fair_baseline_evaluator.py

Direct, fair baseline evaluation for ZigZag / CancerZigZag.

This script deliberately avoids the failed posthoc/pseudo-run approach.
It does not compare ZigZag to random real tumor retrieval as a main baseline.
It does not use a geometric progress-only score as the main success criterion.
It does not use a hard kNN-realism filter that favors centroid interpolation.

Core benchmark
--------------
Same seeds, same candidate pools, same scoring model, same selection rule:

    For each method and each seed:
        generate/evaluate candidate pool
        if any candidate has tumor_proba >= threshold:
            select candidate with minimal id_dist_expr
        else:
            fallback to candidate with maximal tumor_logit

Then compare methods jointly by:
    - tumor discovery
    - seed distance
    - residual seed anchoring
    - candidate-cloud structure
    - DE/pathway concordance
    - realism diagnostics, reported separately, not as a hard filter

Inputs
------
Requires candidate pools created by baseline_candidate_discovery.py:

    RUN_DIR/baseline_pools/baseline_pools_latent.npz
    RUN_DIR/baseline_pools/baseline_pools_meta.csv

Outputs
-------
Written to RUN_DIR/fair_baseline_eval/ by default:

    fair_baseline_summary.csv
    fair_baseline_selected_per_seed.csv
    fair_baseline_pool_summary.csv
    fair_baseline_candidate_scores.csv
    fair_baseline_matched_bins.csv
    fair_baseline_de_pathway_summary.csv
    fair_baseline_meta.json

Important interpretation
------------------------
ZigZag does not need to win every single metric. A meaningful result is a
distinct operating regime: robust tumor-like discovery with residual seed
anchoring and stronger biological concordance than simple seed-based controls.
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
# Project imports
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
# Utilities
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
        out[sl] = np.sqrt(np.maximum(D2, 0.0)).astype(np.float32)
    return out


def own_seed_ranks(X_final, X_seed, batch=512):
    D = pairwise_dist(X_final, X_seed, batch=batch)
    ranks = []
    for i in range(D.shape[0]):
        own = D[i, i]
        ranks.append(int(1 + np.sum(D[i] < own)))
    return np.asarray(ranks, dtype=int)


def actual_over_permuted(X_final, X_seed, rng, n_perm=1000):
    n = X_final.shape[0]
    actual = np.linalg.norm(X_final - X_seed, axis=1)
    if n < 2:
        return {
            "actual_mean": safe_mean(actual),
            "perm_mean": np.nan,
            "actual_over_perm_mean": np.nan,
            "p_perm_mean_le_actual": np.nan,
        }

    perm_means = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        tries = 0
        while np.any(perm == np.arange(n)) and tries < 100:
            perm = rng.permutation(n)
            tries += 1
        d = np.linalg.norm(X_final - X_seed[perm], axis=1)
        perm_means.append(float(np.mean(d)))

    perm_means = np.asarray(perm_means, dtype=float)
    return {
        "actual_mean": float(np.mean(actual)),
        "perm_mean": float(np.mean(perm_means)),
        "actual_over_perm_mean": float(np.mean(actual) / (np.mean(perm_means) + 1e-9)),
        "p_perm_mean_le_actual": float(np.mean(perm_means <= np.mean(actual))),
    }


def sampled_within_between(X, seed_order, rng, n_pairs=20000):
    X = np.asarray(X, dtype=np.float32)
    seed_order = np.asarray(seed_order, dtype=int)
    groups = {s: np.where(seed_order == s)[0] for s in np.unique(seed_order)}
    valid_within = [s for s, idx in groups.items() if len(idx) >= 2]
    all_seeds = list(groups.keys())

    within = []
    between = []
    for _ in range(n_pairs):
        if valid_within:
            s = rng.choice(valid_within)
            i, j = rng.choice(groups[s], size=2, replace=False)
            within.append(float(np.linalg.norm(X[i] - X[j])))
        if len(all_seeds) >= 2:
            s1, s2 = rng.choice(all_seeds, size=2, replace=False)
            i = rng.choice(groups[s1])
            j = rng.choice(groups[s2])
            between.append(float(np.linalg.norm(X[i] - X[j])))

    within = np.asarray(within, dtype=float)
    between = np.asarray(between, dtype=float)
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


def load_baseline_pools(run_dir):
    pool_dir = Path(run_dir) / "baseline_pools"
    npz_path = pool_dir / "baseline_pools_latent.npz"
    meta_path = pool_dir / "baseline_pools_meta.csv"
    manifest_path = pool_dir / "baseline_pools_manifest.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run pool generator first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run pool generator first.")

    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)

    Z_seed = z["Z_seed"].astype(np.float32)
    Z_final = z["Z_final"].astype(np.float32)

    if len(meta) != Z_seed.shape[0]:
        raise ValueError(f"meta rows {len(meta)} != Z rows {Z_seed.shape[0]}")

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    return Z_seed, Z_final, meta, manifest


# ============================================================
# Scoring and reference geometry
# ============================================================

def fit_eval_classifier(X_h_train, X_t_train, n_pcs, seed):
    X = np.vstack([X_h_train, X_t_train]).astype(np.float32)
    y = np.array([0] * len(X_h_train) + [1] * len(X_t_train), dtype=int)

    n_pcs_eff = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=n_pcs_eff, random_state=seed),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
    )
    clf.fit(X, y)
    return clf


def score_classifier(clf, X):
    proba = clf.predict_proba(X)[:, 1].astype(float)
    logit = clf.decision_function(X).astype(float)
    return logit, proba


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
        "expr_h_centroid": X_h.mean(axis=0).astype(np.float32),
        "expr_t_centroid": X_t.mean(axis=0).astype(np.float32),
    }


def add_candidate_metrics(meta, Z_seed, Z_final, X_seed, X_final, clf, geom, nn_tumor, args):
    logit, proba = score_classifier(clf, X_final)

    proj = (X_final - geom["expr_center"][None, :]) @ geom["expr_axis"]
    progress01 = (proj - geom["expr_h_median"]) / (geom["expr_denom"] + 1e-9)

    Xs_res = residualize(X_seed, geom["expr_axis"], geom["expr_center"])
    Xf_res = residualize(X_final, geom["expr_axis"], geom["expr_center"])
    Zs_res = residualize(Z_seed, geom["latent_axis"], geom["latent_center"])
    Zf_res = residualize(Z_final, geom["latent_axis"], geom["latent_center"])

    d_tumor_knn, _ = nn_tumor.kneighbors(X_final, n_neighbors=min(args.knn_k, nn_tumor.n_neighbors))
    tumor_knn = d_tumor_knn.mean(axis=1)

    out = meta.copy()
    out["tumor_logit"] = logit
    out["tumor_proba"] = proba
    out["progress01_geom"] = progress01.astype(float)
    out["id_dist_expr"] = np.linalg.norm(X_final - X_seed, axis=1).astype(float)
    out["resid_id_dist_expr"] = np.linalg.norm(Xf_res - Xs_res, axis=1).astype(float)
    out["id_dist_latent"] = np.linalg.norm(Z_final - Z_seed, axis=1).astype(float)
    out["resid_id_dist_latent"] = np.linalg.norm(Zf_res - Zs_res, axis=1).astype(float)
    out["tumor_knn_dist_expr"] = tumor_knn.astype(float)
    return out


# ============================================================
# Selection / evaluation
# ============================================================

def select_per_method_seed(scores, threshold):
    rows = []
    for (method, seed_order), g in scores.groupby(["method", "seed_order"], sort=True):
        ok = g[g["tumor_proba"] >= threshold]
        if len(ok):
            idx = ok["id_dist_expr"].idxmin()
            used_fallback = 0
        else:
            idx = g["tumor_logit"].idxmax()
            used_fallback = 1
        row = g.loc[idx].copy()
        row["selection_rule"] = f"proba_ge_{threshold:g}_min_identity"
        row["selected_success"] = int(row["tumor_proba"] >= threshold)
        row["used_fallback"] = int(used_fallback)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_selected(selected, Z_seed, Z_final, X_seed, X_final, scores, geom, rng, args):
    rows = []

    # Map selected global idx to array positions.
    selected = selected.copy()
    selected["_array_idx"] = selected["candidate_global_idx"].astype(int)

    for method, g in selected.groupby("method", sort=True):
        idx = g["_array_idx"].values.astype(int)

        Zs = Z_seed[idx]
        Zf = Z_final[idx]
        Xs = X_seed[idx]
        Xf = X_final[idx]

        Xs_res = residualize(Xs, geom["expr_axis"], geom["expr_center"])
        Xf_res = residualize(Xf, geom["expr_axis"], geom["expr_center"])
        Zs_res = residualize(Zs, geom["latent_axis"], geom["latent_center"])
        Zf_res = residualize(Zf, geom["latent_axis"], geom["latent_center"])

        own_raw_expr = own_seed_ranks(Xf, Xs, batch=args.batch)
        own_resid_expr = own_seed_ranks(Xf_res, Xs_res, batch=args.batch)

        raw_perm = actual_over_permuted(Xf, Xs, rng, n_perm=args.n_perm)
        resid_perm = actual_over_permuted(Xf_res, Xs_res, rng, n_perm=args.n_perm)
        lat_perm = actual_over_permuted(Zf_res, Zs_res, rng, n_perm=args.n_perm)

        row = {
            "method": method,
            "n_selected": int(len(g)),
            "success_frac_proba_ge_threshold": safe_mean(g["selected_success"]),
            "fallback_frac": safe_mean(g["used_fallback"]),
            "mean_tumor_proba": safe_mean(g["tumor_proba"]),
            "median_tumor_proba": safe_median(g["tumor_proba"]),
            "mean_tumor_logit": safe_mean(g["tumor_logit"]),
            "mean_progress01_geom": safe_mean(g["progress01_geom"]),
            "mean_id_dist_expr": safe_mean(g["id_dist_expr"]),
            "median_id_dist_expr": safe_median(g["id_dist_expr"]),
            "mean_resid_id_dist_expr": safe_mean(g["resid_id_dist_expr"]),
            "median_resid_id_dist_expr": safe_median(g["resid_id_dist_expr"]),
            "mean_id_dist_latent": safe_mean(g["id_dist_latent"]),
            "mean_resid_id_dist_latent": safe_mean(g["resid_id_dist_latent"]),
            "mean_tumor_knn_dist_expr": safe_mean(g["tumor_knn_dist_expr"]),
            "own_seed_rank_raw_expr_mean": safe_mean(own_raw_expr),
            "own_seed_rank_resid_expr_mean": safe_mean(own_resid_expr),
            "own_seed_top1_raw_expr": float(np.mean(own_raw_expr <= 1)),
            "own_seed_top1_resid_expr": float(np.mean(own_resid_expr <= 1)),
            "raw_expr_actual_over_perm_mean": raw_perm["actual_over_perm_mean"],
            "raw_expr_p_perm_mean_le_actual": raw_perm["p_perm_mean_le_actual"],
            "resid_expr_actual_over_perm_mean": resid_perm["actual_over_perm_mean"],
            "resid_expr_p_perm_mean_le_actual": resid_perm["p_perm_mean_le_actual"],
            "resid_latent_actual_over_perm_mean": lat_perm["actual_over_perm_mean"],
            "resid_latent_p_perm_mean_le_actual": lat_perm["p_perm_mean_le_actual"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_pool(scores, Z_final, X_final, geom, rng, args):
    rows = []
    for method, g in scores.groupby("method", sort=True):
        idx = g["candidate_global_idx"].astype(int).values

        Xf_res = residualize(X_final[idx], geom["expr_axis"], geom["expr_center"])
        Zf_res = residualize(Z_final[idx], geom["latent_axis"], geom["latent_center"])

        wb_expr = sampled_within_between(Xf_res, g["seed_order"].values, rng, n_pairs=args.cloud_pairs)
        wb_lat = sampled_within_between(Zf_res, g["seed_order"].values, rng, n_pairs=args.cloud_pairs)

        row = {
            "method": method,
            "n_candidates": int(len(g)),
            "n_seeds": int(g["seed_order"].nunique()),
            "pool_frac_tumor_proba_ge_threshold": float((g["tumor_proba"] >= args.success_proba).mean()),
            "pool_mean_tumor_proba": safe_mean(g["tumor_proba"]),
            "pool_mean_progress01_geom": safe_mean(g["progress01_geom"]),
            "pool_mean_id_dist_expr": safe_mean(g["id_dist_expr"]),
            "pool_mean_resid_id_dist_expr": safe_mean(g["resid_id_dist_expr"]),
            "pool_mean_tumor_knn_dist_expr": safe_mean(g["tumor_knn_dist_expr"]),
            "cloud_resid_expr_within_over_between_mean": wb_expr["within_over_between_mean"],
            "cloud_resid_latent_within_over_between_mean": wb_lat["within_over_between_mean"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def matched_bins(scores, bins):
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


# ============================================================
# DE / pathway
# ============================================================

def top_overlap(a, b, n=100):
    ia = np.argsort(-np.abs(a))[:n]
    ib = np.argsort(-np.abs(b))[:n]
    return float(len(set(ia).intersection(set(ib))) / max(n, 1))


def corr_safe(a, b, method="pearson"):
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
            name = parts[0]
            idx = [gene_to_idx[g] for g in parts[2:] if g in gene_to_idx]
            if len(idx) >= 5:
                sets[name] = np.asarray(idx, dtype=int)
    return sets


def de_pathway_summary(selected, X_seed, X_final, X_h_ref, X_t_ref, genes, gmt_sets):
    rows = []
    real_de = X_t_ref.mean(axis=0) - X_h_ref.mean(axis=0)

    for method, g in selected.groupby("method", sort=True):
        idx = g["candidate_global_idx"].astype(int).values
        gen_de = X_final[idx].mean(axis=0) - X_seed[idx].mean(axis=0)

        row = {
            "method": method,
            "de_pearson_all": corr_safe(gen_de, real_de, "pearson"),
            "de_spearman_all": corr_safe(gen_de, real_de, "spearman"),
            "de_top50_abs_overlap": top_overlap(gen_de, real_de, 50),
            "de_top100_abs_overlap": top_overlap(gen_de, real_de, 100),
            "de_top200_abs_overlap": top_overlap(gen_de, real_de, 200),
        }

        if gmt_sets:
            real_pw = []
            gen_pw = []
            for name, idxs in gmt_sets.items():
                real_pw.append(float(np.mean(real_de[idxs])))
                gen_pw.append(float(np.mean(gen_de[idxs])))

            real_pw = np.asarray(real_pw, dtype=float)
            gen_pw = np.asarray(gen_pw, dtype=float)

            nz = (np.abs(real_pw) > 1e-12) & (np.abs(gen_pw) > 1e-12)
            agree = np.sign(real_pw[nz]) == np.sign(gen_pw[nz])

            row.update({
                "pathway_n_sets": int(len(real_pw)),
                "pathway_pearson": corr_safe(gen_pw, real_pw, "pearson"),
                "pathway_spearman": corr_safe(gen_pw, real_pw, "spearman"),
                "pathway_direction_agree_frac": float(np.mean(agree)) if len(agree) else np.nan,
            })

        rows.append(row)

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
    p.add_argument("--out_prefix", default="fair_baseline")
    p.add_argument("--seed", type=int, default=17)

    # accepted for compatibility with older slurm calls, ignored
    p.add_argument("--single_cycle_run_dir", default=None)
    p.add_argument("--selection_rule", default=None)
    p.add_argument("--progress_threshold", default=None)
    p.add_argument("--matched_progress_tolerance", default=None)
    p.add_argument("--budget", default=None)
    p.add_argument("--n_tumor_clusters", default=None)
    p.add_argument("--interp_alphas", nargs="*", default=None)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.outdir is None:
        args.outdir = os.path.join(args.run_dir, "fair_baseline_eval")
    os.makedirs(args.outdir, exist_ok=True)

    funcs = import_project_functions(args.repo_dir)

    print("[load baseline pools]")
    Z_seed, Z_final, meta, manifest = load_baseline_pools(args.run_dir)
    meta = meta.copy()
    if "candidate_global_idx" not in meta.columns:
        meta.insert(0, "candidate_global_idx", np.arange(len(meta), dtype=int))

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

    if len(train_h) == 0 or len(train_t) == 0:
        raise RuntimeError("Need healthy and tumor cells in train for fair classifier.")
    if len(test_h) == 0 or len(test_t) == 0:
        raise RuntimeError("Need healthy and tumor cells in test for held-out DE/realism.")

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

    print("[decode train refs]")
    X_h_train = decode_project(Z_train[h_tr], funcs, vae, device, genes, target_ref, args)
    X_t_train = decode_project(Z_train[t_tr], funcs, vae, device, genes, target_ref, args)

    print("[decode test refs]")
    X_h_test = decode_project(Z_test[h_te], funcs, vae, device, genes, target_ref, args)
    X_t_test = decode_project(Z_test[t_te], funcs, vae, device, genes, target_ref, args)

    print("[fit classifier and geometry]")
    clf = fit_eval_classifier(X_h_train, X_t_train, args.eval_pcs, args.seed)
    geom = fit_geometry(X_h_train, X_t_train, Z_train[h_tr], Z_train[t_tr])

    print("[fit tumor kNN realism diagnostic]")
    k_eff = min(args.knn_k, len(X_t_test))
    nn_tumor = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn_tumor.fit(X_t_test)

    print("[decode baseline candidates]")
    X_seed = decode_project(Z_seed, funcs, vae, device, genes, target_ref, args)
    X_final = decode_project(Z_final, funcs, vae, device, genes, target_ref, args)

    print("[score candidates]")
    scores = add_candidate_metrics(meta, Z_seed, Z_final, X_seed, X_final, clf, geom, nn_tumor, args)

    print("[select per method/seed]")
    selected = select_per_method_seed(scores, args.success_proba)

    print("[summarize selected]")
    summary = summarize_selected(selected, Z_seed, Z_final, X_seed, X_final, scores, geom, rng, args)

    print("[summarize pool]")
    pool_sum = summarize_pool(scores, Z_final, X_final, geom, rng, args)

    print("[matched tumor-proba bins]")
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]
    bin_df = matched_bins(scores, bins)

    print("[DE/pathway]")
    gmt_sets = load_gmt(args.pathway_gmt, genes)
    de_df = de_pathway_summary(selected, X_seed, X_final, X_h_test, X_t_test, genes, gmt_sets)

    # Merge core summary with DE/pathway summary for easy table use.
    summary_merged = summary.merge(de_df, on="method", how="left").merge(pool_sum, on="method", how="left", suffixes=("", "_pool"))

    prefix = args.out_prefix
    paths = {
        "summary": os.path.join(args.outdir, f"{prefix}_summary.csv"),
        "selected": os.path.join(args.outdir, f"{prefix}_selected_per_seed.csv"),
        "pool": os.path.join(args.outdir, f"{prefix}_pool_summary.csv"),
        "scores": os.path.join(args.outdir, f"{prefix}_candidate_scores.csv"),
        "bins": os.path.join(args.outdir, f"{prefix}_matched_bins.csv"),
        "de_pathway": os.path.join(args.outdir, f"{prefix}_de_pathway_summary.csv"),
        "meta": os.path.join(args.outdir, f"{prefix}_meta.json"),
    }

    summary_merged.to_csv(paths["summary"], index=False)
    selected.to_csv(paths["selected"], index=False)
    pool_sum.to_csv(paths["pool"], index=False)
    scores.to_csv(paths["scores"], index=False)
    bin_df.to_csv(paths["bins"], index=False)
    de_df.to_csv(paths["de_pathway"], index=False)

    meta_json = {
        "task": "fair seed-initialized baseline comparison",
        "selection_rule": f"tumor_proba >= {args.success_proba} then min id_dist_expr else max tumor_logit",
        "main_baselines": sorted(scores["method"].unique().tolist()),
        "classifier": "PCA + logistic regression trained once on decoded TRAIN healthy vs tumor refs; applied to all methods equally",
        "reference_de": "held-out TEST tumor vs healthy refs",
        "pathway_gmt": args.pathway_gmt,
        "manifest": manifest,
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta_json, f, indent=2)

    print("\n================ DONE ================")
    for k, v in paths.items():
        print(f"{k}: {v}")

    show_cols = [
        "method",
        "success_frac_proba_ge_threshold",
        "fallback_frac",
        "mean_tumor_proba",
        "mean_id_dist_expr",
        "mean_resid_id_dist_expr",
        "resid_expr_actual_over_perm_mean",
        "cloud_resid_expr_within_over_between_mean",
        "de_pearson_all",
        "pathway_direction_agree_frac",
    ]
    show_cols = [c for c in show_cols if c in summary_merged.columns]
    print("\n[summary preview]")
    print(summary_merged[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
