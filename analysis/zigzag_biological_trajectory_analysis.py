#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zigzag_biological_trajectory_analysis.py

Biological trajectory and candidate-mode analysis for CancerZigZag.

This analysis asks what biological programs ZigZag explores from individual
healthy-like seeds. It is meant to add the missing biological case-study layer:

1) Selected ZigZag trajectories:
   seed -> intermediate states -> selected tumor-like candidate
   with tumor scores, pathway scores, gene scores along the trajectory.

2) Successful candidate modes per seed:
   all successful ZigZag candidates per seed are clustered into candidate modes,
   and each mode is described by enriched/shifted pathway programs.

Required previous outputs:
  RUN_DIR/trajectories_latent_all_reps.npz
  RUN_DIR/baseline_pools/baseline_pools_latent.npz
  RUN_DIR/baseline_pools/baseline_pools_meta.csv
  RUN_DIR/zigzag_falsification_eval/candidate_scores.csv
  RUN_DIR/zigzag_falsification_eval/selected_per_seed.csv
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
from sklearn.cluster import KMeans


# -----------------------------
# Project imports
# -----------------------------

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


# -----------------------------
# Generic helpers
# -----------------------------

def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


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


def unit_axis(a, b):
    v = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Cannot compute axis from identical centroids.")
    return (v / n).astype(np.float32)


# -----------------------------
# Loading / decoding
# -----------------------------

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


def load_trajectory_pool(run_dir):
    path = Path(run_dir) / "trajectories_latent_all_reps.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    return {
        "Z_traj": z["Z_traj"].astype(np.float32),
        "cell_idx": z["cell_idx"].astype(int),
        "rep_id": z["rep_id"].astype(int),
    }


def load_baseline_pool(run_dir):
    pool_dir = Path(run_dir) / "baseline_pools"
    npz_path = pool_dir / "baseline_pools_latent.npz"
    meta_path = pool_dir / "baseline_pools_meta.csv"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)
    if "candidate_global_idx" not in meta.columns:
        meta.insert(0, "candidate_global_idx", np.arange(len(meta), dtype=int))
    return z["Z_seed"].astype(np.float32), z["Z_final"].astype(np.float32), meta


def load_eval_tables(run_dir):
    eval_dir = Path(run_dir) / "zigzag_falsification_eval"
    scores_path = eval_dir / "candidate_scores.csv"
    selected_path = eval_dir / "selected_per_seed.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}. Run zigzag_fair_falsification_eval.py first.")
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing {selected_path}. Run zigzag_fair_falsification_eval.py first.")
    return pd.read_csv(scores_path), pd.read_csv(selected_path)


# -----------------------------
# Tumor scoring / pathway scoring
# -----------------------------

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


def fit_geometry(X_h, X_t):
    expr_axis = unit_axis(X_h.mean(axis=0), X_t.mean(axis=0))
    expr_center = ((X_h.mean(axis=0) + X_t.mean(axis=0)) / 2.0).astype(np.float32)
    proj_h = (X_h - expr_center[None, :]) @ expr_axis
    proj_t = (X_t - expr_center[None, :]) @ expr_axis
    h_med = float(np.median(proj_h))
    t_med = float(np.median(proj_t))
    denom = t_med - h_med
    if abs(denom) < 1e-9:
        denom = 1.0
    return {"expr_axis": expr_axis, "expr_center": expr_center, "h_med": h_med, "denom": denom}


def score_tumor(clf, geom, X):
    proba = clf.predict_proba(X)[:, 1].astype(float)
    logit = clf.decision_function(X).astype(float)
    proj = (X - geom["expr_center"][None, :]) @ geom["expr_axis"]
    progress = (proj - geom["h_med"]) / (geom["denom"] + 1e-9)
    return logit, proba, progress.astype(float)


def load_gmt(path, genes, include_keywords=None, min_genes=5, max_sets=None):
    if path is None or not os.path.exists(path):
        return {}
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    include_keywords = [k.upper() for k in include_keywords] if include_keywords else None
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            if include_keywords is not None:
                name_u = name.upper()
                if not any(k in name_u for k in include_keywords):
                    continue
            idx = [gene_to_idx[g] for g in parts[2:] if g in gene_to_idx]
            if len(idx) >= min_genes:
                sets[name] = np.asarray(idx, dtype=int)
    if max_sets is not None and len(sets) > max_sets:
        keys = list(sets.keys())[:max_sets]
        sets = {k: sets[k] for k in keys}
    return sets


def pathway_scores(X, ref_mean, ref_std, gmt_sets):
    Xz = (X - ref_mean[None, :]) / (ref_std[None, :] + 1e-6)
    names = []
    mat = []
    for name, idx in gmt_sets.items():
        names.append(name)
        mat.append(Xz[:, idx].mean(axis=1))
    if len(mat) == 0:
        return [], np.zeros((X.shape[0], 0), dtype=np.float32)
    return names, np.vstack(mat).T.astype(np.float32)


def select_top_genes(real_de, genes, n=50, marker_genes=None):
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    selected = []
    if marker_genes:
        for g in marker_genes:
            if g in gene_to_idx and g not in selected:
                selected.append(g)
    order = np.argsort(-np.abs(real_de))
    for i in order:
        g = genes[int(i)]
        if g not in selected:
            selected.append(g)
        if len(selected) >= n:
            break
    idx = np.asarray([gene_to_idx[g] for g in selected], dtype=int)
    return selected, idx


# -----------------------------
# Selected trajectory analysis
# -----------------------------

def map_selected_to_traj(selected, traj):
    key_to_idx = {f"{int(c)}__{int(r)}": i for i, (c, r) in enumerate(zip(traj["cell_idx"], traj["rep_id"]))}
    rows = []
    for _, r in selected.iterrows():
        if str(r.get("method", "")) != "zigzag_pool":
            continue
        key = f"{int(r['cell_idx'])}__{int(r['rep_id'])}"
        if key in key_to_idx:
            row = r.to_dict()
            row["traj_idx"] = int(key_to_idx[key])
            rows.append(row)
    return pd.DataFrame(rows)


def build_trajectory_state_table(selected_map, traj, X_traj, clf, geom):
    rows = []
    offset = 0
    for _, r in selected_map.iterrows():
        ti = int(r["traj_idx"])
        n_states = traj["Z_traj"][ti].shape[0]
        X = X_traj[offset:offset + n_states]
        logit, proba, progress = score_tumor(clf, geom, X)
        trajectory_id = f"{ti}__{int(r['cell_idx'])}__{int(r['rep_id'])}"
        for s in range(n_states):
            rows.append({
                "trajectory_id": trajectory_id,
                "traj_idx": ti,
                "seed_order": int(r["seed_order"]),
                "cell_idx": int(r["cell_idx"]),
                "rep_id": int(r["rep_id"]),
                "candidate_global_idx": int(r["candidate_global_idx"]),
                "state_idx": int(s),
                "state_frac": float(s / max(n_states - 1, 1)),
                "tumor_logit": float(logit[s]),
                "tumor_proba": float(proba[s]),
                "progress01_geom": float(progress[s]),
            })
        offset += n_states
    return pd.DataFrame(rows)


def trajectory_pathway_long(selected_map, traj, pathway_names, pathway_mat, real_pathway_delta):
    rows = []
    offset = 0
    for _, r in selected_map.iterrows():
        ti = int(r["traj_idx"])
        n_states = traj["Z_traj"][ti].shape[0]
        M = pathway_mat[offset:offset + n_states, :]
        seed_scores = M[0, :]
        final_scores = M[-1, :]
        trajectory_id = f"{ti}__{int(r['cell_idx'])}__{int(r['rep_id'])}"
        for j, name in enumerate(pathway_names):
            rd = real_pathway_delta[j]
            for s in range(n_states):
                delta = float(M[s, j] - seed_scores[j])
                rows.append({
                    "trajectory_id": trajectory_id,
                    "seed_order": int(r["seed_order"]),
                    "cell_idx": int(r["cell_idx"]),
                    "rep_id": int(r["rep_id"]),
                    "state_idx": int(s),
                    "state_frac": float(s / max(n_states - 1, 1)),
                    "pathway": name,
                    "score": float(M[s, j]),
                    "delta_from_seed": delta,
                    "final_delta_from_seed": float(final_scores[j] - seed_scores[j]),
                    "real_tumor_delta": float(rd),
                    "direction_matches_real": int(np.sign(delta) == np.sign(rd)) if abs(delta) > 1e-12 and abs(rd) > 1e-12 else np.nan,
                })
        offset += n_states
    return pd.DataFrame(rows)


def summarize_trajectory_pathways(pathway_long, top_n=10):
    rows = []
    if len(pathway_long) == 0:
        return pd.DataFrame()
    final = pathway_long.sort_values("state_idx").groupby(["trajectory_id", "pathway"], as_index=False).tail(1)
    final = final.copy()
    final["abs_final_delta"] = final["final_delta_from_seed"].abs()
    for tid, g in final.groupby("trajectory_id", sort=True):
        gg = g.sort_values("abs_final_delta", ascending=False).head(top_n)
        for rank, (_, r) in enumerate(gg.iterrows(), start=1):
            rows.append({
                "trajectory_id": tid,
                "rank": rank,
                "pathway": r["pathway"],
                "final_delta_from_seed": float(r["final_delta_from_seed"]),
                "real_tumor_delta": float(r["real_tumor_delta"]),
                "direction_matches_real": r["direction_matches_real"],
                "abs_final_delta": float(r["abs_final_delta"]),
            })
    return pd.DataFrame(rows)


def pathway_monotonicity(pathway_long):
    rows = []
    if len(pathway_long) == 0:
        return pd.DataFrame()
    for (tid, pw), g in pathway_long.groupby(["trajectory_id", "pathway"], sort=True):
        g = g.sort_values("state_idx")
        rho = safe_corr(g["state_frac"].values, g["score"].values, method="spearman")
        rows.append({
            "trajectory_id": tid,
            "pathway": pw,
            "spearman_state_score": rho,
            "final_delta_from_seed": float(g["final_delta_from_seed"].iloc[-1]),
            "real_tumor_delta": float(g["real_tumor_delta"].iloc[-1]),
            "direction_matches_real_final": g["direction_matches_real"].iloc[-1],
        })
    return pd.DataFrame(rows)


def trajectory_gene_long(selected_map, traj, X_traj, gene_names, gene_idx, ref_mean, ref_std, real_de):
    Xz = (X_traj - ref_mean[None, :]) / (ref_std[None, :] + 1e-6)
    G_all = Xz[:, gene_idx]
    rows = []
    offset = 0
    for _, r in selected_map.iterrows():
        ti = int(r["traj_idx"])
        n_states = traj["Z_traj"][ti].shape[0]
        G = G_all[offset:offset + n_states, :]
        seed_z = G[0, :]
        final_z = G[-1, :]
        trajectory_id = f"{ti}__{int(r['cell_idx'])}__{int(r['rep_id'])}"
        for j, gene in enumerate(gene_names):
            rd = real_de[gene_idx[j]]
            for s in range(n_states):
                delta = float(G[s, j] - seed_z[j])
                rows.append({
                    "trajectory_id": trajectory_id,
                    "seed_order": int(r["seed_order"]),
                    "cell_idx": int(r["cell_idx"]),
                    "rep_id": int(r["rep_id"]),
                    "state_idx": int(s),
                    "state_frac": float(s / max(n_states - 1, 1)),
                    "gene": gene,
                    "zscore": float(G[s, j]),
                    "delta_from_seed": delta,
                    "final_delta_from_seed": float(final_z[j] - seed_z[j]),
                    "real_tumor_delta": float(rd),
                    "direction_matches_real": int(np.sign(delta) == np.sign(rd)) if abs(delta) > 1e-12 and abs(rd) > 1e-12 else np.nan,
                })
        offset += n_states
    return pd.DataFrame(rows)


def summarize_trajectory_genes(gene_long, top_n=20):
    rows = []
    if len(gene_long) == 0:
        return pd.DataFrame()
    final = gene_long.sort_values("state_idx").groupby(["trajectory_id", "gene"], as_index=False).tail(1).copy()
    final["abs_final_delta"] = final["final_delta_from_seed"].abs()
    for tid, g in final.groupby("trajectory_id", sort=True):
        gg = g.sort_values("abs_final_delta", ascending=False).head(top_n)
        for rank, (_, r) in enumerate(gg.iterrows(), start=1):
            rows.append({
                "trajectory_id": tid,
                "rank": rank,
                "gene": r["gene"],
                "final_delta_from_seed": float(r["final_delta_from_seed"]),
                "real_tumor_delta": float(r["real_tumor_delta"]),
                "direction_matches_real": r["direction_matches_real"],
                "abs_final_delta": float(r["abs_final_delta"]),
            })
    return pd.DataFrame(rows)


# -----------------------------
# Successful candidate modes
# -----------------------------

def candidate_modes(scores, Z_final, pathway_names, pathway_scores_all, real_pathway_delta, args):
    zig = scores[scores["method"].astype(str) == "zigzag_pool"].copy()
    success = zig[zig["tumor_proba"] >= args.success_proba].copy()
    if len(success) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    candidate_rows, pathway_rows, summary_rows = [], [], []
    for seed, g in success.groupby("seed_order", sort=True):
        idx = g["candidate_global_idx"].astype(int).values
        n = len(idx)
        if n == 1:
            labels = np.zeros(1, dtype=int)
        else:
            Xclust = pathway_scores_all[idx] if pathway_scores_all.shape[1] else Z_final[idx]
            k = int(min(args.max_modes_per_seed, n))
            k = max(1, min(k, len(np.unique(np.round(Xclust, 6), axis=0))))
            labels = np.zeros(n, dtype=int) if k == 1 else KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(Xclust)
        g = g.copy()
        g["mode_id"] = labels
        candidate_rows.append(g)
        for mode_id, gm in g.groupby("mode_id", sort=True):
            midx = gm["candidate_global_idx"].astype(int).values
            mpath = pathway_scores_all[midx, :] if pathway_scores_all.shape[1] else np.zeros((len(midx), 0))
            summary_rows.append({
                "seed_order": int(seed),
                "mode_id": int(mode_id),
                "n_candidates_in_mode": int(len(gm)),
                "mean_tumor_proba": safe_mean(gm["tumor_proba"]),
                "median_tumor_proba": float(np.median(pd.to_numeric(gm["tumor_proba"], errors="coerce"))),
                "mean_id_dist_expr": safe_mean(gm["id_dist_expr"]) if "id_dist_expr" in gm else np.nan,
                "mean_resid_id_dist_expr": safe_mean(gm["resid_id_dist_expr"]) if "resid_id_dist_expr" in gm else np.nan,
                "mean_progress01_geom": safe_mean(gm["progress01_geom"]) if "progress01_geom" in gm else np.nan,
            })
            if pathway_scores_all.shape[1]:
                mean_scores = mpath.mean(axis=0)
                order = np.argsort(-np.abs(mean_scores))[:args.top_pathways_per_mode]
                for rank, j in enumerate(order, start=1):
                    pathway_rows.append({
                        "seed_order": int(seed),
                        "mode_id": int(mode_id),
                        "rank": int(rank),
                        "pathway": pathway_names[j],
                        "mode_mean_pathway_score": float(mean_scores[j]),
                        "real_tumor_delta": float(real_pathway_delta[j]),
                        "direction_matches_real": int(np.sign(mean_scores[j]) == np.sign(real_pathway_delta[j])) if abs(mean_scores[j]) > 1e-12 and abs(real_pathway_delta[j]) > 1e-12 else np.nan,
                        "n_candidates_in_mode": int(len(gm)),
                        "mean_tumor_proba": safe_mean(gm["tumor_proba"]),
                    })
    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    return candidates, pd.DataFrame(pathway_rows), pd.DataFrame(summary_rows)


def seed_program_summary(mode_summary, mode_pathways):
    rows = []
    if len(mode_summary) == 0:
        return pd.DataFrame()
    for seed, g in mode_summary.groupby("seed_order", sort=True):
        mp = mode_pathways[mode_pathways["seed_order"] == seed] if len(mode_pathways) else pd.DataFrame()
        top_terms = []
        if len(mp):
            top = mp.sort_values(["mode_id", "rank"]).groupby("mode_id").head(3)
            for _, r in top.iterrows():
                top_terms.append(f"mode{int(r['mode_id'])}:{r['pathway']}")
        rows.append({
            "seed_order": int(seed),
            "n_success_modes": int(g["mode_id"].nunique()),
            "n_success_candidates": int(g["n_candidates_in_mode"].sum()),
            "mean_mode_tumor_proba": safe_mean(g["mean_tumor_proba"]),
            "top_mode_pathways": "; ".join(top_terms[:12]),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

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
    p.add_argument("--pathway_gmt", required=True)
    p.add_argument("--pathway_keywords", nargs="*", default=None)
    p.add_argument("--max_pathways", type=int, default=None)
    p.add_argument("--top_pathways_per_trajectory", type=int, default=12)
    p.add_argument("--top_pathways_per_mode", type=int, default=8)
    p.add_argument("--top_genes", type=int, default=60)
    p.add_argument("--marker_genes", nargs="*", default=None)
    p.add_argument("--max_modes_per_seed", type=int, default=4)
    p.add_argument("--outdir", default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--sparsity_project", action="store_true")
    p.add_argument("--sparsity_min_detect_rate", type=float, default=0.0)
    p.add_argument("--sparsity_max_detect_rate", type=float, default=1.0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.outdir is None:
        args.outdir = os.path.join(args.run_dir, "biological_trajectory_analysis")
    os.makedirs(args.outdir, exist_ok=True)
    funcs = import_project_functions(args.repo_dir)

    print("[load tables]")
    traj = load_trajectory_pool(args.run_dir)
    Z_pool_seed, Z_pool_final, pool_meta = load_baseline_pool(args.run_dir)
    scores, selected = load_eval_tables(args.run_dir)

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
    ref_all = np.vstack([X_h_test, X_t_test]).astype(np.float32)
    ref_mean = ref_all.mean(axis=0).astype(np.float32)
    ref_std = ref_all.std(axis=0).astype(np.float32)
    real_de = (X_t_test.mean(axis=0) - X_h_test.mean(axis=0)).astype(np.float32)

    print("[fit common tumor scorer]")
    clf = fit_classifier(X_h_train, X_t_train, args.eval_pcs, args.seed)
    geom = fit_geometry(X_h_train, X_t_train)

    print("[load pathways / genes]")
    gmt_sets = load_gmt(args.pathway_gmt, genes, include_keywords=args.pathway_keywords, max_sets=args.max_pathways)
    if len(gmt_sets) == 0:
        raise RuntimeError("No pathway gene sets loaded. Check --pathway_gmt / --pathway_keywords.")
    pathway_names, h_path = pathway_scores(X_h_test, ref_mean, ref_std, gmt_sets)
    _, t_path = pathway_scores(X_t_test, ref_mean, ref_std, gmt_sets)
    real_pathway_delta = (t_path.mean(axis=0) - h_path.mean(axis=0)).astype(np.float32)
    top_gene_names, top_gene_idx = select_top_genes(real_de, genes, n=args.top_genes, marker_genes=args.marker_genes)

    print("[map selected trajectories]")
    selected_map = map_selected_to_traj(selected, traj)
    if len(selected_map) == 0:
        raise RuntimeError("No selected zigzag trajectories could be mapped to trajectory pool.")

    print(f"[decode selected trajectories] n={len(selected_map)}")
    Z_traj_list, traj_lengths = [], []
    for _, r in selected_map.iterrows():
        Zp = traj["Z_traj"][int(r["traj_idx"])]
        Z_traj_list.append(Zp)
        traj_lengths.append(Zp.shape[0])
    Z_traj_flat = np.vstack(Z_traj_list).astype(np.float32)
    X_traj = decode_project(Z_traj_flat, funcs, vae, device, genes, target_ref, args)

    print("[trajectory tumor scores]")
    traj_state_scores = build_trajectory_state_table(selected_map, traj, X_traj, clf, geom)

    print("[trajectory pathway scores]")
    _, traj_path_mat = pathway_scores(X_traj, ref_mean, ref_std, gmt_sets)
    traj_path_long = trajectory_pathway_long(selected_map, traj, pathway_names, traj_path_mat, real_pathway_delta)
    traj_path_summary = summarize_trajectory_pathways(traj_path_long, top_n=args.top_pathways_per_trajectory)
    traj_path_mono = pathway_monotonicity(traj_path_long)

    print("[trajectory gene scores]")
    traj_gene_long = trajectory_gene_long(selected_map, traj, X_traj, top_gene_names, top_gene_idx, ref_mean, ref_std, real_de)
    traj_gene_summary = summarize_trajectory_genes(traj_gene_long, top_n=min(20, args.top_genes))

    print("[decode all baseline pool candidates for successful candidate modes]")
    X_pool_final = decode_project(Z_pool_final, funcs, vae, device, genes, target_ref, args)
    _, pool_path_mat = pathway_scores(X_pool_final, ref_mean, ref_std, gmt_sets)

    print("[successful candidate modes]")
    mode_candidates, mode_pathways, mode_summary = candidate_modes(
        scores=scores,
        Z_final=Z_pool_final,
        pathway_names=pathway_names,
        pathway_scores_all=pool_path_mat,
        real_pathway_delta=real_pathway_delta,
        args=args,
    )
    seed_summary = seed_program_summary(mode_summary, mode_pathways)

    selected_program_rows = []
    for tid, g in traj_path_summary.groupby("trajectory_id", sort=True):
        top = g.sort_values("rank").head(5)
        selected_program_rows.append({
            "trajectory_id": tid,
            "top5_pathways": "; ".join(top["pathway"].astype(str).tolist()),
            "top5_direction_match_frac": safe_mean(top["direction_matches_real"]),
            "top5_mean_abs_final_delta": safe_mean(top["abs_final_delta"]),
        })
    selected_program_summary = pd.DataFrame(selected_program_rows)

    paths = {
        "trajectory_state_tumor_scores": os.path.join(args.outdir, "trajectory_state_tumor_scores.csv"),
        "trajectory_pathway_scores_long": os.path.join(args.outdir, "trajectory_pathway_scores_long.csv"),
        "trajectory_pathway_summary": os.path.join(args.outdir, "trajectory_pathway_summary.csv"),
        "trajectory_pathway_monotonicity": os.path.join(args.outdir, "trajectory_pathway_monotonicity.csv"),
        "trajectory_gene_scores_long": os.path.join(args.outdir, "trajectory_gene_scores_long.csv"),
        "trajectory_gene_summary": os.path.join(args.outdir, "trajectory_gene_summary.csv"),
        "selected_trajectory_program_summary": os.path.join(args.outdir, "selected_trajectory_program_summary.csv"),
        "successful_candidate_modes": os.path.join(args.outdir, "successful_candidate_modes.csv"),
        "successful_candidate_mode_pathways": os.path.join(args.outdir, "successful_candidate_mode_pathways.csv"),
        "successful_candidate_mode_summary": os.path.join(args.outdir, "successful_candidate_mode_summary.csv"),
        "seed_program_summary": os.path.join(args.outdir, "seed_program_summary.csv"),
        "meta": os.path.join(args.outdir, "biological_trajectory_meta.json"),
    }
    traj_state_scores.to_csv(paths["trajectory_state_tumor_scores"], index=False)
    traj_path_long.to_csv(paths["trajectory_pathway_scores_long"], index=False)
    traj_path_summary.to_csv(paths["trajectory_pathway_summary"], index=False)
    traj_path_mono.to_csv(paths["trajectory_pathway_monotonicity"], index=False)
    traj_gene_long.to_csv(paths["trajectory_gene_scores_long"], index=False)
    traj_gene_summary.to_csv(paths["trajectory_gene_summary"], index=False)
    selected_program_summary.to_csv(paths["selected_trajectory_program_summary"], index=False)
    mode_candidates.to_csv(paths["successful_candidate_modes"], index=False)
    mode_pathways.to_csv(paths["successful_candidate_mode_pathways"], index=False)
    mode_summary.to_csv(paths["successful_candidate_mode_summary"], index=False)
    seed_summary.to_csv(paths["seed_program_summary"], index=False)

    meta = {
        "analysis": "biological trajectory and candidate-mode analysis",
        "run_dir": args.run_dir,
        "n_selected_trajectories": int(len(selected_map)),
        "trajectory_lengths": [int(x) for x in traj_lengths],
        "n_pathways": int(len(pathway_names)),
        "pathway_names": pathway_names,
        "top_gene_names": top_gene_names,
        "scoring": {
            "pathway_score": "mean z-scored decoded expression over pathway genes; reference = held-out test healthy+tumor decoded cells",
            "real_pathway_delta": "mean pathway score in held-out test tumor minus held-out test healthy",
            "tumor_score": "common PCA+logistic regression trained on decoded train healthy vs tumor",
        },
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    print("\n================ DONE ================")
    for k, v in paths.items():
        print(f"{k}: {v}")
    print("\n[selected trajectory program preview]")
    if len(selected_program_summary):
        print(selected_program_summary.head(20).to_string(index=False))
    print("\n[seed program summary preview]")
    if len(seed_summary):
        print(seed_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
