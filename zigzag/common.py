#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nature-Methods-ready single-file script (cleaned + critical fixes):
- Runtime manifest (versions/hardware/args)
- Optional VAE gene_order.tsv alignment (prevents silent gene-order bugs)
- Reference split (cellwise OR groupwise via --split_col) to reduce eval leakage
- DE prep fixed (NO sc.pp.scale for rank_genes_groups)
- Robust UMAP saving (no scanpy save-to-figures hack)
- Bootstrap CIs for key decoded metrics
- decode_frac=0 no longer aborts run aggregation (latent-only eval still saved)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import os
import sys
import json
import glob
import time
import hashlib
import argparse
import platform
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import sklearn
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_classifier_and_diffusion,
    classifier_and_diffusion_defaults,
)
from guided_diffusion.cell_datasets_loader import load_VAE


# ============================================================
# -------------------- IO / REPRO HELPERS --------------------
# ============================================================


import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from guided_diffusion import dist_util
from guided_diffusion.cell_datasets_loader import load_VAE as _load_VAE


def _latent_centroid_score(Z_src: np.ndarray, Z_tgt: np.ndarray, Z_edit: np.ndarray) -> float:
    """
    Score in [0..1] (roughly): how much closer edits are to target centroid than source centroid.
    >0.5 means edits closer to target than source on average.
    """
    Z_src = np.asarray(Z_src)
    Z_tgt = np.asarray(Z_tgt)
    Z_edit = np.asarray(Z_edit)

    c_src = Z_src.mean(axis=0, keepdims=True)
    c_tgt = Z_tgt.mean(axis=0, keepdims=True)

    d_src = np.linalg.norm(Z_edit - c_src, axis=1)
    d_tgt = np.linalg.norm(Z_edit - c_tgt, axis=1)

    # convert to a bounded score: fraction where target is closer
    return float(np.mean(d_tgt < d_src))


def load_latents_from_h5ad(
    h5ad_path: str,
    vae_ckpt: str,
    hidden_dim: int = 128,
    *,
    # preprocessing (must match what you used for training/inference)
    normalize_total: bool = True,
    target_sum: float = 1e4,
    log1p: bool = True,
    # optional: if your counts are stored in a layer (recommended)
    layer: Optional[str] = None,   # e.g. "counts"
    # optional: cell filtering (OFF by default)
    filter_cells_min_genes: Optional[int] = None,
    # encoding
    encode_batch: int = 4096,
    return_obs: bool = True,
    return_var_names: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a .h5ad, apply the SAME scaling as your training loader, encode ALL cells with the VAE,
    and return latents Z as a single numpy array.

    Returns:
      Z: (N, latent_dim) float32
      info: dict with metadata (obs/var, shapes, preprocessing flags)

    Notes:
      - No gene filtering/reordering here. Your h5ad MUST already be aligned to gene_order.tsv
        and have the exact gene order the VAE expects.
      - Does NOT shuffle, does NOT drop_last. Deterministic given the file contents.
    """
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"h5ad_path not found: {h5ad_path}")
    if not os.path.exists(vae_ckpt):
        raise FileNotFoundError(f"vae_ckpt not found: {vae_ckpt}")

    # --- read ---
    adata = sc.read_h5ad(h5ad_path)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise RuntimeError(f"Empty AnnData: {adata.shape} | file={h5ad_path}")

    # --- optional cell filtering (no gene filtering!) ---
    if filter_cells_min_genes is not None and filter_cells_min_genes > 0:
        try:
            sc.pp.filter_cells(adata, min_genes=int(filter_cells_min_genes))
        except Exception as e:
            raise RuntimeError(f"filter_cells failed: {e}")
    if adata.n_obs == 0:
        raise RuntimeError(
            f"All cells filtered out (min_genes={filter_cells_min_genes}). "
            f"Disable filtering or lower threshold."
        )

    # --- choose matrix ---
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"Requested layer='{layer}' not found. Available layers: {list(adata.layers.keys())}"
            )
        X_in = adata.layers[layer]
    else:
        X_in = adata.X

    # Make a lightweight working AnnData for preprocessing (keeps obs/var)
    A = adata.copy()
    A.X = X_in

    # --- preprocessing: match training loader ---
    # These do not change dimensionality.
    if normalize_total:
        sc.pp.normalize_total(A, target_sum=float(target_sum))
    if log1p:
        sc.pp.log1p(A)

    # --- to dense float32 for VAE ---
    X = A.X
    if sp.issparse(X):
        X = X.tocsr()
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32, order="C")

    if X.shape[0] == 0:
        raise RuntimeError("X has 0 rows after preprocessing.")

    # --- load VAE on correct device ---
    device = dist_util.dev()  # cuda if available
    n_genes = int(X.shape[1])

    vae = _load_VAE(vae_ckpt, num_gene=n_genes, hidden_dim=int(hidden_dim)).eval().to(device)

    # --- encode in batches ---
    Z_chunks = []
    with torch.no_grad():
        N = X.shape[0]
        for start in range(0, N, int(encode_batch)):
            sl = slice(start, min(N, start + int(encode_batch)))
            x_t = torch.from_numpy(X[sl]).float().to(device)
            if x_t.shape[0] == 0:
                continue
            z_t = vae(x_t, return_latent=True).detach().cpu().numpy()
            Z_chunks.append(z_t.astype(np.float32, copy=False))

    if not Z_chunks:
        raise RuntimeError(
            f"No chunks encoded. N={X.shape[0]}, encode_batch={encode_batch}, file={h5ad_path}"
        )

    Z = np.concatenate(Z_chunks, axis=0).astype(np.float32, copy=False)
    if Z.shape[0] != X.shape[0]:
        raise RuntimeError(f"Row mismatch: Z has {Z.shape[0]} rows but X has {X.shape[0]} rows")

    info: Dict[str, Any] = {
        "h5ad_path": h5ad_path,
        "vae_ckpt": vae_ckpt,
        "n_cells": int(Z.shape[0]),
        "n_genes": int(n_genes),
        "latent_dim": int(Z.shape[1]),
        "preprocess": {
            "layer": layer,
            "normalize_total": bool(normalize_total),
            "target_sum": float(target_sum),
            "log1p": bool(log1p),
            "filter_cells_min_genes": int(filter_cells_min_genes) if filter_cells_min_genes else None,
        },
    }

    if return_obs:
        # Keep it light; user can merge later
        info["obs"] = A.obs.copy()
    if return_var_names:
        info["var_names"] = A.var_names.astype(str).to_list()

    if verbose:
        print(
            f"[load_latents_from_h5ad] {os.path.basename(h5ad_path)} "
            f"-> X {A.n_obs}x{A.n_vars} -> Z {Z.shape} | "
            f"norm={normalize_total} log1p={log1p} layer={layer}"
        )

    return Z, info

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _square_axes(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    rng = max(xmax - xmin, ymax - ymin)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    ax.set_xlim(cx - rng / 2, cx + rng / 2)
    ax.set_ylim(cy - rng / 2, cy + rng / 2)
    ax.set_aspect("equal", adjustable="box")

def make_masks(adata_obj, label_col):
    labels_raw = adata_obj.obs[label_col]
    labels_str = labels_raw.astype(str)
    unique_vals = sorted(pd.unique(labels_str))

    if any("tumor" in v.lower() for v in unique_vals) or any("normal" in v.lower() for v in unique_vals) or any("healthy" in v.lower() for v in unique_vals):
        status = labels_str.str.lower()
        mask_h = status.str.contains("normal") | status.str.contains("healthy")
        mask_t = status.str.contains("tumor")
        cls0_name, cls1_name = "healthy", "tumor"
    elif len(unique_vals) == 2:
        v0, v1 = unique_vals[0], unique_vals[1]
        mask_h = (labels_str == v0)
        mask_t = (labels_str == v1)
        cls0_name, cls1_name = str(v0), str(v1)
    else:
        raise ValueError(f"Label '{label_col}' has {len(unique_vals)} unique values. Got: {unique_vals}")

    if mask_h.sum() == 0 or mask_t.sum() == 0:
        raise ValueError(f"Need both classes in '{label_col}'. Got healthy={mask_h.sum()} tumor={mask_t.sum()}")

    return mask_h, mask_t, {"unique_vals": unique_vals, "class0": cls0_name, "class1": cls1_name,
                            "n_class0": int(mask_h.sum()), "n_class1": int(mask_t.sum())}

def plot_latent_trajectories_pca(
    Z_ref_h, Z_ref_t, traj_npz, out_png,
    max_ref=5000,
    ref_s=6, ref_alpha=0.12,
    line_w=2.2,
    start_s=45, end_s=55,
    seed_colors=("tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"),
):
    dat = np.load(traj_npz)
    Z_traj = dat["Z_traj"]  # (R, K, D)
    R, K, D = Z_traj.shape

    Zh = np.asarray(Z_ref_h, dtype=np.float32)
    Zt = np.asarray(Z_ref_t, dtype=np.float32)

    # subsample refs (deterministic)
    rng0 = np.random.default_rng(0)
    if Zh.shape[0] > max_ref:
        Zh = Zh[rng0.choice(Zh.shape[0], size=max_ref, replace=False)]
    if Zt.shape[0] > max_ref:
        Zt = Zt[rng0.choice(Zt.shape[0], size=max_ref, replace=False)]

    Z_flat = Z_traj.reshape(R * K, D).astype(np.float32)

    # sanitize (critical for "empty plot" issues)
    Z_all = np.vstack([Zh, Zt, Z_flat])
    Z_all = np.nan_to_num(Z_all, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(Z_all)

    Ph = P[:Zh.shape[0]]
    Pt = P[Zh.shape[0]:Zh.shape[0] + Zt.shape[0]]
    Ptr = P[Zh.shape[0] + Zt.shape[0]:].reshape(R, K, 2)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    # >>> SWAP COLORS: healthy=orange, tumor=blue
    ax.scatter(Ph[:, 0], Ph[:, 1], s=ref_s, alpha=ref_alpha, color="tab:orange",
               label="healthy reference", rasterized=True)
    ax.scatter(Pt[:, 0], Pt[:, 1], s=ref_s, alpha=ref_alpha, color="tab:blue",
               label="tumor reference", rasterized=True)

    # trajectories
    for k in range(K):
        c = seed_colors[k % len(seed_colors)]
        traj = Ptr[:, k, :]
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=line_w, color=c, zorder=5)

        # small start/end markers (readable)
        ax.scatter(traj[0, 0], traj[0, 1], s=start_s, color=c, edgecolor="black",
                   linewidth=0.5, zorder=7)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=end_s, color=c, marker="X",
                   edgecolor="black", linewidth=0.6, zorder=8)

    ax.set_title("Latent trajectories (PCA2)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # equal aspect (nice) but only if limits are meaningful
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if np.isfinite([xmin, xmax, ymin, ymax]).all() and (xmax > xmin) and (ymax > ymin):
        rng = max(xmax - xmin, ymax - ymin)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        ax.set_xlim(cx - rng / 2, cx + rng / 2)
        ax.set_ylim(cy - rng / 2, cy + rng / 2)
        ax.set_aspect("equal", adjustable="box")

    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)




def plot_expression_trajectories_pca_from_latent_traj(
    vae,
    device,
    X_ref_h_dec: np.ndarray,
    X_ref_t_dec: np.ndarray,
    traj_npz: str,
    out_png: str,
    batch_size: int = 512,
    top_var_genes: int = 2000,
    max_ref: int = 8000,
    zscore: bool = True,
    ref_s: int = 6,
    ref_alpha: float = 0.12,
    line_w: float = 2.2,
    start_s: int = 45,
    end_s: int = 55,
    seed_colors=("tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"),
    # NEW:
    genes=None,                         # list of gene names (len=G). Needed if sparsity_project=True
    sparsity_project: bool = False,
    sparsity_target_adata=None,         # AnnData with real target distribution for detect rates
):
    dat = np.load(traj_npz)
    Z_traj = dat["Z_traj"]  # (R, K, D)
    R, K, D = Z_traj.shape

    # --- decode trajectory points
    Z_flat = Z_traj.reshape(R * K, D).astype(np.float32)
    X_flat = decode_latents_in_batches(vae, Z_flat, device, batch_size=batch_size).astype(np.float32)  # (R*K, G)
    X_traj = X_flat.reshape(R, K, -1).astype(np.float32)
    G = X_traj.shape[2]

    # --- subsample refs
    rng = np.random.default_rng(0)
    Xh = np.asarray(X_ref_h_dec, dtype=np.float32)
    Xt = np.asarray(X_ref_t_dec, dtype=np.float32)
    if Xh.shape[0] > max_ref:
        Xh = Xh[rng.choice(Xh.shape[0], size=max_ref, replace=False)]
    if Xt.shape[0] > max_ref:
        Xt = Xt[rng.choice(Xt.shape[0], size=max_ref, replace=False)]

    # --- sanitize (avoid NaN/Inf -> degenerate PCA)
    Xh = np.nan_to_num(Xh, nan=0.0, posinf=0.0, neginf=0.0)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    X_traj = np.nan_to_num(X_traj, nan=0.0, posinf=0.0, neginf=0.0)

    # --- OPTIONAL: apply gene-wise sparsity projection CONSISTENTLY to refs + traj
    if sparsity_project:
        if genes is None:
            raise ValueError("sparsity_project=True requires genes=list-of-gene-names.")
        if sparsity_target_adata is None:
            raise ValueError("sparsity_project=True requires sparsity_target_adata (real target AnnData).")

        # refs
        refH_ad = ad.AnnData(X=Xh.copy(), var=pd.DataFrame(index=list(genes)))
        refT_ad = ad.AnnData(X=Xt.copy(), var=pd.DataFrame(index=list(genes)))
        refH_ad = project_sparsity_gene_wise(refH_ad, sparsity_target_adata)
        refT_ad = project_sparsity_gene_wise(refT_ad, sparsity_target_adata)
        Xh = np.asarray(refH_ad.X, dtype=np.float32)
        Xt = np.asarray(refT_ad.X, dtype=np.float32)

        # traj points
        traj_ad = ad.AnnData(X=X_traj.reshape(R*K, G).copy(), var=pd.DataFrame(index=list(genes)))
        traj_ad = project_sparsity_gene_wise(traj_ad, sparsity_target_adata)
        X_traj = np.asarray(traj_ad.X, dtype=np.float32).reshape(R, K, G)

    # --- pick top-variance genes on REF ONLY (stable)
    Xref = np.vstack([Xh, Xt])  # (Nh+Nt, G)
    var = Xref.var(axis=0)

    nonzero = var > 0
    if nonzero.sum() < 10:
        raise RuntimeError(
            f"Expression PCA degenerate: only {int(nonzero.sum())} genes with nonzero variance. "
            "Check decoded refs / sparsity projection."
        )

    var2 = var.copy()
    var2[~nonzero] = -1.0
    idx_g = np.argsort(var2)[::-1][:min(top_var_genes, int(nonzero.sum()), G)]

    # subset genes
    Xh_g = Xh[:, idx_g]
    Xt_g = Xt[:, idx_g]
    Xtr_g = X_traj[:, :, idx_g].reshape(R*K, -1)

    # --- REFERENCE-FIT + PROJECT (what you want for caption)
    Xref_g = np.vstack([Xh_g, Xt_g])

    if zscore:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xref_s = scaler.fit_transform(Xref_g)
        Xtr_s = scaler.transform(Xtr_g)
    else:
        Xref_s = Xref_g
        Xtr_s = Xtr_g

    pca = PCA(n_components=2, random_state=0)
    P_ref = pca.fit_transform(Xref_s)
    P_tr = pca.transform(Xtr_s)

    Ph = P_ref[:Xh_g.shape[0]]
    Pt = P_ref[Xh_g.shape[0]:]
    Ptr = P_tr.reshape(R, K, 2)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    ax.scatter(Ph[:, 0], Ph[:, 1], s=ref_s, alpha=ref_alpha, color="tab:orange",
               label="healthy reference (decoded)", rasterized=True)
    ax.scatter(Pt[:, 0], Pt[:, 1], s=ref_s, alpha=ref_alpha, color="tab:blue",
               label="tumor reference (decoded)", rasterized=True)

    for k in range(K):
        c = seed_colors[k % len(seed_colors)]
        traj = Ptr[:, k, :]
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=line_w, color=c, zorder=5)
        ax.scatter(traj[0, 0], traj[0, 1], s=start_s, color=c, edgecolor="black",
                   linewidth=0.5, zorder=7)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=end_s, marker="X", color=c,
                   edgecolor="black", linewidth=0.6, zorder=8)

    ax.set_title(
        f"Expression trajectories (PCA2) | top_var_genes={len(idx_g)} | zscore={zscore} | "
        f"sparsity_project={sparsity_project}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    # square axes if meaningful
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if np.isfinite([xmin, xmax, ymin, ymax]).all() and (xmax > xmin) and (ymax > ymin):
        rng2 = max(xmax - xmin, ymax - ymin)
        cx, cy = 0.5*(xmin + xmax), 0.5*(ymin + ymax)
        ax.set_xlim(cx - rng2/2, cx + rng2/2)
        ax.set_ylim(cy - rng2/2, cy + rng2/2)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def _add_arrow(ax, p0, p1, color):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=2.5,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=9,
    )


def pick_boundary_central_outlier_seeds(
    Z_pool: np.ndarray,          # (N, D) latents der Seed-Pool (z.B. nur healthy train seeds)
    Z_ref_h: np.ndarray,         # (Nh, D) healthy ref latents (eval oder train)
    Z_ref_t: np.ndarray,         # (Nt, D) tumor   ref latents (eval oder train)
    k: int = 50,
):
    """
    Returns 3 indices (local to Z_pool):
      - boundary: approx equidistant to healthy/tumor manifolds (kNN mean distances)
      - central: closest to healthy centroid (of refs)
      - outlier: farthest from healthy centroid
    """
    Z_pool = np.asarray(Z_pool, dtype=np.float32)
    Z_ref_h = np.asarray(Z_ref_h, dtype=np.float32)
    Z_ref_t = np.asarray(Z_ref_t, dtype=np.float32)

    k_h = min(k, max(1, Z_ref_h.shape[0]))
    k_t = min(k, max(1, Z_ref_t.shape[0]))

    nn_h = NearestNeighbors(n_neighbors=k_h, metric="euclidean")
    nn_t = NearestNeighbors(n_neighbors=k_t, metric="euclidean")
    nn_h.fit(Z_ref_h)
    nn_t.fit(Z_ref_t)

    d_h = nn_h.kneighbors(Z_pool, return_distance=True)[0].mean(axis=1)
    d_t = nn_t.kneighbors(Z_pool, return_distance=True)[0].mean(axis=1)

    # progress in [0,1], 0=healthy-like, 1=tumor-like
    prog = d_h / (d_h + d_t + 1e-9)
    boundary_idx = int(np.argmin(np.abs(prog - 0.5)))

    h_centroid = Z_ref_h.mean(axis=0, keepdims=True)
    dist_cent = np.linalg.norm(Z_pool - h_centroid, axis=1)
    central_idx = int(np.argmin(dist_cent))
    outlier_idx = int(np.argmax(dist_cent))

    # guarantee uniqueness
    chosen = [boundary_idx, central_idx, outlier_idx]
    uniq = []
    for x in chosen:
        if x not in uniq:
            uniq.append(x)
    # if duplicates happen, fill with far-apart picks as fallback
    if len(uniq) < 4:
        extra = pick_far_apart_indices(Z_pool, n_pick=4, seed=0).tolist()
        for x in extra:
            if x not in uniq:
                uniq.append(int(x))
            if len(uniq) == 4:
                break

    return np.array(uniq[:4], dtype=int)

def pick_far_apart_indices(Z: np.ndarray, n_pick=4, seed=0):
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    if n <= n_pick:
        return np.arange(n, dtype=int)

    picks = [int(rng.integers(0, n))]
    D = np.linalg.norm(Z - Z[picks[0]][None, :], axis=1)

    while len(picks) < n_pick:
        j = int(np.argmax(D))
        picks.append(j)
        D = np.minimum(D, np.linalg.norm(Z - Z[j][None, :], axis=1))
    return np.array(picks, dtype=int)


def save_json(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def to_numpy_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    if hasattr(X, "A"):
        return X.A
    return np.asarray(X)


def run_guided_baseline_for_direction(
    direction,
    seed_idx,
    target_id,
    out_root,
    *,
    args,
    device,
    model,
    diffusion,
    clf,
    # NEW: split-safe latents
    Z_seed_source_all,          # latents for the dataset you draw SEEDS from (typically TRAIN)
    Z_eval_ref_all=None,        # latents for the dataset you use as EVAL refs (typically TEST). If None -> use Z_seed_source_all
    # Seed latents in the SAME ORDER as seed_idx (optional, avoids re-index bugs)
    Z_seed_src=None,            # e.g. Z_seed_source_all[seed_idx]
    # indices for refs (must be relative to Z_eval_ref_all if provided, otherwise relative to Z_seed_source_all)
    h_tr=None,
    h_te=None,
    t_tr=None,
    t_te=None,
    vae=None,
    genes=None,
    real_h_eval=None,
    real_t_eval=None,
):
    """
    Guided baseline runner (single-pass classifier guidance), split-safe.

    - Uses Z_seed_source_all for editing seeds.
    - Uses Z_eval_ref_all (if provided) for evaluation refs (kNN + decoded eval refs).
    - If Z_eval_ref_all is None, falls back to Z_seed_source_all (single-file mode).
    """


    Z_seed_source_all = np.asarray(Z_seed_source_all, dtype=np.float32)
    if Z_eval_ref_all is None:
        Z_eval_ref_all = Z_seed_source_all
    else:
        Z_eval_ref_all = np.asarray(Z_eval_ref_all, dtype=np.float32)

    seed_idx = np.asarray(seed_idx, dtype=int)

    # If caller did not provide Z_seed_src, compute it robustly from seed source
    if Z_seed_src is None:
        Z_seed_src = Z_seed_source_all[seed_idx]
    else:
        Z_seed_src = np.asarray(Z_seed_src, dtype=np.float32)
        if Z_seed_src.shape[0] != seed_idx.shape[0]:
            raise ValueError(
                f"Z_seed_src has {Z_seed_src.shape[0]} rows but seed_idx has {seed_idx.shape[0]}."
            )

    # Build eval refs (prefer *_te if provided and non-trivial; otherwise fallback to *_tr)
    def _pick_ref(idx_te, idx_tr):
        if idx_te is not None and len(idx_te) > 10:
            return np.asarray(idx_te, dtype=int)
        if idx_tr is not None and len(idx_tr) > 10:
            return np.asarray(idx_tr, dtype=int)
        # last resort: allow empty, but downstream metrics will be nan
        return np.asarray(idx_te if idx_te is not None else idx_tr, dtype=int)

    h_ref_idx = _pick_ref(h_te, h_tr)
    t_ref_idx = _pick_ref(t_te, t_tr)

    # centroid is unused (centroid guidance removed)
    for t0 in args.guided_t:
        for s in args.guided_s:
            out_base = ensure_dir(os.path.join(out_root, f"guided_{direction}", f"t{t0}_s{s}"))

            edited_latents = []
            edited_seed_ids = []

            # ----- EDIT: use TRAIN (seed source) latents -----
            for start in range(0, len(seed_idx), args.batch):
                sl = slice(start, min(len(seed_idx), start + args.batch))
                batch_seed_ids = seed_idx[sl]  # indices into Z_seed_source_all

                z_orig = torch.from_numpy(Z_seed_source_all[batch_seed_ids]).float().to(device)

                if int(t0) <= 0:
                    x_final = z_orig
                else:
                    t = torch.full((z_orig.size(0),), int(t0), device=device, dtype=torch.long)
                    noise = torch.randn_like(z_orig)
                    xt = diffusion.q_sample(x_start=z_orig, t=t, noise=noise)

                    # classifier-guided reverse pass
                    x_final = guided_reverse_pass(
                        model, diffusion, clf,
                        xt, int(t0), target_id, float(s), device,
                        centroid=None, centroid_scale=0.0, debug=False
                    )

                edited_latents.append(x_final.detach().cpu().numpy().astype(np.float32))
                edited_seed_ids.append(batch_seed_ids.copy())

            Z_edit = np.vstack(edited_latents).astype(np.float32, copy=False)
            seed_ids = np.concatenate(edited_seed_ids).astype(int, copy=False)

            np.savez(os.path.join(out_base, "edited_latents.npz"), Z_edit=Z_edit, seed_ids=seed_ids)
            save_json(
                {"direction": direction, "t_param": int(t0), "guidance_s": float(s)},
                os.path.join(out_base, "config.json")
            )

            # ----- EVAL refs: use TEST (eval ref) latents -----
            Z_ref_h_lat_eval = Z_eval_ref_all[h_ref_idx] if h_ref_idx.size else np.zeros((0, Z_edit.shape[1]), dtype=np.float32)
            Z_ref_t_lat_eval = Z_eval_ref_all[t_ref_idx] if t_ref_idx.size else np.zeros((0, Z_edit.shape[1]), dtype=np.float32)

            knn_ref_mean, _ = real_reference_knn_target_fraction(
                Z_query=Z_edit,
                Z_ref_healthy=Z_ref_h_lat_eval,
                Z_ref_tumor=Z_ref_t_lat_eval,
                k=50,
            )
            desired_latent_knn = float(knn_ref_mean) if direction == "healthy2tumor" else float(1.0 - knn_ref_mean)

            # ----- Seed-specificity (paired one per seed) -----
            # Use the provided aligned seed latents, not re-indexing (avoids split bugs)
            Z_seed = Z_seed_src  # (n_seed, D)

            first_row_for_seed = {}
            for i, sid in enumerate(seed_ids):
                if sid not in first_row_for_seed:
                    first_row_for_seed[sid] = i

            # paired rows correspond to the order of seed_idx you edited
            paired_rows = np.array(
                [first_row_for_seed[sid] for sid in seed_idx if sid in first_row_for_seed],
                dtype=int
            )
            Z_edit_paired = Z_edit[paired_rows] if paired_rows.size else np.zeros((0, Z_edit.shape[1]), dtype=np.float32)

            seed_pres = seed_nn_preservation(Z_src=Z_seed, Z_edit=Z_edit_paired, k=30) if Z_edit_paired.shape[0] == Z_seed.shape[0] else {
                "seed_nn_recall": float("nan"),
                "seed_nn_jaccard": float("nan"),
            }
            seed_rank = seed_retrieval_rank(Z_src=Z_seed, Z_edit=Z_edit_paired) if Z_edit_paired.shape[0] == Z_seed.shape[0] else {
                "own_seed_rank_median": float("nan"),
                "own_seed_rank_mean": float("nan"),
                "own_seed_top1": float("nan"),
                "own_seed_top10": float("nan"),
                "own_seed_top50": float("nan"),
            }

            src_pw = pairwise_dist_summary(Z_seed, seed=args.seed)
            edt_pw = pairwise_dist_summary(Z_edit, seed=args.seed)
            collapse_ratio = (edt_pw["pairwise_dist_mean"] / (src_pw["pairwise_dist_mean"] + 1e-9))

            # ----- optional decoded subset eval -----
            n_decode = int(round(Z_edit.shape[0] * max(0.0, min(1.0, args.decode_frac))))
            decoded_available = n_decode > 0
            knn_desired_expr = np.nan
            sig_proj = np.nan

            if decoded_available and vae is not None and genes is not None:
                pick = np.random.default_rng(0).choice(Z_edit.shape[0], size=n_decode, replace=False)
                Z_sub = Z_edit[pick]

                X_dec = decode_latents_in_batches(vae, Z_sub, device, batch_size=args.batch)
                fake = ad.AnnData(X=X_dec, var=pd.DataFrame(index=genes))

                if getattr(args, "sparsity_project", False):
                    # use real_*_eval passed from main() (should be TEST if you’re split-safe)
                    real_tgt = real_t_eval if direction == "healthy2tumor" else real_h_eval
                    fake = project_sparsity_gene_wise(
                        fake, real_tgt,
                        min_detect_rate=args.sparsity_min_detect_rate,
                        max_detect_rate=args.sparsity_max_detect_rate
                    )

                X_fake = to_numpy_dense(fake.X).astype(np.float32)

                # decoded refs for expr-kNN (decode eval refs)
                X_ref_h_eval = decode_latents_in_batches(vae, Z_ref_h_lat_eval, device, batch_size=args.batch).astype(np.float32)
                X_ref_t_eval = decode_latents_in_batches(vae, Z_ref_t_lat_eval, device, batch_size=args.batch).astype(np.float32)

                knn_ref_mean_expr, _ = expr_pca_reference_knn_target_fraction(
                    X_query=X_fake,
                    X_ref_healthy=X_ref_h_eval,
                    X_ref_tumor=X_ref_t_eval,
                    k=50,
                    n_pcs=50
                )
                knn_desired_expr = np.nan
                if X_ref_h_eval.shape[0] >= 5 and X_ref_t_eval.shape[0] >= 5 and X_fake.shape[0] >= 1:
                    knn_ref_mean_expr, _ = expr_pca_reference_knn_target_fraction(
                        X_query=X_fake, X_ref_healthy=X_ref_h_eval, X_ref_tumor=X_ref_t_eval, k=50, n_pcs=50
                    )
                    knn_desired_expr = float(knn_ref_mean_expr) if direction=="healthy2tumor" else float(1.0-knn_ref_mean_expr)



                try:
                    fake.write_h5ad(os.path.join(out_base, "decoded_subset.h5ad"), compression="lzf")
                except Exception:
                    pass
            if not decoded_available:
                save_json(
                    {"decoded_available": False, "decode_frac": float(args.decode_frac), "n_decode": int(n_decode)},
                    os.path.join(out_base, "decoded_skipped.json")
                )

            # ----- run_eval -----
            run_eval = {
                "method": "guided_baseline",
                "mode": "single",
                "direction": direction,
                "t_param": int(t0),
                "guidance_s": float(s),
                "multi_pass_rounds": 0,
                "n_reverse_steps": int(t0),

                "decoded_available": bool(decoded_available),
                "n_decode": int(n_decode),

                "knn_desired_lat": float(desired_latent_knn),
                "knn_desired_expr": float(knn_desired_expr) if np.isfinite(knn_desired_expr) else float("nan"),

                "pairwise_dist_ratio_edit_over_seed": float(collapse_ratio),

            }
            run_eval.update(seed_pres)
            run_eval.update(seed_rank)

            save_json(run_eval, os.path.join(out_base, "run_eval.json"))

def de_logfc_concordance(
    de_real: pd.DataFrame,
    de_edit: pd.DataFrame,
    top_n: int = 200,
):
    """
    Compute Spearman correlation of logFC between
    real tumor-vs-healthy DE and edited-vs-healthy DE.

    Returns dict with rho, pval, n_genes.
    """

    # Ensure required columns
    for df, name in [(de_real, "de_real"), (de_edit, "de_edit")]:
        if "gene" not in df.columns or "logfc" not in df.columns:
            raise ValueError(f"{name} must contain columns ['gene', 'logfc']")

    # Take top-N by absolute logFC
    de_real_top = (
        de_real.assign(abs_logfc=lambda d: d["logfc"].abs())
               .sort_values("abs_logfc", ascending=False)
               .head(top_n)
               .loc[:, ["gene", "logfc"]]
    )

    de_edit_top = (
        de_edit.assign(abs_logfc=lambda d: d["logfc"].abs())
               .sort_values("abs_logfc", ascending=False)
               .head(top_n)
               .loc[:, ["gene", "logfc"]]
    )

    # Merge on genes
    merged = de_real_top.merge(
        de_edit_top,
        on="gene",
        suffixes=("_real", "_edit"),
    )

    if len(merged) < 10:
        return {
            "rho": np.nan,
            "pval": np.nan,
            "n_genes": int(len(merged)),
        }

    rho, pval = spearmanr(
        merged["logfc_real"],
        merged["logfc_edit"],
    )

    return {
        "rho": float(rho),
        "pval": float(pval),
        "n_genes": int(len(merged)),
    }


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_runtime_manifest(args, adata_obj, device):

    def _sha256_head(path, max_bytes=50_000_000):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                remaining = max_bytes
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    h.update(chunk)
                    remaining -= len(chunk)
            return h.hexdigest()
        except Exception:
            return None

    resolved_clf = args.clf_ckpt
    try:
        resolved_clf = resolve_classifier_ckpt(args.clf_ckpt)
    except Exception:
        pass

    return {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "cmd": " ".join([sys.executable] + sys.argv),
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": str(device),
        "scanpy": sc.__version__,
        "anndata": ad.__version__,
        "sklearn": sklearn.__version__,
        "deterministic_requested": True,
        "n_obs": int(adata_obj.n_obs),
        "n_vars": int(adata_obj.n_vars),
        "obs_columns": list(map(str, adata_obj.obs.columns)),
        "var_names_head200": list(map(str, adata_obj.var_names[:200])),
        "checkpoints": {
            "diff_ckpt": {"path": args.diff_ckpt, "sha256_head": _sha256_head(args.diff_ckpt)},
            "vae_ckpt": {"path": args.vae_ckpt, "sha256_head": _sha256_head(args.vae_ckpt)},
            "clf_ckpt_input": {"path": args.clf_ckpt},
            "clf_ckpt_resolved": {"path": resolved_clf, "sha256_head": _sha256_head(resolved_clf) if os.path.isfile(resolved_clf) else None},
        },
    }

def _load_decoded_subset(path_h5):
    fake = ad.read_h5ad(path_h5)
    X = fake.X.toarray() if hasattr(fake.X, "toarray") else np.asarray(fake.X)
    X = _sanitize_dense(X)
    fake.X = X
    return fake
# ============================================================
# ------------------------- NUM HELPERS ----------------------
# ============================================================

def _subsample_rows(X, max_n=20000, seed=0):
    X = np.asarray(X)
    if X.shape[0] <= max_n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx]

def _as_dense_float32(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    elif hasattr(X, "A"):
        X = X.A
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0] = 0.0
    return X

def _sanitize_dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    elif hasattr(X, "A"):
        X = X.A
    X = np.asarray(X, dtype=np.float32, order="C")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0.0] = 0.0
    return X

def bootstrap_ci(x, n_boot=500, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return {"mean": float(np.mean(x)) if x.size else np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": int(x.size)}
    rng = np.random.default_rng(seed)
    n = x.size
    boots = np.empty((n_boot,), dtype=np.float64)
    for b in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        boots[b] = np.mean(samp)
    boots.sort()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"mean": float(np.mean(x)), "ci_low": lo, "ci_high": hi, "n": int(x.size)}

def summarize_obs_columns(adata_obj, top_k=15):
    rep = {}
    for c in adata_obj.obs.columns:
        s = adata_obj.obs[c]
        info = {
            "dtype": str(s.dtype),
            "nunique": int(s.nunique(dropna=True)),
            "n_missing": int(s.isna().sum()),
        }
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s)):
            vc = s.astype(str).value_counts(dropna=True).head(top_k)
            info["top_values"] = {k: int(v) for k, v in vc.items()}
        rep[c] = info
    return rep


# ============================================================
# ------------------- GROUPWISE SPLIT (NO LEAK) --------------
# ============================================================

def groupwise_split_indices(adata_obj, mask_h, mask_t, split_col=None, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)

    idx_h_all = np.where(mask_h.values)[0]
    idx_t_all = np.where(mask_t.values)[0]

    if split_col is None or split_col not in adata_obj.obs.columns:
        def _split(idx):
            n = idx.size
            n_test = int(round(test_frac * n))
            perm = rng.permutation(idx)
            return perm[n_test:], perm[:n_test]
        h_tr, h_te = _split(idx_h_all)
        t_tr, t_te = _split(idx_t_all)
        return (h_tr, h_te, t_tr, t_te, {"mode": "random_cellwise", "test_frac": float(test_frac)})

    groups = adata_obj.obs[split_col].astype(str)
    h_groups = pd.unique(groups.iloc[idx_h_all])
    t_groups = pd.unique(groups.iloc[idx_t_all])
    all_groups = np.unique(np.concatenate([h_groups, t_groups]))
    n_test_g = max(1, int(round(test_frac * len(all_groups))))
    test_groups = set(rng.choice(all_groups, size=n_test_g, replace=False).tolist())

    h_te = idx_h_all[groups.iloc[idx_h_all].isin(test_groups).values]
    h_tr = idx_h_all[~groups.iloc[idx_h_all].isin(test_groups).values]
    t_te = idx_t_all[groups.iloc[idx_t_all].isin(test_groups).values]
    t_tr = idx_t_all[~groups.iloc[idx_t_all].isin(test_groups).values]

    info = {
        "mode": "groupwise",
        "split_col": str(split_col),
        "test_frac": float(test_frac),
        "n_total_groups": int(len(all_groups)),
        "n_test_groups": int(n_test_g),
        "test_groups_first20": list(sorted(test_groups))[:20],
        "n_h_train": int(len(h_tr)), "n_h_test": int(len(h_te)),
        "n_t_train": int(len(t_tr)), "n_t_test": int(len(t_te)),
    }
    return (h_tr, h_te, t_tr, t_te, info)


# ============================================================
# ----------------------- BASELINES --------------------------
# ============================================================

def baseline_random_tumor(X_ref_t_dec, n_total, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n_ref = X_ref_t_dec.shape[0]
    idx = rng.integers(0, n_ref, size=n_total)
    return X_ref_t_dec[idx].astype(np.float32), idx.astype(np.int32)

def baseline_nearest_tumor_1(X_seed_dec, X_ref_t_dec, n_pcs=50, rng_seed=0):
    X_seed_dec = _as_dense_float32(X_seed_dec)
    X_ref_t_dec = _as_dense_float32(X_ref_t_dec)

    X_all = np.vstack([X_ref_t_dec, X_seed_dec])
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_all)

    n_pcs_eff = int(min(n_pcs, Xs.shape[1], Xs.shape[0] - 1))
    pca = PCA(n_components=n_pcs_eff, random_state=0)
    Z = pca.fit_transform(Xs).astype(np.float32)

    Z_ref = Z[:X_ref_t_dec.shape[0]]
    Z_q = Z[X_ref_t_dec.shape[0]:]

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(Z_ref)
    inds = nn.kneighbors(Z_q, return_distance=False).reshape(-1)
    X_nn = X_ref_t_dec[inds].astype(np.float32)
    return X_nn, inds.astype(np.int32)

def _make_match_keys(adata_obj, idx_seed, idx_tumor, match_cols=None, n_bins=10):
    if match_cols is None:
        match_cols = []
    obs = adata_obj.obs

    edges = {}
    for col in match_cols:
        if col not in obs.columns:
            continue
        s = obs[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = np.asarray(s.iloc[idx_tumor], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size < 20:
                continue
            qs = np.linspace(0, 1, n_bins + 1)
            edges[col] = np.quantile(vals, qs)

    def _key_for_rows(idxs):
        out = []
        for i in idxs:
            parts = []
            for col in match_cols:
                if col not in obs.columns:
                    continue
                v = obs.iloc[i][col]
                if pd.isna(v):
                    parts.append((col, "NA"))
                    continue
                if col in edges:
                    e = edges[col]
                    b = int(np.digitize([float(v)], e[1:-1], right=True)[0])
                    parts.append((col, b))
                else:
                    parts.append((col, str(v)))
            out.append(tuple(parts))
        return out

    return _key_for_rows(idx_seed), _key_for_rows(idx_tumor)

def baseline_matched_tumor(
    adata_obj,
    seed_ids_sub,
    idx_tumor_pool,
    X_ref_t_dec,
    n_pick=3,
    match_cols=None,
    n_bins=10,
    rng_seed=0,
):
    rng = np.random.default_rng(rng_seed)
    idx_seed = np.asarray(seed_ids_sub, dtype=int)
    idx_tumor = np.asarray(idx_tumor_pool, dtype=int)

    seed_keys, tumor_keys = _make_match_keys(adata_obj, idx_seed, idx_tumor, match_cols=match_cols, n_bins=n_bins)

    key_to_tum = {}
    for j, k in enumerate(tumor_keys):
        key_to_tum.setdefault(k, []).append(j)

    X_out, pick_out, seed_rep = [], [], []
    for i, sid in enumerate(idx_seed):
        k = seed_keys[i]
        cand = key_to_tum.get(k, [])
        if len(cand) == 0:
            cand = np.arange(X_ref_t_dec.shape[0])
        chosen = rng.choice(cand, size=n_pick, replace=(n_pick > len(cand)))
        X_out.append(X_ref_t_dec[chosen])
        pick_out.append(chosen)
        seed_rep.append(np.full((n_pick,), sid, dtype=int))

    return (
        np.vstack(X_out).astype(np.float32),
        np.concatenate(pick_out).astype(np.int32),
        np.concatenate(seed_rep).astype(np.int32),
    )

def baseline_latent_gaussian_editor(
    X_seed_dec,
    mu_h,
    mu_t,
    X_ref_t_dec,
    alpha=1.0,
    cov_mode="diag",
    eps_scale=1.0,
    rng_seed=0,
):
    rng = np.random.default_rng(rng_seed)
    X_seed_dec = _as_dense_float32(X_seed_dec)
    X_ref_t_dec = _as_dense_float32(X_ref_t_dec)

    delta = (mu_t - mu_h).astype(np.float32)

    Xt = X_ref_t_dec
    Xt_center = Xt - Xt.mean(axis=0, keepdims=True)

    if cov_mode == "full":
        C = np.cov(Xt_center, rowvar=False).astype(np.float32)
        C += 1e-6 * np.eye(C.shape[0], dtype=np.float32)
        eps = rng.multivariate_normal(mean=np.zeros(C.shape[0]), cov=C, size=X_seed_dec.shape[0]).astype(np.float32)
    else:
        var = Xt_center.var(axis=0).astype(np.float32) + 1e-6
        eps = rng.normal(loc=0.0, scale=np.sqrt(var)[None, :], size=X_seed_dec.shape).astype(np.float32)

    X_out = X_seed_dec + alpha * delta[None, :] + eps_scale * eps
    X_out = np.clip(X_out, 0.0, None).astype(np.float32)
    return X_out

def baseline_linear_shift(X_seed_dec: np.ndarray, mu_h: np.ndarray, mu_t: np.ndarray, alpha: float):
    X_seed_dec = np.asarray(X_seed_dec, dtype=np.float32)
    delta = (mu_t - mu_h).astype(np.float32)
    return (X_seed_dec + alpha * delta[None, :]).astype(np.float32)

def fit_pca_knn_ref(X_ref: np.ndarray, n_pcs=50, k=50):
    X_ref = np.asarray(X_ref, dtype=np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_ref)
    n_pcs_eff = int(min(n_pcs, Xs.shape[1], Xs.shape[0]-1))
    pca = PCA(n_components=n_pcs_eff, random_state=0)
    Z = pca.fit_transform(Xs).astype(np.float32)

    nn = NearestNeighbors(n_neighbors=min(k, Z.shape[0]-1), metric="euclidean")
    nn.fit(Z)
    return scaler, pca, nn, Z


# ============================================================
# --------------- PROGRESS AXIS (DECODER SPACE) --------------
# ============================================================

def fit_progress_axis_decoder_space(Xh: np.ndarray, Xt: np.ndarray, n_pcs: int = 128):
    Xh = np.asarray(Xh, dtype=np.float32)
    Xt = np.asarray(Xt, dtype=np.float32)

    X = np.vstack([Xh, Xt])
    y = np.concatenate([
        np.zeros(Xh.shape[0], dtype=np.int32),
        np.ones(Xt.shape[0], dtype=np.int32)
    ])

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    n_pcs_eff = int(min(n_pcs, Xs.shape[1], Xs.shape[0] - 1))
    if n_pcs_eff < 2:
        raise RuntimeError(f"Not enough samples/features for PCA. n_pcs_eff={n_pcs_eff}")

    pca = PCA(n_components=n_pcs_eff, random_state=0)
    Z = pca.fit_transform(Xs).astype(np.float32)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=1)
    clf.fit(Z, y)

    w = clf.coef_.reshape(-1).astype(np.float32)
    b = float(clf.intercept_[0])

    raw = (Z @ w + b).astype(np.float32)
    raw_h = raw[:Xh.shape[0]]
    raw_t = raw[Xh.shape[0]:]

    if raw_t.mean() < raw_h.mean():
        w = -w
        b = -b
        raw_h = -raw_h
        raw_t = -raw_t

    return scaler, pca, w, b, raw_h.astype(np.float32), raw_t.astype(np.float32)

def progress_score_01_decoder(X: np.ndarray, scaler, pca, w, b, ref_raw_h, ref_raw_t):
    X = np.asarray(X, dtype=np.float32)
    Xs = scaler.transform(X)
    Z = pca.transform(Xs).astype(np.float32)
    raw = (Z @ w + b).astype(np.float32)

    lo = float(np.median(ref_raw_h))
    hi = float(np.median(ref_raw_t))
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    s01 = (raw - lo) / denom
    return raw, np.clip(s01, 0.0, 1.0).astype(np.float32)

def fit_tumor_logreg_axis(
    X_ref_healthy: np.ndarray,
    X_ref_tumor: np.ndarray,
    n_pcs: int = 50,
    random_state: int = 0,
):
    """
    Fits a linear healthy-vs-tumor classifier in PCA expression space.

    Returns scaler, pca, classifier.
    Tumor is class 1.
    """
    Xh = np.asarray(X_ref_healthy, dtype=np.float32)
    Xt = np.asarray(X_ref_tumor, dtype=np.float32)

    X = np.vstack([Xh, Xt]).astype(np.float32)
    y = np.concatenate([
        np.zeros(Xh.shape[0], dtype=np.int32),
        np.ones(Xt.shape[0], dtype=np.int32),
    ])

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    n_pcs_eff = int(min(n_pcs, Xs.shape[1], Xs.shape[0] - 1))
    if n_pcs_eff <= 1:
        raise ValueError("Not enough cells/features for tumor logreg axis.")

    pca = PCA(n_components=n_pcs_eff, random_state=random_state)
    Z = pca.fit_transform(Xs).astype(np.float32)

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(Z, y)

    return scaler, pca, clf


def tumor_logreg_score(
    X_query: np.ndarray,
    scaler,
    pca,
    clf,
):
    """
    Scores query cells with the fitted tumor-vs-healthy classifier.

    tumor_logit > 0 means classifier-side tumor.
    tumor_proba close to 1 means high tumor probability.
    """
    Xq = np.asarray(X_query, dtype=np.float32)
    Zq = pca.transform(scaler.transform(Xq)).astype(np.float32)

    tumor_logit = clf.decision_function(Zq).astype(np.float32)
    tumor_proba = clf.predict_proba(Zq)[:, 1].astype(np.float32)

    return tumor_logit, tumor_proba

def stage_bins(scores01: np.ndarray, edges=(0.0, .2, .4, .6, .8, 1.0)):
    s = np.asarray(scores01, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    return np.digitize(s, edges[1:-1], right=True).astype(np.int32)  # 0..4


# ============================================================
# ---------------- PARETO + SEED SPECIFICITY -----------------
# ============================================================

def pareto_front(conversion: np.ndarray, identity_dist: np.ndarray) -> np.ndarray:
    conversion = np.asarray(conversion, dtype=np.float32)
    identity_dist = np.asarray(identity_dist, dtype=np.float32)

    order = np.argsort(-conversion)
    best_id = np.inf
    on_front = np.zeros(conversion.shape[0], dtype=bool)
    for idx in order:
        if identity_dist[idx] < best_id:
            on_front[idx] = True
            best_id = identity_dist[idx]
    return on_front

def _safe_k_for_selfdrop(n: int, k: int) -> int:
    if n is None or n <= 2:
        return 0
    return int(max(0, min(k, n - 2)))

def _knn_graph_indices(X: np.ndarray, k: int, include_self: bool = False):
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        return np.zeros((n, 0), dtype=int)

    if include_self:
        k_eff = int(min(k, n - 1))
        if k_eff <= 0:
            return np.zeros((n, 0), dtype=int)
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(X)
        return nn.kneighbors(return_distance=False)

    k_eff = int(min(k, n - 1))
    if k_eff <= 0:
        return np.zeros((n, 0), dtype=int)

    nn = NearestNeighbors(n_neighbors=min(k_eff + 1, n), metric="euclidean")
    nn.fit(X)
    inds = nn.kneighbors(return_distance=False)

    out = np.zeros((n, k_eff), dtype=int)
    for i in range(n):
        row = inds[i]
        row = row[row != i]
        if row.size < k_eff:
            pad = np.pad(row, (0, k_eff - row.size), mode="edge")
            out[i] = pad[:k_eff]
        else:
            out[i] = row[:k_eff]
    return out

def seed_nn_preservation(Z_src: np.ndarray, Z_edit: np.ndarray, k: int = 30):
    assert Z_src.shape[0] == Z_edit.shape[0]
    n = Z_src.shape[0]
    k = _safe_k_for_selfdrop(n, k)
    if k <= 0:
        return dict(seed_nn_recall=np.nan, seed_nn_jaccard=np.nan)

    src_nn = _knn_graph_indices(Z_src, k=k, include_self=False)
    edt_nn = _knn_graph_indices(Z_edit, k=k, include_self=False)

    recalls, jaccs = [], []
    for i in range(n):
        a = set(src_nn[i].tolist())
        b = set(edt_nn[i].tolist())
        inter = len(a & b)
        recalls.append(inter / (len(a) + 1e-9))
        jaccs.append(inter / (len(a | b) + 1e-9))

    return dict(
        seed_nn_recall=float(np.mean(recalls)),
        seed_nn_jaccard=float(np.mean(jaccs)),
    )

def seed_retrieval_rank(Z_src: np.ndarray, Z_edit: np.ndarray):
    assert Z_src.shape[0] == Z_edit.shape[0]
    Z_src = np.asarray(Z_src, dtype=np.float32)
    Z_edit = np.asarray(Z_edit, dtype=np.float32)

    A2 = (Z_edit**2).sum(axis=1, keepdims=True)
    B2 = (Z_src**2).sum(axis=1, keepdims=True).T
    D2 = A2 + B2 - 2.0 * (Z_edit @ Z_src.T)
    D2 = np.maximum(D2, 0.0)

    ranks = []
    top1 = top10 = top50 = 0
    for i in range(D2.shape[0]):
        order = np.argsort(D2[i])
        r = int(np.where(order == i)[0][0]) + 1
        ranks.append(r)
        top1 += (r <= 1)
        top10 += (r <= 10)
        top50 += (r <= 50)

    ranks = np.array(ranks)
    return dict(
        own_seed_rank_median=float(np.median(ranks)),
        own_seed_rank_mean=float(np.mean(ranks)),
        own_seed_top1=float(top1 / len(ranks)),
        own_seed_top10=float(top10 / len(ranks)),
        own_seed_top50=float(top50 / len(ranks)),
    )

def pairwise_dist_summary(Z: np.ndarray, max_pairs: int = 200000, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=np.float32)
    n = Z.shape[0]
    if n < 2:
        return dict(pairwise_dist_mean=np.nan, pairwise_dist_median=np.nan)

    m = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    mask = (i != j)
    i, j = i[mask], j[mask]
    d = np.linalg.norm(Z[i] - Z[j], axis=1)

    return dict(
        pairwise_dist_mean=float(np.mean(d)),
        pairwise_dist_median=float(np.median(d)),
    )

def pick_chain_per_seed(seed_ids, conversion, identity_dist, n_chain=5):
    seed_ids = np.asarray(seed_ids)
    conversion = np.asarray(conversion, dtype=np.float32)
    identity_dist = np.asarray(identity_dist, dtype=np.float32)
    rows = np.arange(seed_ids.shape[0])

    out = {}
    for sid in np.unique(seed_ids):
        r = rows[seed_ids == sid]
        if r.size == 0:
            continue
        pf = pareto_front(conversion[r], identity_dist[r])
        r_pf = r[pf] if pf.any() else r
        rr = r_pf[np.argsort(conversion[r_pf])]
        qs = np.linspace(0, 1, n_chain)
        picks = []
        for q in qs:
            j = int(np.clip(round(q * (rr.size - 1)), 0, rr.size - 1))
            picks.append(rr[j])
        out[sid] = np.array(picks, dtype=int)
    return out


# ============================================================
# -------------------- VAE / DIFFUSION IO --------------------
# ============================================================

@torch.no_grad()
def encode_vae(vae, Xnp: np.ndarray, device):
    X = torch.from_numpy(Xnp).float().to(device)
    return vae(X, return_latent=True).cpu().numpy()

def robust_load_vae(vae, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if missing:
        print(f"[VAE] missing keys: {len(missing)}")
    if unexpected:
        print(f"[VAE] unexpected keys: {len(unexpected)}")

@torch.no_grad()
def decode_latents_in_batches(vae, Z: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float32)
    out = []
    for start in range(0, Z.shape[0], batch_size):
        sl = slice(start, min(Z.shape[0], start + batch_size))
        z_t = torch.from_numpy(Z[sl]).float().to(device)
        x = vae.decoder(z_t).detach().cpu().numpy()
        out.append(x.astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

def build_vae_input_from_adata(adata_obj, genes_model):
    X_full = np.zeros((adata_obj.n_obs, len(genes_model)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(adata_obj.var_names.astype(str))}
    for j, g in enumerate(genes_model):
        i = gene_to_idx.get(g)
        if i is None:
            continue
        col = adata_obj.X[:, i]
        if hasattr(col, "toarray"):
            col = col.toarray().ravel()
        elif hasattr(col, "A1"):
            col = col.A1
        X_full[:, j] = col.astype(np.float32)
    return X_full


# ============================================================
# --------------------- SPARSITY PROJECTION ------------------
# ============================================================

def project_sparsity_gene_wise(adata_gen: ad.AnnData,
                              adata_target: ad.AnnData,
                              min_detect_rate: float = 0.0,
                              max_detect_rate: float = 1.0) -> ad.AnnData:
    G = adata_gen.copy()
    Xg = to_numpy_dense(G.X).astype(np.float32)
    Xt = to_numpy_dense(adata_target.X).astype(np.float32)

    n_gen_cells = Xg.shape[0]
    if n_gen_cells == 0:
        return G

    p = (Xt > 0).mean(axis=0)
    p = np.clip(p, min_detect_rate, max_detect_rate)

    for j in range(Xg.shape[1]):
        k = int(round(p[j] * n_gen_cells))
        if k <= 0:
            Xg[:, j] = 0.0
            continue
        if k >= n_gen_cells:
            continue
        col = Xg[:, j]
        thr = np.partition(col, n_gen_cells - k)[n_gen_cells - k]
        col[col < thr] = 0.0
        Xg[:, j] = col

    G.X = Xg
    return G


# ============================================================
# ------------------------ PLOTS (ROBUST) --------------------
# ============================================================

def plot_umap_latent(Z_h, Z_t, Z_edit, out_png: str):
    X = np.vstack([Z_h, Z_t, Z_edit])
    labels = (
        ["healthy_real"] * Z_h.shape[0]
        + ["tumor_real"] * Z_t.shape[0]
        + ["edited"] * Z_edit.shape[0]
    )

    A = ad.AnnData(X=X, obs=pd.DataFrame({"source": labels}))
    A.var_names = [f"dim_{i}" for i in range(X.shape[1])]

    n_pcs = int(min(50, X.shape[1], X.shape[0] - 1))
    if n_pcs < 2:
        return

    sc.tl.pca(A, n_comps=n_pcs)
    sc.pp.neighbors(A, n_pcs=n_pcs, n_neighbors=30)
    sc.tl.umap(A, min_dist=0.3)

    emb = A.obsm["X_umap"]
    src = A.obs["source"].astype(str).values

    style_map = {
        "healthy_real": {"color": "tab:orange", "marker": "o", "label": "healthy_real"},
        "tumor_real": {"color": "tab:blue", "marker": "s", "label": "tumor_real"},
        "edited": {"color": "tab:green", "marker": "^", "label": "edited"},
    }

    plt.figure(figsize=(7, 6))
    for key in ["healthy_real", "tumor_real", "edited"]:
        mask = src == key
        if not np.any(mask):
            continue
        st = style_map[key]
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=st["color"],
            marker=st["marker"],
            label=st["label"],
            s=18,
            alpha=0.35,
            edgecolors="none",
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def plot_umap_expr(real_h: ad.AnnData, real_t: ad.AnnData, fake: ad.AnnData, out_png: str):
    common = real_h.var_names.intersection(fake.var_names).intersection(real_t.var_names)
    real_h = real_h[:, common].copy()
    real_t = real_t[:, common].copy()
    fake = fake[:, common].copy()

    real_h.X = _sanitize_dense(real_h.X)
    real_t.X = _sanitize_dense(real_t.X)
    fake.X = _sanitize_dense(fake.X)

    real_h.obs["source"] = "healthy_real"
    real_t.obs["source"] = "tumor_real"
    fake.obs["source"] = "edited"

    A = ad.concat([real_h, real_t, fake], join="inner", merge="same")

    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)

    n_pcs = int(min(50, A.n_vars, A.n_obs - 1))
    if n_pcs < 2:
        return

    sc.tl.pca(A, n_comps=n_pcs)
    sc.pp.neighbors(A, n_pcs=n_pcs, n_neighbors=30)
    sc.tl.umap(A, min_dist=0.3)

    emb = A.obsm["X_umap"]
    src = A.obs["source"].astype(str).values

    style_map = {
        "healthy_real": {"color": "tab:orange", "marker": "o", "label": "healthy_real"},
        "tumor_real": {"color": "tab:blue", "marker": "s", "label": "tumor_real"},
        "edited": {"color": "tab:green", "marker": "^", "label": "edited"},
    }

    plt.figure(figsize=(7, 6))
    for key in ["healthy_real", "tumor_real", "edited"]:
        mask = src == key
        if not np.any(mask):
            continue
        st = style_map[key]
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=st["color"],
            marker=st["marker"],
            label=st["label"],
            s=18,
            alpha=0.35,
            edgecolors="none",
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def _fit_expr_umap_reference(real_h: ad.AnnData, real_t: ad.AnnData):
    """
    Fit one shared UMAP on real healthy + real tumor reference cells.
    Returns:
        A_ref : concatenated AnnData with X_umap
        common : common genes used
    """
    common = real_h.var_names.intersection(real_t.var_names)
    real_h = real_h[:, common].copy()
    real_t = real_t[:, common].copy()

    real_h.X = _sanitize_dense(real_h.X)
    real_t.X = _sanitize_dense(real_t.X)

    real_h.obs["source"] = "healthy_real"
    real_t.obs["source"] = "tumor_real"

    A_ref = ad.concat([real_h, real_t], join="inner", merge="same")

    sc.pp.normalize_total(A_ref, target_sum=1e4)
    sc.pp.log1p(A_ref)

    n_pcs = int(min(50, A_ref.n_vars, A_ref.n_obs - 1))
    if n_pcs < 2:
        raise ValueError("Not enough observations/variables to compute reference UMAP.")

    n_neighbors = max(2, min(30, A_ref.n_obs - 1))

    sc.tl.pca(A_ref, n_comps=n_pcs)
    sc.pp.neighbors(A_ref, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.tl.umap(A_ref, min_dist=0.3)

    return A_ref, common


def _project_points_to_ref_umap(ref_pca_coords: np.ndarray, ref_umap_coords: np.ndarray, query_pca_coords: np.ndarray, k: int = 15):
    """
    Simple kNN projection from PCA space into an existing UMAP space.
    query point gets weighted average of neighbors' UMAP coordinates.
    """
    k = int(max(1, min(k, ref_pca_coords.shape[0])))

    proj = np.zeros((query_pca_coords.shape[0], 2), dtype=float)
    for i in range(query_pca_coords.shape[0]):
        d = np.sum((ref_pca_coords - query_pca_coords[i]) ** 2, axis=1)
        nn = np.argpartition(d, k - 1)[:k]
        dd = d[nn]
        w = 1.0 / np.maximum(dd, 1e-8)
        w = w / w.sum()
        proj[i] = (ref_umap_coords[nn] * w[:, None]).sum(axis=0)
    return proj


def plot_umap_expr_seed_gallery(
    real_h_eval: ad.AnnData,
    real_t_eval: ad.AnnData,
    real_seed_adata: ad.AnnData,
    fake: ad.AnnData,
    seed_ids: np.ndarray,
    rep_ids: np.ndarray,
    out_png: str,
    n_show: int = 12,
    select_by: str = "best_progress",
    seed_scores: pd.DataFrame | None = None,
):
    """
    Multi-panel gallery:
      - fit one reference UMAP on real_h_eval + real_t_eval
      - project healthy seeds and generated samples into that same space
      - show selected seeds with all repetitions

    fake.obs must correspond row-wise to seed_ids / rep_ids order.
    """
    if fake.n_obs == 0:
        return

    # shared genes
    common = real_h_eval.var_names.intersection(real_t_eval.var_names).intersection(fake.var_names)
    real_h = real_h_eval[:, common].copy()
    real_t = real_t_eval[:, common].copy()
    fake = fake[:, common].copy()

    real_h.X = _sanitize_dense(real_h.X)
    real_t.X = _sanitize_dense(real_t.X)
    fake.X = _sanitize_dense(fake.X)

    # fit reference UMAP
    A_ref, common = _fit_expr_umap_reference(real_h, real_t)

    real_seed = real_seed_adata[:, common].copy()
    real_seed.X = _sanitize_dense(real_seed.X)

    A_q = ad.concat(
        [
            real_seed.copy(),
            fake.copy(),
        ],
        join="inner",
        merge="same",
    )

    sc.pp.normalize_total(A_q, target_sum=1e4)
    sc.pp.log1p(A_q)

    ref_loadings = A_ref.varm["PCs"]
    ref_center = A_ref.X.mean(axis=0)
    if hasattr(ref_center, "A1"):
        ref_center = ref_center.A1
    ref_center = np.asarray(ref_center).ravel()

    X_q = A_q.X
    if hasattr(X_q, "toarray"):
        X_q = X_q.toarray()
    X_q = np.asarray(X_q)

    X_ref_pca = A_ref.obsm["X_pca"]
    X_ref_umap = A_ref.obsm["X_umap"]

    X_q_centered = X_q - ref_center[None, :]
    X_q_pca = X_q_centered @ ref_loadings

    X_q_umap = _project_points_to_ref_umap(
        ref_pca_coords=X_ref_pca,
        ref_umap_coords=X_ref_umap,
        query_pca_coords=X_q_pca,
        k=15,
    )

    n_seed = real_seed.n_obs
    seed_umap = X_q_umap[:n_seed]
    fake_umap = X_q_umap[n_seed:]

    # choose which seeds to show
    unique_seed_ids = np.unique(seed_ids.astype(int))

    if seed_scores is not None and not seed_scores.empty and "cell_idx" in seed_scores.columns:
        S = seed_scores.copy()
        if select_by == "best_progress" and "progress01" in S.columns:
            S = S.sort_values("progress01", ascending=False)
        elif select_by == "median_progress" and "progress01_median" in S.columns:
            S = S.sort_values("progress01_median", ascending=False)
        elif select_by == "high_identity" and "id_dist_expr" in S.columns:
            S = S.sort_values("id_dist_expr", ascending=True)
        chosen = S["cell_idx"].drop_duplicates().astype(int).tolist()[:n_show]
    else:
        chosen = unique_seed_ids[:n_show].tolist()

    chosen = [c for c in chosen if c < seed_umap.shape[0]]
    if len(chosen) == 0:
        return

    # panel layout
    n_show_eff = len(chosen)
    ncols = 4
    nrows = int(np.ceil(n_show_eff / ncols))

    ref_src = A_ref.obs["source"].astype(str).values
    ref_emb = A_ref.obsm["X_umap"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    unique_seed_order = pd.Index(seed_ids.astype(int)).drop_duplicates().tolist()
    seed_pos = {int(gid): i for i, gid in enumerate(unique_seed_order)}
    for i, sid in enumerate(chosen):
        ax = axes[i]

        # background refs
        mh = ref_src == "healthy_real"
        mt = ref_src == "tumor_real"
        ax.scatter(
            ref_emb[mh, 0], ref_emb[mh, 1],
            c="tab:orange", s=8, alpha=0.10, marker="o", edgecolors="none"
        )
        ax.scatter(
            ref_emb[mt, 0], ref_emb[mt, 1],
            c="tab:blue", s=8, alpha=0.10, marker="s", edgecolors="none"
        )

        # seed
        if sid not in seed_pos:
            continue

        seed_local_idx = seed_pos[sid]
        sx, sy = seed_umap[seed_local_idx]

        ax.scatter(
            [sx], [sy],
            c="black", s=55, marker="o", edgecolors="white", linewidths=0.7, zorder=5
        )

        # this seed's reps
        mask = seed_ids.astype(int) == sid
        reps_xy = fake_umap[mask]
        reps_id = rep_ids[mask]

        if reps_xy.shape[0] > 0:
            # lines from seed to each sample
            for j in range(reps_xy.shape[0]):
                ax.plot(
                    [sx, reps_xy[j, 0]],
                    [sy, reps_xy[j, 1]],
                    color="gray",
                    alpha=0.25,
                    linewidth=0.8,
                    zorder=2,
                )

            ax.scatter(
                reps_xy[:, 0], reps_xy[:, 1],
                c="tab:green", s=28, alpha=0.75, marker="^",
                edgecolors="black", linewidths=0.2, zorder=4
            )

        title = f"seed {sid}  |  n={reps_xy.shape[0]}"
        if seed_scores is not None and not seed_scores.empty and "cell_idx" in seed_scores.columns:
            row = seed_scores[seed_scores["cell_idx"].astype(int) == sid]
            if len(row) > 0:
                row = row.iloc[0]
                extra = []
                if "progress01" in row.index:
                    extra.append(f"best={row['progress01']:.2f}")
                if "progress01_median" in row.index:
                    extra.append(f"med={row['progress01_median']:.2f}")
                if "id_dist_expr" in row.index:
                    extra.append(f"id={row['id_dist_expr']:.2f}")
                if len(extra) > 0:
                    title += "\n" + " | ".join(extra)

        ax.set_title(title, fontsize=9)

    # hide unused panels
    for j in range(n_show_eff, len(axes)):
        axes[j].axis("off")

    # legend in first panel only
    if len(chosen) > 0:
        axes[0].scatter([], [], c="tab:orange", s=20, alpha=0.6, marker="o", label="healthy ref")
        axes[0].scatter([], [], c="tab:blue", s=20, alpha=0.6, marker="s", label="tumor ref")
        axes[0].scatter([], [], c="black", s=30, marker="o", label="seed")
        axes[0].scatter([], [], c="tab:green", s=25, alpha=0.8, marker="^", label="generated")
        axes[0].legend(frameon=False, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def plot_umap_expr_all_samples(
    real_h_eval: ad.AnnData,
    real_t_eval: ad.AnnData,
    fake: ad.AnnData,
    seed_ids: np.ndarray,
    out_png: str,
):
    """
    Global overview plot:
      - shared reference UMAP on real healthy + real tumor
      - all generated samples projected into same space
    """
    if fake.n_obs == 0:
        return

    common = real_h_eval.var_names.intersection(real_t_eval.var_names).intersection(fake.var_names)
    real_h = real_h_eval[:, common].copy()
    real_t = real_t_eval[:, common].copy()
    fake = fake[:, common].copy()

    real_h.X = _sanitize_dense(real_h.X)
    real_t.X = _sanitize_dense(real_t.X)
    fake.X = _sanitize_dense(fake.X)

    A_ref, _ = _fit_expr_umap_reference(real_h, real_t)

    A_q = fake.copy()
    sc.pp.normalize_total(A_q, target_sum=1e4)
    sc.pp.log1p(A_q)

    ref_loadings = A_ref.varm["PCs"]
    ref_center = A_ref.X.mean(axis=0)
    if hasattr(ref_center, "A1"):
        ref_center = ref_center.A1
    ref_center = np.asarray(ref_center).ravel()

    X_q = A_q.X
    if hasattr(X_q, "toarray"):
        X_q = X_q.toarray()
    X_q = np.asarray(X_q)

    X_ref_pca = A_ref.obsm["X_pca"]
    X_ref_umap = A_ref.obsm["X_umap"]

    X_q_centered = X_q - ref_center[None, :]
    X_q_pca = X_q_centered @ ref_loadings
    X_q_umap = _project_points_to_ref_umap(
        ref_pca_coords=X_ref_pca,
        ref_umap_coords=X_ref_umap,
        query_pca_coords=X_q_pca,
        k=15,
    )

    ref_src = A_ref.obs["source"].astype(str).values
    ref_emb = A_ref.obsm["X_umap"]

    plt.figure(figsize=(8, 7))

    mh = ref_src == "healthy_real"
    mt = ref_src == "tumor_real"

    plt.scatter(
        ref_emb[mh, 0], ref_emb[mh, 1],
        c="tab:orange", s=8, alpha=0.08, marker="o",
        edgecolors="none", label="healthy ref"
    )
    plt.scatter(
        ref_emb[mt, 0], ref_emb[mt, 1],
        c="tab:blue", s=8, alpha=0.08, marker="s",
        edgecolors="none", label="tumor ref"
    )
    plt.scatter(
        X_q_umap[:, 0], X_q_umap[:, 1],
        c="tab:green", s=16, alpha=0.35, marker="^",
        edgecolors="none", label="generated"
    )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def plot_best_vs_identity(
    per_seed_summary: pd.DataFrame,
    out_png: str,
    x_col: str = "progress01",
    y_col: str = "id_dist_expr",
):
    """
    One point per seed:
      x = best tumor-like score
      y = identity loss of that best sample
    """
    if per_seed_summary is None or len(per_seed_summary) == 0:
        return
    if x_col not in per_seed_summary.columns or y_col not in per_seed_summary.columns:
        return

    D = per_seed_summary.copy()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        D[x_col].values,
        D[y_col].values,
        s=28,
        alpha=0.65,
        edgecolors="none",
    )

    # optional guide line for progress threshold
    plt.axvline(0.5, linestyle="--", linewidth=1, alpha=0.7)

    plt.xlabel(f"best per-seed {x_col}")
    plt.ylabel(f"identity loss ({y_col})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
# ============================================================
# ------------------------ EVAL HELPERS ----------------------
# ============================================================
from scipy.stats import pearsonr, spearmanr

def de_recovery_score(
    X_gen, X_healthy, X_tumor,
    gene_names=None,
    eps=1e-6,
):
    """
    Compare DE patterns:
    tumor vs healthy  vs  generated vs healthy

    Returns:
        dict with:
            - logfc_pearson
            - logfc_spearman
            - top_gene_overlap_50
            - top_gene_overlap_100
    """

    # mean expression
    mean_gen = X_gen.mean(axis=0)
    mean_h   = X_healthy.mean(axis=0)
    mean_t   = X_tumor.mean(axis=0)

    # log fold changes
    logfc_t = np.log1p(mean_t + eps) - np.log1p(mean_h + eps)
    logfc_g = np.log1p(mean_gen + eps) - np.log1p(mean_h + eps)

    # correlations
    pearson = pearsonr(logfc_t, logfc_g)[0]
    spearman = spearmanr(logfc_t, logfc_g)[0]

    # top genes
    def top_overlap(k):
        idx_t = np.argsort(-logfc_t)[:k]
        idx_g = np.argsort(-logfc_g)[:k]
        return len(set(idx_t) & set(idx_g)) / k

    return {
        "de_logfc_pearson": float(pearson),
        "de_logfc_spearman": float(spearman),
        "de_top50_overlap": float(top_overlap(50)),
        "de_top100_overlap": float(top_overlap(100)),
    }
def real_reference_knn_target_fraction(
    Z_query: np.ndarray,
    Z_ref_healthy: np.ndarray,
    Z_ref_tumor: np.ndarray,
    k: int = 50,
):
    Z_query = np.asarray(Z_query, dtype=np.float32)
    Zh = np.asarray(Z_ref_healthy, dtype=np.float32)
    Zt = np.asarray(Z_ref_tumor, dtype=np.float32)

    Z_ref = np.vstack([Zh, Zt])
    y_ref = np.concatenate([np.zeros(Zh.shape[0], dtype=np.int32),
                            np.ones(Zt.shape[0], dtype=np.int32)], axis=0)

    k = int(min(k, Z_ref.shape[0] - 1))
    if k <= 0 or Z_query.shape[0] == 0:
        return float("nan"), np.full((Z_query.shape[0],), np.nan, dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Z_ref)
    inds = nn.kneighbors(Z_query, return_distance=False)
    frac = y_ref[inds].mean(axis=1).astype(np.float32)
    return float(frac.mean()), frac

def expr_pca_reference_knn_target_fraction(
    X_query: np.ndarray,
    X_ref_healthy: np.ndarray,
    X_ref_tumor: np.ndarray,
    k: int = 50,
    n_pcs: int = 50,
):
    Xq = np.asarray(X_query, dtype=np.float32)
    Xh = np.asarray(X_ref_healthy, dtype=np.float32)
    Xt = np.asarray(X_ref_tumor, dtype=np.float32)

    if Xq.shape[0] == 0:
        return float("nan"), np.full((0,), np.nan, dtype=np.float32)

    Xref = np.vstack([Xh, Xt])
    yref = np.concatenate([np.zeros(Xh.shape[0], dtype=np.int32),
                           np.ones(Xt.shape[0], dtype=np.int32)])

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xref_s = scaler.fit_transform(Xref)
    Xq_s = scaler.transform(Xq)

    n_pcs_eff = int(min(n_pcs, Xref_s.shape[1], Xref_s.shape[0] - 1))
    if n_pcs_eff <= 1:
        return float("nan"), np.full((Xq.shape[0],), np.nan, dtype=np.float32)

    pca = PCA(n_components=n_pcs_eff, random_state=0)
    Zref = pca.fit_transform(Xref_s).astype(np.float32)
    Zq = pca.transform(Xq_s).astype(np.float32)

    k_eff = int(min(k, Zref.shape[0] - 1))
    if k_eff <= 0:
        return float("nan"), np.full((Xq.shape[0],), np.nan, dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Zref)
    inds = nn.kneighbors(Zq, return_distance=False)
    frac = yref[inds].mean(axis=1).astype(np.float32)
    return float(frac.mean()), frac

def latent_reference_distance_ratio_score(
    Z_query: np.ndarray,
    Z_ref_healthy: np.ndarray,
    Z_ref_tumor: np.ndarray,
    k: int = 50,
):
    """
    Continuous tumor-likeness score in latent space based on classwise kNN distances.

    For each query sample:
      d_h = mean distance to k nearest healthy refs
      d_t = mean distance to k nearest tumor refs

      score = d_h / (d_h + d_t)

    Interpretation:
      ~0   -> healthy-like
      ~1   -> tumor-like
    """
    Zq = np.asarray(Z_query, dtype=np.float32)
    Zh = np.asarray(Z_ref_healthy, dtype=np.float32)
    Zt = np.asarray(Z_ref_tumor, dtype=np.float32)

    if Zq.shape[0] == 0 or Zh.shape[0] < 2 or Zt.shape[0] < 2:
        return {
            "score_mean": float("nan"),
            "score_per_cell": np.full((Zq.shape[0],), np.nan, dtype=np.float32),
            "dist_h_mean": float("nan"),
            "dist_t_mean": float("nan"),
        }

    k_h = int(min(k, Zh.shape[0]))
    k_t = int(min(k, Zt.shape[0]))

    nn_h = NearestNeighbors(n_neighbors=k_h, metric="euclidean")
    nn_t = NearestNeighbors(n_neighbors=k_t, metric="euclidean")
    nn_h.fit(Zh)
    nn_t.fit(Zt)

    d_h = nn_h.kneighbors(Zq, return_distance=True)[0].mean(axis=1).astype(np.float32)
    d_t = nn_t.kneighbors(Zq, return_distance=True)[0].mean(axis=1).astype(np.float32)

    score = d_h / (d_h + d_t + 1e-9)
    return {
        "score_mean": float(np.mean(score)),
        "score_per_cell": score,
        "dist_h_mean": float(np.mean(d_h)),
        "dist_t_mean": float(np.mean(d_t)),
    }


def expr_pca_reference_distance_ratio_score(
    X_query: np.ndarray,
    X_ref_healthy: np.ndarray,
    X_ref_tumor: np.ndarray,
    k: int = 50,
    n_pcs: int = 50,
):
    """
    Continuous tumor-likeness score in expression space after PCA on refs.

    Fits scaler+PCA on ref healthy/tumor, then computes classwise kNN distances
    in PCA space and returns:
      score = d_h / (d_h + d_t)

    Interpretation:
      ~0   -> healthy-like
      ~1   -> tumor-like
    """
    Xq = np.asarray(X_query, dtype=np.float32)
    Xh = np.asarray(X_ref_healthy, dtype=np.float32)
    Xt = np.asarray(X_ref_tumor, dtype=np.float32)

    if Xq.shape[0] == 0 or Xh.shape[0] < 2 or Xt.shape[0] < 2:
        return {
            "score_mean": float("nan"),
            "score_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
            "dist_h_mean": float("nan"),
            "dist_t_mean": float("nan"),
            "dist_h_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
            "dist_t_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
        }

    Xref = np.vstack([Xh, Xt])
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xref_s = scaler.fit_transform(Xref)
    Xq_s = scaler.transform(Xq)

    n_pcs_eff = int(min(n_pcs, Xref_s.shape[1], Xref_s.shape[0] - 1))
    if n_pcs_eff <= 1:
        return {
            "score_mean": float("nan"),
            "score_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
            "dist_h_mean": float("nan"),
            "dist_t_mean": float("nan"),
            "dist_h_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
            "dist_t_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
        }

    pca = PCA(n_components=n_pcs_eff, random_state=0)
    Zref = pca.fit_transform(Xref_s).astype(np.float32)
    Zq = pca.transform(Xq_s).astype(np.float32)

    Zh = Zref[:Xh.shape[0]]
    Zt = Zref[Xh.shape[0]:]

    k_h = int(min(k, Zh.shape[0]))
    k_t = int(min(k, Zt.shape[0]))

    nn_h = NearestNeighbors(n_neighbors=k_h, metric="euclidean")
    nn_t = NearestNeighbors(n_neighbors=k_t, metric="euclidean")
    nn_h.fit(Zh)
    nn_t.fit(Zt)

    d_h = nn_h.kneighbors(Zq, return_distance=True)[0].mean(axis=1).astype(np.float32)
    d_t = nn_t.kneighbors(Zq, return_distance=True)[0].mean(axis=1).astype(np.float32)

    score = d_h / (d_h + d_t + 1e-9)
    return {
        "score_mean": float(np.mean(score)),
        "score_per_cell": score,
        "dist_h_mean": float(np.mean(d_h)),
        "dist_t_mean": float(np.mean(d_t)),
        "dist_h_per_cell": d_h.astype(np.float32),
        "dist_t_per_cell": d_t.astype(np.float32),
    }


def latent_axis_progress_score(
    Z_query: np.ndarray,
    Z_ref_healthy: np.ndarray,
    Z_ref_tumor: np.ndarray,
):
    """
    Projection score along healthy->tumor centroid axis in latent space.

    Returns normalized score:
      0 -> healthy centroid
      1 -> tumor centroid
    Can be <0 or >1 if outside the segment.
    """
    Zq = np.asarray(Z_query, dtype=np.float32)
    Zh = np.asarray(Z_ref_healthy, dtype=np.float32)
    Zt = np.asarray(Z_ref_tumor, dtype=np.float32)

    if Zq.shape[0] == 0 or Zh.shape[0] < 2 or Zt.shape[0] < 2:
        return {
            "score_mean": float("nan"),
            "score_per_cell": np.full((Zq.shape[0],), np.nan, dtype=np.float32),
        }

    c_h = Zh.mean(axis=0)
    c_t = Zt.mean(axis=0)
    v = c_t - c_h
    denom = float(np.dot(v, v)) + 1e-9

    s = ((Zq - c_h[None, :]) @ v) / denom
    s = s.astype(np.float32)

    return {
        "score_mean": float(np.mean(s)),
        "score_per_cell": s,
    }


def expr_axis_progress_score(
    X_query: np.ndarray,
    X_ref_healthy: np.ndarray,
    X_ref_tumor: np.ndarray,
    n_pcs: int = 50,
):
    """
    Projection score along healthy->tumor centroid axis in PCA expression space.

    0 -> healthy centroid
    1 -> tumor centroid
    """
    Xq = np.asarray(X_query, dtype=np.float32)
    Xh = np.asarray(X_ref_healthy, dtype=np.float32)
    Xt = np.asarray(X_ref_tumor, dtype=np.float32)

    if Xq.shape[0] == 0 or Xh.shape[0] < 2 or Xt.shape[0] < 2:
        return {
            "score_mean": float("nan"),
            "score_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
        }

    Xref = np.vstack([Xh, Xt])
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xref_s = scaler.fit_transform(Xref)
    Xq_s = scaler.transform(Xq)

    n_pcs_eff = int(min(n_pcs, Xref_s.shape[1], Xref_s.shape[0] - 1))
    if n_pcs_eff <= 1:
        return {
            "score_mean": float("nan"),
            "score_per_cell": np.full((Xq.shape[0],), np.nan, dtype=np.float32),
        }

    pca = PCA(n_components=n_pcs_eff, random_state=0)
    Zref = pca.fit_transform(Xref_s).astype(np.float32)
    Zq = pca.transform(Xq_s).astype(np.float32)

    Zh = Zref[:Xh.shape[0]]
    Zt = Zref[Xh.shape[0]:]

    c_h = Zh.mean(axis=0)
    c_t = Zt.mean(axis=0)
    v = c_t - c_h
    denom = float(np.dot(v, v)) + 1e-9

    s = ((Zq - c_h[None, :]) @ v) / denom
    s = s.astype(np.float32)

    return {
        "score_mean": float(np.mean(s)),
        "score_per_cell": s,
    }

def _pairwise_mean_dist(X: np.ndarray, max_pairs: int = 200000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        return float("nan")
    m = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    mask = (i != j)
    i, j = i[mask], j[mask]
    d = np.linalg.norm(X[i] - X[j], axis=1)
    return float(np.mean(d))


def intra_seed_diversity(X: np.ndarray, seed_ids: np.ndarray, max_pairs_per_seed: int = 20000, seed: int = 0):
    """
    Mean pairwise distance within each seed group (in the given space X).
    Returns mean_over_seeds and per-seed table.
    """
    X = np.asarray(X, dtype=np.float32)
    seed_ids = np.asarray(seed_ids).astype(int)
    uniq = np.unique(seed_ids)
    rows = []
    for sid in uniq:
        idx = np.where(seed_ids == sid)[0]
        if idx.size < 2:
            rows.append((sid, np.nan, int(idx.size)))
            continue
        d = _pairwise_mean_dist(X[idx], max_pairs=max_pairs_per_seed, seed=seed + sid)
        rows.append((sid, d, int(idx.size)))
    df = pd.DataFrame(rows, columns=["cell_idx", "intra_seed_pairwise_mean", "n_samples"])
    return float(df["intra_seed_pairwise_mean"].mean(skipna=True)), df


def seed_retrieval_rank_from_candidates(X_seed: np.ndarray, X_cand: np.ndarray, seed_ids_cand: np.ndarray):
    """
    For each candidate sample, rank its true seed among all seeds by Euclidean distance.
    Lower is better. Also compute top1/top10.
    """
    X_seed = np.asarray(X_seed, dtype=np.float32)
    X_cand = np.asarray(X_cand, dtype=np.float32)
    seed_ids_cand = np.asarray(seed_ids_cand).astype(int)

    # Distance candidate -> all seeds
    A2 = (X_cand**2).sum(axis=1, keepdims=True)
    B2 = (X_seed**2).sum(axis=1, keepdims=True).T
    D2 = A2 + B2 - 2.0 * (X_cand @ X_seed.T)
    D2 = np.maximum(D2, 0.0)

    # Map global cell indices -> row in X_seed (0..n_seed-1)
    # Here we assume X_seed corresponds to your selected seed_idx array order.
    # So we need seed_ids_cand in that same index space; we’ll enforce that later.
    true = seed_ids_cand

    ranks = []
    top1 = top10 = 0
    for i in range(D2.shape[0]):
        order = np.argsort(D2[i])
        # true[i] already should be 0..n_seed-1
        r = int(np.where(order == true[i])[0][0]) + 1
        ranks.append(r)
        top1 += (r <= 1)
        top10 += (r <= 10)

    ranks = np.asarray(ranks, dtype=np.int32)
    return {
        "cand_seed_rank_median": float(np.median(ranks)),
        "cand_seed_rank_mean": float(np.mean(ranks)),
        "cand_seed_top1": float(top1 / len(ranks)),
        "cand_seed_top10": float(top10 / len(ranks)),
    }


def evaluate_method_sym(
    *,
    method_name: str,
    direction: str,
    X_cand_dec: np.ndarray,          # decoder-space candidates (n x G)
    Z_cand_lat: np.ndarray,          # latent candidates (n x D)
    seed_ids_global: np.ndarray,     # global indices of original seed cells (length n) OR mapped indices
    seed_ids_mapped: np.ndarray,     # mapped 0..n_seed-1 per candidate (length n)
    X_seed_dec: np.ndarray,          # decoded seeds aligned to candidates (n x G) if per-candidate, OR (n_seed x G) if per-seed
    X_seed_dec_unique: np.ndarray,   # (n_seed x G) decoded seed reps in fixed order
    Z_seed_lat_unique: np.ndarray,   # (n_seed x D) seed latents fixed order
    # reference sets (EVAL split preferred)
    Z_ref_h_lat_eval: np.ndarray,
    Z_ref_t_lat_eval: np.ndarray,
    X_ref_h_eval: np.ndarray,
    X_ref_t_eval: np.ndarray,
    # optional “signature projection” inputs
    real_h_eval: ad.AnnData = None,
    real_t_eval: ad.AnnData = None,
    fake_eval: ad.AnnData = None,
    src_label_qc: str = "healthy_real",
    tgt_label_qc: str = "tumor_real",
    # centroids
    Z_src_seed: np.ndarray = None,
    Z_tgt_seed: np.ndarray = None,
):
    """
    Returns a dict of symmetric metrics usable for BOTH zigzag and baselines.
    No classifier probabilities, no progress01.
    """

    # 1) Latent kNN tumor-fraction (ref-only)
    knn_ref_mean_lat, _ = real_reference_knn_target_fraction(
        Z_query=Z_cand_lat,
        Z_ref_healthy=Z_ref_h_lat_eval,
        Z_ref_tumor=Z_ref_t_lat_eval,
        k=50,
    )
    knn_desired_lat = float(knn_ref_mean_lat) if direction == "healthy2tumor" else float(1.0 - knn_ref_mean_lat)

    # 2) Expr PCA+kNN tumor-fraction (ref-only)
    knn_ref_mean_expr, _ = expr_pca_reference_knn_target_fraction(
        X_query=X_cand_dec,
        X_ref_healthy=X_ref_h_eval,
        X_ref_tumor=X_ref_t_eval,
        k=50,
        n_pcs=50,
    )
    knn_desired_expr = float(knn_ref_mean_expr) if direction == "healthy2tumor" else float(1.0 - knn_ref_mean_expr)
    de_scores = de_recovery_score(
        X_gen=X_cand_dec,
        X_healthy=X_ref_h_eval,
        X_tumor=X_ref_t_eval,
    )
    # 2b) Continuous tumor-likeness via classwise distance ratios
    lat_dist_ratio = latent_reference_distance_ratio_score(
        Z_query=Z_cand_lat,
        Z_ref_healthy=Z_ref_h_lat_eval,
        Z_ref_tumor=Z_ref_t_lat_eval,
        k=50,
    )
    expr_dist_ratio = expr_pca_reference_distance_ratio_score(
        X_query=X_cand_dec,
        X_ref_healthy=X_ref_h_eval,
        X_ref_tumor=X_ref_t_eval,
        k=50,
        n_pcs=50,
    )

    # 2c) Progress along healthy->tumor axis
    lat_axis = latent_axis_progress_score(
        Z_query=Z_cand_lat,
        Z_ref_healthy=Z_ref_h_lat_eval,
        Z_ref_tumor=Z_ref_t_lat_eval,
    )
    expr_axis = expr_axis_progress_score(
        X_query=X_cand_dec,
        X_ref_healthy=X_ref_h_eval,
        X_ref_tumor=X_ref_t_eval,
        n_pcs=50,
    )
    # 3) Identity distance to its seed (expr space)
    # X_seed_dec is per-candidate aligned (same shape as X_cand_dec)
    id_dist_expr = np.linalg.norm(X_cand_dec - X_seed_dec, axis=1).astype(np.float32)
    id_dist_expr_mean = float(np.mean(id_dist_expr))

    # 4) Diversity within each seed (expr space)
    intra_seed_mean, intra_seed_df = intra_seed_diversity(X_cand_dec, seed_ids_global, seed=0)

    # 5) Seed retrieval from candidates (expr space) — ranks against unique seed set
    seed_retr = seed_retrieval_rank_from_candidates(
        X_seed=X_seed_dec_unique,
        X_cand=X_cand_dec,
        seed_ids_cand=seed_ids_mapped,
    )

    # 6) Collapse/expansion (latent or expr) relative to seeds (use expr here)
    src_pw = _pairwise_mean_dist(X_seed_dec_unique, seed=0)
    edt_pw = _pairwise_mean_dist(X_cand_dec, seed=0)
    collapse_ratio = float(edt_pw / (src_pw + 1e-9)) if np.isfinite(src_pw) and np.isfinite(edt_pw) else float("nan")



    # 8) Latent centroid score (optional)
    latent_centroid = float("nan")
    if (Z_src_seed is not None) and (Z_tgt_seed is not None):
        latent_centroid = float(_latent_centroid_score(Z_src_seed, Z_tgt_seed, Z_cand_lat))



    out = {
        "method": method_name,
        "direction": direction,

        # existing
        "knn_desired_lat": float(knn_desired_lat),
        "knn_desired_expr": float(knn_desired_expr),

        # NEW: continuous proximity metrics
        "dist_ratio_lat": float(lat_dist_ratio["score_mean"]),
        "dist_ratio_expr": float(expr_dist_ratio["score_mean"]),
        "dist_to_healthy_lat": float(lat_dist_ratio["dist_h_mean"]),
        "dist_to_tumor_lat": float(lat_dist_ratio["dist_t_mean"]),
        "dist_to_healthy_expr": float(expr_dist_ratio["dist_h_mean"]),
        "dist_to_tumor_expr": float(expr_dist_ratio["dist_t_mean"]),

        # NEW: axis projection metrics
        "axis_progress_lat": float(lat_axis["score_mean"]),
        "axis_progress_expr": float(expr_axis["score_mean"]),

        # existing identity/diversity
        "id_dist_expr_mean": float(id_dist_expr_mean),
        "intra_seed_div_mean": float(intra_seed_mean),
        "pairwise_dist_ratio_edit_over_seed": float(collapse_ratio),
        "latent_centroid_score": float(latent_centroid) if np.isfinite(latent_centroid) else float("nan"),
        "de_logfc_pearson": de_scores["de_logfc_pearson"],
        "de_logfc_spearman": de_scores["de_logfc_spearman"],
        "de_top50_overlap": de_scores["de_top50_overlap"],
        "de_top100_overlap": de_scores["de_top100_overlap"],
    }
    out.update(seed_retr)

    return out, intra_seed_df, id_dist_expr




def _latent_centroid_score(Z_src, Z_tgt, Z_edit):
    Z_src = np.asarray(Z_src, dtype=np.float32)
    Z_tgt = np.asarray(Z_tgt, dtype=np.float32)
    Z_edit = np.asarray(Z_edit, dtype=np.float32)

    c_src = Z_src.mean(axis=0, keepdims=True)
    c_tgt = Z_tgt.mean(axis=0, keepdims=True)
    d_src = np.linalg.norm(Z_edit - c_src, axis=1)
    d_tgt = np.linalg.norm(Z_edit - c_tgt, axis=1)
    return float(np.mean(d_src / (d_src + d_tgt + 1e-9)))


# ============================================================
# ---------------------- BIO CONCAT (QC) ---------------------
# ============================================================

def prep_bio_umap_space(A: ad.AnnData, n_pcs=50, n_neighbors=30):
    B = A.copy()
    sc.pp.scale(B, max_value=10)
    sc.tl.pca(B, n_comps=n_pcs)
    sc.pp.neighbors(B, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.tl.umap(B, min_dist=0.3)
    return B


def _prep_concat_for_bio(real_h, real_t, fake):
    common = real_h.var_names.intersection(fake.var_names)
    real_h = real_h[:, common].copy()
    real_t = real_t[:, common].copy()
    fake   = fake[:, common].copy()

    real_h.X = _sanitize_dense(real_h.X)
    real_t.X = _sanitize_dense(real_t.X)
    fake.X   = _sanitize_dense(fake.X)

    real_h.obs["source"] = "healthy_real"
    real_t.obs["source"] = "tumor_real"
    fake.obs["source"]   = "edited"

    A = ad.concat([real_h, real_t, fake], join="inner", merge="same")
    return A


def _prep_concat_for_de(real_src, real_tgt, fake, do_log_norm=False):
    """
    DE-ready: NO sc.pp.scale() here.
    """
    common = real_src.var_names.intersection(fake.var_names)
    real_src = real_src[:, common].copy()
    real_tgt = real_tgt[:, common].copy()
    fake = fake[:, common].copy()

    real_src.X = _sanitize_dense(real_src.X)
    real_tgt.X = _sanitize_dense(real_tgt.X)
    fake.X = _sanitize_dense(fake.X)

    real_src.obs["grp"] = "source"
    real_tgt.obs["grp"] = "target"
    fake.obs["grp"] = "edited"

    A = ad.concat([real_src, real_tgt, fake], join="inner", merge="same")

    if do_log_norm:
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)

    return A

def _rank_genes(A, groups, reference, top_n=50, outfile=None):
    sc.tl.rank_genes_groups(
        A,
        groupby="grp",
        groups=[groups],
        reference=reference,
        method="wilcoxon",
        use_raw=False,
        pts=True,
        tie_correct=True,
    )
    r = A.uns["rank_genes_groups"]
    names = np.array(r["names"][groups]).astype(str)
    logfc = np.array(r["logfoldchanges"][groups]).astype(float)
    pval_adj = np.array(r["pvals_adj"][groups]).astype(float)
    scores = np.array(r["scores"][groups]).astype(float)

    if "pts" in r:
        pct_in_group = np.array(r["pts"][groups]).astype(float)
    else:
        pct_in_group = np.full_like(scores, np.nan)

    if "pts_rest" in r:
        pct_in_ref = np.array(r["pts_rest"][groups]).astype(float)
    else:
        pct_in_ref = np.full_like(scores, np.nan)

    df = pd.DataFrame({
        "gene": names,
        "logfc": logfc,
        "pval_adj": pval_adj,
        "score": scores,
        "pct_in_group": pct_in_group,
        "pct_in_ref": pct_in_ref,
    }).iloc[:top_n].reset_index(drop=True)

    if outfile:
        df.to_csv(outfile, index=False)
    return df

def _geneset_score(A, genes, outcol):
    genes = [g for g in genes if g in A.var_names]
    if not genes:
        A.obs[outcol] = np.nan
        return
    X = A[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    A.obs[outcol] = X.mean(axis=1)


# ============================================================
# --------------- CLASSIFIER CHECKPOINT RESOLVE --------------
# ============================================================

def resolve_classifier_ckpt(path_or_dir: str) -> str:
    def _is_model_file(p):
        bn = os.path.basename(p).lower()
        return bn.startswith("model") and (bn.endswith(".pt") or bn.endswith(".pth") or bn.endswith(".ckpt"))

    if os.path.isdir(path_or_dir):
        cands = []
        for pat in ("model*.pt", "model*.pth", "model*.ckpt"):
            cands += glob.glob(os.path.join(path_or_dir, pat))
        if not cands:
            raise FileNotFoundError(f"No model checkpoints (model*.pt/pth/ckpt) found in dir: {path_or_dir}")
        cands.sort(key=lambda p: os.path.getmtime(p))
        ckpt = cands[-1]
        print(f"[clf] using checkpoint: {ckpt}")
        return ckpt

    if not os.path.exists(path_or_dir):
        d = os.path.dirname(path_or_dir)
        if os.path.isdir(d):
            return resolve_classifier_ckpt(d)
        raise FileNotFoundError(f"Classifier checkpoint not found: {path_or_dir}")

    bn = os.path.basename(path_or_dir).lower()
    if bn.startswith("opt"):
        mapped = os.path.join(os.path.dirname(path_or_dir), "model" + bn[3:])
        if os.path.exists(mapped):
            print(f"[clf] mapped optimizer to model: {mapped}")
            return mapped
        return resolve_classifier_ckpt(os.path.dirname(path_or_dir))

    if _is_model_file(path_or_dir):
        print(f"[clf] using checkpoint: {path_or_dir}")
        return path_or_dir

    return resolve_classifier_ckpt(os.path.dirname(path_or_dir))


# ============================================================
# ------------- GUIDED REVERSE STEP (classifier guidance) ----
# ============================================================

def make_cond_fn(clf, target_id: int, guidance_scale: float):
    def cond_fn(x, t, y=None, **kwargs):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = clf(x_in, t)
            score = logits[:, target_id].mean()
            grad = torch.autograd.grad(score, x_in)[0]
        return guidance_scale * grad
    return cond_fn

def guided_reverse_pass(
    model,
    diffusion,
    clf,
    x_start,
    t_start,
    target_id,
    s,
    device,
    centroid=None,
    centroid_scale: float = 0.0,
    step_scale: float = 1.0,
    debug: bool = False,
):
    """
    Reverse pass with optional per-step damping.

    step_scale=1.0:
        old behavior (full reverse updates)

    step_scale<1.0:
        each reverse step is only partially applied:
            x <- x + step_scale * (x_prop - x)
    """
    cond_fn = None if (s is None or float(s) == 0.0) else make_cond_fn(
        clf, target_id=target_id, guidance_scale=s
    )
    x = x_start
    step_scale = float(step_scale)

    for ti in range(t_start - 1, -1, -1):
        tt = torch.full((x.size(0),), ti, device=device, dtype=torch.long)

        if debug and ti in {t_start - 1, t_start // 2, 0}:
            with torch.enable_grad():
                x_tmp = x.detach().requires_grad_(True)
                logits = clf(x_tmp, tt)
                score = logits[:, target_id].mean()
                g = torch.autograd.grad(score, x_tmp, retain_graph=False, create_graph=False)[0]
            print(
                f"[dbg] t={ti:3d} logit={score.item(): .4f} "
                f"grad_norm={g.norm(dim=1).mean().item(): .6f}"
            )

        out = diffusion.p_sample(
            model=model,
            x=x,
            t=tt,
            cond_fn=cond_fn,
            model_kwargs={},
        )

        x_prop = out["sample"]

        # Apply only a partial reverse update if requested
        if step_scale < 1.0:
            x = x + step_scale * (x_prop - x)
        else:
            x = x_prop

        # Keep centroid guidance on the same effective scale
        if centroid is not None and centroid_scale > 0.0:
            x = x + (step_scale * centroid_scale) * (centroid - x)

    return x

# ============================================================
# ----------------------------- MAIN -------------------------
# ============================================================
