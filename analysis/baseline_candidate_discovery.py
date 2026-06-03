#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_candidate_discovery.py

FAIR BASELINE POOL GENERATOR for ZigZag / CancerZigZag.

This script creates seed-initialized baseline candidate pools only. It does not
train a new classifier and does not introduce a new evaluation metric. The point
is to evaluate all candidate pools later with the exact same ZigZag evaluation
pipeline: same tumor_proba classifier, same selection rule, same residual seed
anchoring, same DE/pathway analysis.

Compatible with older Slurm files: it accepts and ignores older arguments such
as --progress_threshold, --matched_progress_tolerance, --cloud_pairs, etc.

Outputs in RUN_DIR/baseline_pools/:
  baseline_pools_latent.npz
  baseline_pools_meta.csv
  baseline_pools_manifest.json
  baseline_pools_README.txt
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import MiniBatchKMeans


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
        from zigzag.common import load_latents_from_h5ad
        return {"load_latents_from_h5ad": load_latents_from_h5ad}
    except Exception as e:
        print("[debug] import search paths:")
        for c in candidates:
            print(" ", c, "contains zigzag:", os.path.isdir(os.path.join(c, "zigzag")))
        raise ImportError(
            "Could not import zigzag.common.load_latents_from_h5ad. "
            "Set --repo_dir to the folder containing zigzag/. "
            f"Original error: {e}"
        )


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


def load_trajectory_pool(run_dir):
    path = os.path.join(run_dir, "trajectories_latent_all_reps.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing trajectory pool: {path}\n"
            "Make sure the run was started with --save_all_rep_trajectories."
        )
    z = np.load(path)
    for key in ["Z_traj", "cell_idx", "rep_id"]:
        if key not in z:
            raise KeyError(f"Missing key {key} in {path}. Available keys: {list(z.keys())}")
    Z_traj = z["Z_traj"].astype(np.float32)
    cell_idx = z["cell_idx"].astype(int)
    rep_id = z["rep_id"].astype(int)
    if Z_traj.ndim != 3:
        raise ValueError(f"Expected Z_traj shape (n, states, dim), got {Z_traj.shape}")
    return Z_traj, cell_idx, rep_id


def make_seed_order(cell_idx):
    unique = np.array(list(dict.fromkeys(cell_idx.tolist())), dtype=int)
    m = {int(c): i for i, c in enumerate(unique)}
    order = np.array([m[int(c)] for c in cell_idx], dtype=int)
    return unique, order


def representative_seeds(Z_seed_pool, cell_idx_pool, unique_seeds):
    reps = []
    for c in unique_seeds:
        pos = np.where(cell_idx_pool == c)[0]
        if len(pos) == 0:
            raise RuntimeError(f"Seed {c} missing from pool")
        reps.append(Z_seed_pool[pos[0]])
    return np.vstack(reps).astype(np.float32)


def load_selected_positions(run_dir, selection_rule, pool_cell_idx, pool_rep_id):
    """Only used to estimate Gaussian sigma from selected ZigZag displacement."""
    path = os.path.join(run_dir, "selected_candidates_eval.csv")
    if not os.path.exists(path):
        print(f"[warning] selected_candidates_eval.csv missing: {path}")
        return None
    try:
        sel = pd.read_csv(path)
        if "selection_rule" in sel.columns:
            sel = sel[sel["selection_rule"].astype(str) == str(selection_rule)].copy()
        if sel.empty or "cell_idx" not in sel.columns or "rep_id" not in sel.columns:
            print("[warning] selected_candidates_eval.csv empty/incompatible for sigma estimate")
            return None
        sel["cell_idx"] = sel["cell_idx"].astype(int)
        sel["rep_id"] = sel["rep_id"].astype(int)
        sel["_key"] = sel["cell_idx"].astype(str) + "__" + sel["rep_id"].astype(str)
        keys = np.array([f"{c}__{r}" for c, r in zip(pool_cell_idx, pool_rep_id)], dtype=object)
        key_to_pos = {str(k): i for i, k in enumerate(keys)}
        pos = [key_to_pos[k] for k in sel["_key"].astype(str) if k in key_to_pos]
        if len(pos) == 0:
            print("[warning] selected rows could not be mapped to trajectory pool")
            return None
        return np.asarray(pos, dtype=int)
    except Exception as e:
        print(f"[warning] failed to load selected positions: {e}")
        return None


def add_pool(pool_list, Z_seed, Z_final, method, candidate_type, seed_order, cell_idx, rep_id, source, extra=None):
    n = int(Z_final.shape[0])
    if Z_seed.shape[0] != n:
        raise ValueError("Z_seed and Z_final have different numbers of rows")
    meta = pd.DataFrame({
        "method": [method] * n,
        "candidate_type": [candidate_type] * n,
        "seed_order": np.asarray(seed_order, dtype=int),
        "cell_idx": np.asarray(cell_idx, dtype=int),
        "rep_id": np.asarray(rep_id, dtype=int),
        "source": [source] * n,
    })
    if extra:
        for k, v in extra.items():
            meta[k] = v if not np.isscalar(v) else [v] * n
    pool_list.append({"Z_seed": Z_seed.astype(np.float32), "Z_final": Z_final.astype(np.float32), "meta": meta})


def build_gaussian_pool(args, rng, seed_Z, unique_seeds, sigma):
    n_seed = seed_Z.shape[0]
    Z_seed = np.repeat(seed_Z, args.budget, axis=0)
    Z_final = Z_seed + rng.normal(0.0, sigma, size=Z_seed.shape).astype(np.float32)
    seed_order = np.repeat(np.arange(n_seed), args.budget)
    cell_idx = np.repeat(unique_seeds.astype(int), args.budget)
    rep_id = np.tile(np.arange(args.budget), n_seed)
    return Z_seed, Z_final, seed_order, cell_idx, rep_id


def build_centroid_interpolation(seed_Z, unique_seeds, centroid, alphas):
    alphas = np.asarray(alphas, dtype=np.float32)
    Zs_all, Zf_all, seed_orders, cell_ids, rep_ids, alpha_vals = [], [], [], [], [], []
    for i, c in enumerate(unique_seeds):
        Z0 = seed_Z[i:i+1]
        Zf = Z0 + alphas[:, None] * (centroid[None, :] - Z0)
        Zs_all.append(np.repeat(Z0, len(alphas), axis=0))
        Zf_all.append(Zf)
        seed_orders.extend([i] * len(alphas))
        cell_ids.extend([int(c)] * len(alphas))
        rep_ids.extend(list(range(len(alphas))))
        alpha_vals.extend(alphas.tolist())
    return (np.vstack(Zs_all).astype(np.float32), np.vstack(Zf_all).astype(np.float32),
            np.asarray(seed_orders, dtype=int), np.asarray(cell_ids, dtype=int),
            np.asarray(rep_ids, dtype=int), np.asarray(alpha_vals, dtype=float))


def build_cluster_centroid_interpolation(seed_Z, unique_seeds, centers, alphas):
    alphas = np.asarray(alphas, dtype=np.float32)
    Zs_all, Zf_all, seed_orders, cell_ids, rep_ids, alpha_vals, cluster_ids = [], [], [], [], [], [], []
    for i, c in enumerate(unique_seeds):
        Z0 = seed_Z[i:i+1]
        rep_counter = 0
        for k, center in enumerate(centers):
            Zf = Z0 + alphas[:, None] * (center[None, :] - Z0)
            Zs_all.append(np.repeat(Z0, len(alphas), axis=0))
            Zf_all.append(Zf)
            seed_orders.extend([i] * len(alphas))
            cell_ids.extend([int(c)] * len(alphas))
            rep_ids.extend(list(range(rep_counter, rep_counter + len(alphas))))
            alpha_vals.extend(alphas.tolist())
            cluster_ids.extend([k] * len(alphas))
            rep_counter += len(alphas)
    return (np.vstack(Zs_all).astype(np.float32), np.vstack(Zf_all).astype(np.float32),
            np.asarray(seed_orders, dtype=int), np.asarray(cell_ids, dtype=int),
            np.asarray(rep_ids, dtype=int), np.asarray(alpha_vals, dtype=float),
            np.asarray(cluster_ids, dtype=int))


def save_pools(pool_list, outdir, manifest):
    os.makedirs(outdir, exist_ok=True)
    Z_seed_all = np.vstack([p["Z_seed"] for p in pool_list]).astype(np.float32)
    Z_final_all = np.vstack([p["Z_final"] for p in pool_list]).astype(np.float32)
    meta = pd.concat([p["meta"] for p in pool_list], ignore_index=True)
    meta.insert(0, "candidate_global_idx", np.arange(len(meta), dtype=int))
    npz_path = os.path.join(outdir, "baseline_pools_latent.npz")
    csv_path = os.path.join(outdir, "baseline_pools_meta.csv")
    json_path = os.path.join(outdir, "baseline_pools_manifest.json")
    readme_path = os.path.join(outdir, "baseline_pools_README.txt")
    np.savez_compressed(
        npz_path,
        Z_seed=Z_seed_all,
        Z_final=Z_final_all,
        method=meta["method"].astype(str).values,
        candidate_type=meta["candidate_type"].astype(str).values,
        seed_order=meta["seed_order"].astype(int).values,
        cell_idx=meta["cell_idx"].astype(int).values,
        rep_id=meta["rep_id"].astype(int).values,
        source=meta["source"].astype(str).values,
    )
    meta.to_csv(csv_path, index=False)
    manifest = dict(manifest)
    manifest.update({"npz_path": npz_path, "csv_path": csv_path,
                     "n_candidates_total": int(len(meta)),
                     "methods": meta["method"].value_counts().to_dict()})
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(readme_path, "w") as f:
        f.write(
            "Baseline candidate pools for fair ZigZag evaluation.\n\n"
            "This output only creates candidate pools. It intentionally does not evaluate them\n"
            "with a new classifier or a new selection rule.\n\n"
            "Next step:\n"
            "  Run the same evaluation used for ZigZag selected candidates on each method:\n"
            "  - same tumor_proba classifier\n"
            "  - same selection rule: proba_ge_0.7_min_identity\n"
            "  - same fallback: max_tumor_logit\n"
            "  - same residual seed anchoring\n"
            "  - same DE/pathway analysis\n"
        )
    return npz_path, csv_path, json_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--single_cycle_run_dir", default=None)
    p.add_argument("--h5ad_train", required=True)
    p.add_argument("--h5ad_test", default=None)  # accepted for compatibility, not used
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--repo_dir", default="/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat")
    p.add_argument("--label_col", default="status")
    p.add_argument("--healthy_label", default="healthy")
    p.add_argument("--tumor_label", default="tumor")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--selection_rule", default="proba_ge_0.7_min_identity")
    p.add_argument("--gaussian_sigma", type=float, default=-1.0)
    p.add_argument("--interp_alphas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5])
    p.add_argument("--n_tumor_clusters", type=int, default=8)
    p.add_argument("--max_ref_cells", type=int, default=5000)
    p.add_argument("--outdir", default=None)
    p.add_argument("--out_prefix", default="baseline_pools")
    # Ignored old args, kept so old Slurm files run without errors.
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--progress_threshold", type=float, default=None)
    p.add_argument("--matched_progress_tolerance", type=float, default=None)
    p.add_argument("--cloud_pairs", type=int, default=None)
    p.add_argument("--tumor_score_col", default=None)
    p.add_argument("--tumor_thresholds", nargs="+", type=float, default=None)
    p.add_argument("--distance_cols", nargs="+", default=None)
    p.add_argument("--curve_quantiles", type=int, default=None)
    p.add_argument("--knn_k", type=int, default=None)
    p.add_argument("--realism_quantile", type=float, default=None)
    p.add_argument("--n_perm", type=int, default=None)
    p.add_argument("--sparsity_project", action="store_true")
    p.add_argument("--sparsity_min_detect_rate", type=float, default=0.0)
    p.add_argument("--sparsity_max_detect_rate", type=float, default=1.0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.outdir is None:
        args.outdir = os.path.join(args.run_dir, "baseline_pools")
    funcs = import_project_functions(args.repo_dir)

    print("[load train h5ad]")
    A_train = sc.read_h5ad(args.h5ad_train)
    labels = A_train.obs[args.label_col].astype(str).values
    tumor_idx = np.where(labels == args.tumor_label)[0]
    if len(tumor_idx) == 0:
        raise RuntimeError(
            f"No tumor cells found in train h5ad using {args.label_col} == {args.tumor_label}. "
            f"Available labels: {pd.Series(labels).value_counts().to_dict()}"
        )

    print("[encode train latents]")
    Z_train = load_h5ad_latents(args, funcs, args.h5ad_train)
    tumor_sub = rng.choice(tumor_idx, size=min(args.max_ref_cells, len(tumor_idx)), replace=False)
    Z_tumor_ref = Z_train[tumor_sub].astype(np.float32)

    print("[load ZigZag pool]")
    Z_traj, cell_idx, rep_id = load_trajectory_pool(args.run_dir)
    unique_seeds, seed_order = make_seed_order(cell_idx)
    Z_seed_zig = Z_traj[:, 0, :].astype(np.float32)
    Z_final_zig = Z_traj[:, -1, :].astype(np.float32)
    seed_Z_rep = representative_seeds(Z_seed_zig, cell_idx, unique_seeds)

    pool_list = []
    add_pool(pool_list, Z_seed_zig, Z_final_zig, "zigzag_pool", "zigzag_generated", seed_order, cell_idx, rep_id, args.run_dir)

    if args.single_cycle_run_dir and os.path.isdir(args.single_cycle_run_dir):
        print("[load single-cycle pool]")
        Z1_traj, c1, r1 = load_trajectory_pool(args.single_cycle_run_dir)
        seed_to_order = {int(c): i for i, c in enumerate(unique_seeds)}
        keep, sorder = [], []
        for i, c in enumerate(c1):
            if int(c) in seed_to_order:
                keep.append(i)
                sorder.append(seed_to_order[int(c)])
        keep = np.asarray(keep, dtype=int)
        if len(keep):
            add_pool(pool_list, Z1_traj[keep, 0, :], Z1_traj[keep, -1, :], "single_cycle_pool", "single_cycle_generated",
                     np.asarray(sorder, dtype=int), c1[keep], r1[keep], args.single_cycle_run_dir)
        else:
            print("[warning] single-cycle pool has no overlapping seeds; skipped")
    else:
        print("[info] no single_cycle_run_dir found; skipping single_cycle_pool")

    if args.gaussian_sigma > 0:
        sigma = float(args.gaussian_sigma)
        sigma_source = "user_provided"
    else:
        sel_pos = load_selected_positions(args.run_dir, args.selection_rule, cell_idx, rep_id)
        if sel_pos is not None and len(sel_pos):
            disp = Z_final_zig[sel_pos] - Z_seed_zig[sel_pos]
            sigma_source = "selected_zigzag_displacement"
        else:
            disp = Z_final_zig - Z_seed_zig
            sigma_source = "full_zigzag_pool_displacement_fallback"
        sigma = float(np.mean(np.linalg.norm(disp, axis=1)) / np.sqrt(disp.shape[1]))
    print(f"[gaussian] sigma={sigma:.6f} source={sigma_source}")
    Zs_g, Zf_g, so_g, ci_g, ri_g = build_gaussian_pool(args, rng, seed_Z_rep, unique_seeds, sigma)
    add_pool(pool_list, Zs_g, Zf_g, "gaussian_matched_perturbation", "seed_initialized_baseline", so_g, ci_g, ri_g,
             f"sigma={sigma:.6f};source={sigma_source}")

    print("[build tumor centroid interpolation]")
    tumor_centroid = Z_tumor_ref.mean(axis=0).astype(np.float32)
    Zs_c, Zf_c, so_c, ci_c, ri_c, alpha_c = build_centroid_interpolation(seed_Z_rep, unique_seeds, tumor_centroid, args.interp_alphas)
    add_pool(pool_list, Zs_c, Zf_c, "tumor_centroid_interpolation", "seed_initialized_baseline", so_c, ci_c, ri_c,
             "train_tumor_centroid", extra={"alpha": alpha_c})

    print("[build tumor cluster centroid interpolation]")
    k = min(args.n_tumor_clusters, len(Z_tumor_ref))
    km = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=2048, n_init=10)
    km.fit(Z_tumor_ref)
    Zs_cc, Zf_cc, so_cc, ci_cc, ri_cc, alpha_cc, cluster_cc = build_cluster_centroid_interpolation(
        seed_Z_rep, unique_seeds, km.cluster_centers_.astype(np.float32), args.interp_alphas
    )
    add_pool(pool_list, Zs_cc, Zf_cc, "tumor_cluster_centroid_interpolation", "seed_initialized_baseline",
             so_cc, ci_cc, ri_cc, f"train_tumor_kmeans_k={k}", extra={"alpha": alpha_cc, "cluster": cluster_cc})

    manifest = {
        "purpose": "fair seed-initialized baseline candidate pools",
        "run_dir": args.run_dir,
        "single_cycle_run_dir": args.single_cycle_run_dir,
        "h5ad_train": args.h5ad_train,
        "vae_ckpt": args.vae_ckpt,
        "label_col": args.label_col,
        "tumor_label": args.tumor_label,
        "n_unique_seeds": int(len(unique_seeds)),
        "budget": int(args.budget),
        "interp_alphas": list(map(float, args.interp_alphas)),
        "n_tumor_clusters": int(k),
        "gaussian_sigma_used": sigma,
        "gaussian_sigma_source": sigma_source,
        "important_next_step": "Evaluate these pools with the exact same ZigZag evaluation pipeline: same tumor_proba classifier, same selection rule proba_ge_0.7_min_identity, same fallback max_tumor_logit, same residual anchoring, same DE/pathway.",
    }

    npz_path, csv_path, json_path = save_pools(pool_list, args.outdir, manifest)
    print("\n================ DONE ================")
    print("npz:", npz_path)
    print("csv:", csv_path)
    print("json:", json_path)
    print("\nMethod counts:")
    meta = pd.read_csv(csv_path)
    print(meta["method"].value_counts().to_string())


if __name__ == "__main__":
    main()
