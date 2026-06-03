#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zigzag_linear_control_analysis.py

Additional, claim-aligned analysis for CancerZigZag.

This script is NOT another leaderboard. It asks whether ZigZag can be explained
by simple linear tumor-mode controls.

It uses outputs from:
  1) baseline pool generator:
       RUN_DIR/baseline_pools/baseline_pools_latent.npz
       RUN_DIR/baseline_pools/baseline_pools_meta.csv

  2) zigzag_fair_falsification_eval.py:
       RUN_DIR/zigzag_falsification_eval/candidate_scores.csv
       RUN_DIR/zigzag_falsification_eval/selected_per_seed.csv

Analyses
--------
A) Linear tumor-mode explainability
   For each candidate, compute the shortest distance to any line from its seed
   to a tumor-cluster centroid in latent space.

   If a candidate is very close to such a line, it is explainable by simple
   cluster-centroid interpolation.

   Key metric:
       best_line_resid_norm
   = distance to nearest seed->tumor-cluster line / candidate displacement.

   Interpretation:
       close to 0  -> essentially linear tumor-mode movement
       larger      -> less explained by simple cluster-centroid interpolation

B) Successful candidate diversity per seed
   Among candidates with tumor_proba >= threshold:
       - how many successful candidates exist per seed?
       - how many tumor-mode directions are represented?
       - how diverse are successful candidates in latent space?

C) Trajectory nonlinearity for methods with actual trajectories
   For zigzag_pool and single_cycle_pool, load trajectories_latent_all_reps.npz
   and compute:
       - path length / chord length
       - maximum deviation from straight seed->final chord

   This directly tests whether multi-round ZigZag follows a nontrivial path,
   rather than simply performing a straight latent interpolation.

Outputs
-------
RUN_DIR/linear_control_analysis/

  linear_explainability_candidate_scores.csv
  linear_explainability_summary_all_candidates.csv
  linear_explainability_summary_selected.csv
  success_diversity_per_seed.csv
  success_diversity_summary.csv
  trajectory_nonlinearity_selected.csv
  trajectory_nonlinearity_summary.csv
  linear_control_interpretation_flags.csv
  linear_control_meta.json

This analysis is meaningful because tumor_cluster_centroid_interpolation is not
a fully comparable generative baseline; it is a strong linear control. The right
question is whether ZigZag behavior can be reduced to that control.
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

def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def safe_median(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mean_pairwise_distance(X, max_pairs=5000, rng=None):
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        return np.nan
    if rng is None:
        rng = np.random.default_rng(0)

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        ds = []
        for i in range(n):
            d = np.linalg.norm(X[i + 1:] - X[i], axis=1)
            ds.extend(d.tolist())
        return float(np.mean(ds)) if ds else np.nan

    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    ok = i != j
    i = i[ok]
    j = j[ok]
    if len(i) == 0:
        return np.nan
    return float(np.mean(np.linalg.norm(X[i] - X[j], axis=1)))


def residual_to_line(point, start, end, segment=True):
    """
    Distance from point to line start->end.
    Returns:
      distance, normalized_distance, projection_alpha, line_length
    """
    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)

    v = end - start
    w = point - start
    vv = float(np.dot(v, v))
    line_len = float(np.sqrt(max(vv, 0.0)))

    if vv < 1e-12:
        dist = float(np.linalg.norm(point - start))
        disp = float(np.linalg.norm(point - start))
        return dist, dist / (disp + 1e-9), np.nan, line_len

    alpha = float(np.dot(w, v) / vv)
    if segment:
        alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    else:
        alpha_clamped = alpha

    proj = start + alpha_clamped * v
    dist = float(np.linalg.norm(point - proj))
    disp = float(np.linalg.norm(point - start))
    return dist, dist / (disp + 1e-9), alpha, line_len


def load_pool(run_dir):
    pool_dir = Path(run_dir) / "baseline_pools"
    npz_path = pool_dir / "baseline_pools_latent.npz"
    meta_path = pool_dir / "baseline_pools_meta.csv"
    manifest_path = pool_dir / "baseline_pools_manifest.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)

    if "candidate_global_idx" not in meta.columns:
        meta.insert(0, "candidate_global_idx", np.arange(len(meta), dtype=int))

    Z_seed = z["Z_seed"].astype(np.float32)
    Z_final = z["Z_final"].astype(np.float32)

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    return Z_seed, Z_final, meta, manifest


def load_eval_tables(run_dir, eval_dir=None):
    if eval_dir is None:
        eval_dir = Path(run_dir) / "zigzag_falsification_eval"
    else:
        eval_dir = Path(eval_dir)

    scores_path = eval_dir / "candidate_scores.csv"
    selected_path = eval_dir / "selected_per_seed.csv"

    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}. Run zigzag_fair_falsification_eval.py first.")
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing {selected_path}. Run zigzag_fair_falsification_eval.py first.")

    scores = pd.read_csv(scores_path)
    selected = pd.read_csv(selected_path)
    return scores, selected, eval_dir


# ============================================================
# Reconstruct tumor centroids from baseline pool
# ============================================================

def reconstruct_cluster_centroids(meta, Z_final):
    """
    Reconstruct tumor cluster centroids from tumor_cluster_centroid_interpolation
    candidates with alpha ~= 1.0.

    At alpha=1, candidate = tumor cluster centroid.
    """
    m = meta.copy()
    if "alpha" not in m.columns or "cluster" not in m.columns:
        raise RuntimeError(
            "baseline_pools_meta.csv must contain alpha and cluster columns for "
            "tumor_cluster_centroid_interpolation."
        )

    mask = (
        (m["method"].astype(str) == "tumor_cluster_centroid_interpolation")
        & np.isclose(pd.to_numeric(m["alpha"], errors="coerce"), 1.0, atol=1e-6)
    )
    if not mask.any():
        raise RuntimeError("Could not find tumor_cluster_centroid_interpolation rows with alpha == 1.0")

    m1 = m[mask].copy()
    centers = []
    cluster_ids = []
    for cl, g in m1.groupby("cluster", sort=True):
        idx = g["candidate_global_idx"].astype(int).values
        centers.append(Z_final[idx].mean(axis=0))
        cluster_ids.append(int(cl))

    return np.vstack(centers).astype(np.float32), np.asarray(cluster_ids, dtype=int)


def reconstruct_global_centroid(meta, Z_final):
    if "alpha" not in meta.columns:
        return None
    mask = (
        (meta["method"].astype(str) == "tumor_centroid_interpolation")
        & np.isclose(pd.to_numeric(meta["alpha"], errors="coerce"), 1.0, atol=1e-6)
    )
    if not mask.any():
        return None
    idx = meta.loc[mask, "candidate_global_idx"].astype(int).values
    return Z_final[idx].mean(axis=0).astype(np.float32)


# ============================================================
# Linear explainability
# ============================================================

def compute_line_explainability(scores, Z_seed, Z_final, cluster_centers, cluster_ids, global_centroid=None):
    rows = []

    for _, r in scores.iterrows():
        idx = int(r["candidate_global_idx"])
        z0 = Z_seed[idx]
        z1 = Z_final[idx]

        best = {
            "best_line_cluster": None,
            "best_line_resid": np.inf,
            "best_line_resid_norm": np.inf,
            "best_line_alpha": np.nan,
            "best_line_length": np.nan,
        }

        for cl, c in zip(cluster_ids, cluster_centers):
            d, dn, a, ll = residual_to_line(z1, z0, c, segment=True)
            if dn < best["best_line_resid_norm"]:
                best = {
                    "best_line_cluster": int(cl),
                    "best_line_resid": d,
                    "best_line_resid_norm": dn,
                    "best_line_alpha": a,
                    "best_line_length": ll,
                }

        global_resid = np.nan
        global_resid_norm = np.nan
        global_alpha = np.nan
        if global_centroid is not None:
            gd, gdn, ga, _ = residual_to_line(z1, z0, global_centroid, segment=True)
            global_resid = gd
            global_resid_norm = gdn
            global_alpha = ga

        row = r.to_dict()
        row.update(best)
        row.update({
            "global_centroid_line_resid": global_resid,
            "global_centroid_line_resid_norm": global_resid_norm,
            "global_centroid_line_alpha": global_alpha,
            "linear_explainable_0p05": int(best["best_line_resid_norm"] <= 0.05),
            "linear_explainable_0p10": int(best["best_line_resid_norm"] <= 0.10),
            "linear_explainable_0p20": int(best["best_line_resid_norm"] <= 0.20),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_line_explainability(df, selected_ids=None):
    if selected_ids is not None:
        d = df[df["candidate_global_idx"].astype(int).isin(set(map(int, selected_ids)))].copy()
    else:
        d = df.copy()

    rows = []
    for method, g in d.groupby("method", sort=True):
        rows.append({
            "method": method,
            "n_candidates": int(len(g)),
            "mean_best_line_resid_norm": safe_mean(g["best_line_resid_norm"]),
            "median_best_line_resid_norm": safe_median(g["best_line_resid_norm"]),
            "frac_linear_explainable_0p05": safe_mean(g["linear_explainable_0p05"]),
            "frac_linear_explainable_0p10": safe_mean(g["linear_explainable_0p10"]),
            "frac_linear_explainable_0p20": safe_mean(g["linear_explainable_0p20"]),
            "mean_best_line_alpha": safe_mean(g["best_line_alpha"]),
            "mean_global_centroid_line_resid_norm": safe_mean(g["global_centroid_line_resid_norm"]),
            "mean_tumor_proba": safe_mean(g["tumor_proba"]) if "tumor_proba" in g else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================
# Success diversity
# ============================================================

def success_diversity(line_df, Z_final, threshold, rng, max_pairs=5000):
    rows = []
    for (method, seed), g in line_df.groupby(["method", "seed_order"], sort=True):
        succ = g[g["tumor_proba"] >= threshold].copy()
        if len(succ):
            clusters = succ["best_line_cluster"].dropna().astype(int).values
            vc = pd.Series(clusters).value_counts()
            mode_entropy = entropy_from_counts(vc.values)
            n_modes = int(len(vc))
            idx = succ["candidate_global_idx"].astype(int).values
            mpd = mean_pairwise_distance(Z_final[idx], max_pairs=max_pairs, rng=rng)
            line_res = safe_mean(succ["best_line_resid_norm"])
        else:
            mode_entropy = np.nan
            n_modes = 0
            mpd = np.nan
            line_res = np.nan

        rows.append({
            "method": method,
            "seed_order": int(seed),
            "n_candidates": int(len(g)),
            "n_success": int(len(succ)),
            "success_frac": float(len(succ) / max(len(g), 1)),
            "n_success_nearest_tumor_modes": n_modes,
            "success_mode_entropy": mode_entropy,
            "success_pairwise_latent_distance_mean": mpd,
            "success_mean_best_line_resid_norm": line_res,
        })

    per_seed = pd.DataFrame(rows)

    sum_rows = []
    for method, g in per_seed.groupby("method", sort=True):
        sum_rows.append({
            "method": method,
            "mean_n_success": safe_mean(g["n_success"]),
            "median_n_success": safe_median(g["n_success"]),
            "mean_success_frac": safe_mean(g["success_frac"]),
            "mean_n_success_nearest_tumor_modes": safe_mean(g["n_success_nearest_tumor_modes"]),
            "mean_success_mode_entropy": safe_mean(g["success_mode_entropy"]),
            "mean_success_pairwise_latent_distance": safe_mean(g["success_pairwise_latent_distance_mean"]),
            "mean_success_best_line_resid_norm": safe_mean(g["success_mean_best_line_resid_norm"]),
        })
    summary = pd.DataFrame(sum_rows)
    return per_seed, summary


# ============================================================
# Trajectory nonlinearity
# ============================================================

def load_traj_pool(path):
    p = Path(path) / "trajectories_latent_all_reps.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    if "Z_traj" not in z or "cell_idx" not in z or "rep_id" not in z:
        return None
    return {
        "Z_traj": z["Z_traj"].astype(np.float32),
        "cell_idx": z["cell_idx"].astype(int),
        "rep_id": z["rep_id"].astype(int),
    }


def path_nonlinearity(Z_path):
    """
    Measures trajectory deviation from the straight chord start->end.
    """
    Z_path = np.asarray(Z_path, dtype=np.float32)
    if Z_path.ndim != 2 or Z_path.shape[0] < 2:
        return {
            "n_states": Z_path.shape[0] if Z_path.ndim == 2 else 0,
            "chord_length": np.nan,
            "path_length": np.nan,
            "path_over_chord": np.nan,
            "max_line_deviation": np.nan,
            "max_line_deviation_norm": np.nan,
            "mean_line_deviation_norm": np.nan,
        }

    start = Z_path[0]
    end = Z_path[-1]
    chord = float(np.linalg.norm(end - start))
    steps = np.linalg.norm(np.diff(Z_path, axis=0), axis=1)
    path_len = float(np.sum(steps))

    deviations = []
    for p in Z_path:
        d, dn, _, _ = residual_to_line(p, start, end, segment=True)
        deviations.append((d, dn))
    deviations = np.asarray(deviations, dtype=float)

    return {
        "n_states": int(Z_path.shape[0]),
        "chord_length": chord,
        "path_length": path_len,
        "path_over_chord": float(path_len / (chord + 1e-9)),
        "max_line_deviation": float(np.max(deviations[:, 0])),
        "max_line_deviation_norm": float(np.max(deviations[:, 1])),
        "mean_line_deviation_norm": float(np.mean(deviations[:, 1])),
    }


def trajectory_nonlinearity(run_dir, manifest, selected):
    """
    Compute trajectory nonlinearity for zigzag_pool and single_cycle_pool only.
    Other controls are not generated by trajectories.
    """
    rows = []

    method_to_dir = {
        "zigzag_pool": run_dir,
    }

    sc_dir = manifest.get("single_cycle_run_dir")
    if sc_dir:
        method_to_dir["single_cycle_pool"] = sc_dir

    for method, d in method_to_dir.items():
        traj = load_traj_pool(d)
        if traj is None:
            continue

        key_to_idx = {
            f"{int(c)}__{int(r)}": i
            for i, (c, r) in enumerate(zip(traj["cell_idx"], traj["rep_id"]))
        }

        sm = selected[selected["method"] == method].copy()
        for _, r in sm.iterrows():
            key = f"{int(r['cell_idx'])}__{int(r['rep_id'])}"
            if key not in key_to_idx:
                continue
            ti = key_to_idx[key]
            st = path_nonlinearity(traj["Z_traj"][ti])
            row = {
                "method": method,
                "seed_order": int(r["seed_order"]),
                "cell_idx": int(r["cell_idx"]),
                "rep_id": int(r["rep_id"]),
                "candidate_global_idx": int(r["candidate_global_idx"]),
            }
            row.update(st)
            rows.append(row)

    per = pd.DataFrame(rows)
    if len(per) == 0:
        return per, pd.DataFrame()

    sums = []
    for method, g in per.groupby("method", sort=True):
        sums.append({
            "method": method,
            "n_selected_with_trajectory": int(len(g)),
            "mean_path_over_chord": safe_mean(g["path_over_chord"]),
            "median_path_over_chord": safe_median(g["path_over_chord"]),
            "mean_max_line_deviation_norm": safe_mean(g["max_line_deviation_norm"]),
            "mean_chord_length": safe_mean(g["chord_length"]),
            "mean_path_length": safe_mean(g["path_length"]),
        })
    return per, pd.DataFrame(sums)


# ============================================================
# Interpretation flags
# ============================================================

def interpretation_flags(line_sel, success_sum, traj_sum):
    rows = []

    for _, r in line_sel.iterrows():
        method = r["method"]
        rows.append({
            "method": method,
            "selected_not_explained_by_cluster_lines": bool(r["mean_best_line_resid_norm"] > 0.10),
            "selected_mostly_linear_control_like": bool(r["frac_linear_explainable_0p10"] > 0.5),
            "note": (
                "Cluster-centroid is a linear control. If ZigZag has high line residual, "
                "it is less reducible to seed->tumor-mode interpolation. If it has low "
                "line residual, its selected candidates are largely explainable by simple "
                "linear tumor-mode movement."
            ),
        })

    flags = pd.DataFrame(rows)

    if len(success_sum):
        flags = flags.merge(success_sum, on="method", how="left")
    if len(traj_sum):
        flags = flags.merge(traj_sum, on="method", how="left")

    return flags


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--eval_dir", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--success_proba", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max_pairs", type=int, default=5000)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    run_dir = Path(args.run_dir)
    if args.outdir is None:
        args.outdir = run_dir / "linear_control_analysis"
    else:
        args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[load pools]")
    Z_seed, Z_final, meta, manifest = load_pool(run_dir)

    print("[load evaluation tables]")
    scores, selected, eval_dir = load_eval_tables(run_dir, args.eval_dir)

    print("[reconstruct tumor cluster centroids]")
    cluster_centers, cluster_ids = reconstruct_cluster_centroids(meta, Z_final)
    global_centroid = reconstruct_global_centroid(meta, Z_final)

    print("[compute linear explainability]")
    line_scores = compute_line_explainability(scores, Z_seed, Z_final, cluster_centers, cluster_ids, global_centroid)

    selected_ids = selected["candidate_global_idx"].astype(int).values
    line_all_summary = summarize_line_explainability(line_scores, selected_ids=None)
    line_selected_summary = summarize_line_explainability(line_scores, selected_ids=selected_ids)

    print("[compute success diversity]")
    success_per_seed, success_summary = success_diversity(
        line_scores, Z_final, args.success_proba, rng, max_pairs=args.max_pairs
    )

    print("[compute trajectory nonlinearity]")
    traj_per, traj_summary = trajectory_nonlinearity(str(run_dir), manifest, selected)

    print("[interpretation flags]")
    flags = interpretation_flags(line_selected_summary, success_summary, traj_summary)

    paths = {
        "line_scores": args.outdir / "linear_explainability_candidate_scores.csv",
        "line_summary_all": args.outdir / "linear_explainability_summary_all_candidates.csv",
        "line_summary_selected": args.outdir / "linear_explainability_summary_selected.csv",
        "success_per_seed": args.outdir / "success_diversity_per_seed.csv",
        "success_summary": args.outdir / "success_diversity_summary.csv",
        "trajectory_per_selected": args.outdir / "trajectory_nonlinearity_selected.csv",
        "trajectory_summary": args.outdir / "trajectory_nonlinearity_summary.csv",
        "flags": args.outdir / "linear_control_interpretation_flags.csv",
        "meta": args.outdir / "linear_control_meta.json",
    }

    line_scores.to_csv(paths["line_scores"], index=False)
    line_all_summary.to_csv(paths["line_summary_all"], index=False)
    line_selected_summary.to_csv(paths["line_summary_selected"], index=False)
    success_per_seed.to_csv(paths["success_per_seed"], index=False)
    success_summary.to_csv(paths["success_summary"], index=False)
    traj_per.to_csv(paths["trajectory_per_selected"], index=False)
    traj_summary.to_csv(paths["trajectory_summary"], index=False)
    flags.to_csv(paths["flags"], index=False)

    meta_json = {
        "analysis": "linear tumor-mode control analysis",
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "success_proba": args.success_proba,
        "n_cluster_centroids": int(len(cluster_centers)),
        "cluster_ids": cluster_ids.astype(int).tolist(),
        "interpretation": {
            "best_line_resid_norm": "distance to nearest seed->tumor-cluster-centroid line divided by seed->candidate displacement",
            "low_values": "candidate is explainable by simple linear tumor-mode movement",
            "high_values": "candidate is less reducible to simple cluster-centroid interpolation",
            "trajectory_path_over_chord": "values > 1 indicate non-straight trajectory; applies only to methods with saved trajectories",
        },
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta_json, f, indent=2)

    print("\n================ DONE ================")
    for k, v in paths.items():
        print(f"{k}: {v}")

    show = [
        "method",
        "mean_best_line_resid_norm",
        "median_best_line_resid_norm",
        "frac_linear_explainable_0p10",
        "mean_global_centroid_line_resid_norm",
    ]
    show = [c for c in show if c in line_selected_summary.columns]
    print("\n[selected linear explainability]")
    print(line_selected_summary[show].to_string(index=False))

    if len(traj_summary):
        print("\n[trajectory nonlinearity]")
        print(traj_summary.to_string(index=False))


if __name__ == "__main__":
    main()
