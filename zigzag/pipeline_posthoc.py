#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import scanpy as sc

from .common import *
from .common import _load_decoded_subset, _subsample_rows, _geneset_score, _rank_genes
try:
    _subsample_rows
except NameError:
    import numpy as _np
    def _subsample_rows(X, max_n=20000, seed=0):
        X = _np.asarray(X)
        if X.shape[0] <= max_n:
            return X
        rng = _np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        return X[idx]

def run_posthoc(args):
    run_evals = getattr(args, "run_evals", None)
    if run_evals is None:
        run_evals = []
    rh = os.path.join(args.outdir, 'real_h_eval.h5ad')
    rt = os.path.join(args.outdir, 'real_t_eval.h5ad')
    if not (os.path.exists(rh) and os.path.exists(rt)):
        raise FileNotFoundError('Missing real_h_eval.h5ad/real_t_eval.h5ad in outdir. Run pipeline once first.')
    real_h_eval = sc.read_h5ad(rh)
    real_t_eval = sc.read_h5ad(rt)

    # Load summary produced by the run stage
    summary_csv = os.path.join(args.outdir, 'summary_all_runs.csv')
    if not os.path.exists(summary_csv):
        raise FileNotFoundError('Missing summary_all_runs.csv in outdir. Run pipeline once first.')

    # ---- Aggregation & heatmaps ----
    if run_evals:
        df = pd.DataFrame(run_evals)
        summary_csv = os.path.join(args.outdir, "summary_all_runs.csv")
        df.to_csv(summary_csv, index=False)

        for direction in df["direction"].unique():
            for mode in df["mode"].unique():
                sub = df[(df["direction"] == direction) & (df["mode"] == mode)].copy()
                if sub.empty:
                    continue
                for metric in [
                    "knn_desired_lat",
                    "knn_desired_expr",
                    "id_dist_expr_mean",
                    "intra_seed_div_mean",
                    "cand_seed_rank_median",
                    "pairwise_dist_ratio_edit_over_seed",
                    "latent_centroid_score",
                ]:

                    if metric not in sub.columns:
                        continue
                    try:
                        piv = sub.pivot_table(index="t_param", columns="guidance_s", values=metric, aggfunc="mean")
                        plt.figure(figsize=(6, 4))
                        im = plt.imshow(piv.values, aspect="auto", origin="lower")
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        plt.xticks(range(piv.shape[1]), [str(c) for c in piv.columns])
                        plt.yticks(range(piv.shape[0]), [str(r) for r in piv.index])
                        plt.xlabel("guidance_s")
                        plt.ylabel("t_param")
                        plt.title(f"{direction} — {mode} — {metric}")
                        out_png = os.path.join(args.outdir, f"heatmap_{direction}_{mode}_{metric}.png")
                        plt.tight_layout()
                        plt.savefig(out_png, dpi=150)
                        plt.close()
                    except Exception as e:
                        print(f"[warn] heatmap failed for {direction} {mode} {metric}: {e}")


    else:
        print("[summary] no run_eval collected; skip aggregation.")
        return
    # ---- Realismus-Filter & DE (FIXED) ----
    realistic_csv = os.path.join(args.outdir, "realistic_runs.csv")
    consensus_dir = ensure_dir(os.path.join(args.outdir, "consensus_DE"))

    summary_csv = os.path.join(args.outdir, "summary_all_runs.csv")
    df = pd.read_csv(summary_csv)

    mask = (
        (df["knn_desired_lat"] >= args.eval_min_knn) &
        (df["knn_desired_expr"] >= args.eval_min_knn) &
        (df["pairwise_dist_ratio_edit_over_seed"].between(0.7, 1.5)) &
        (df["cand_seed_rank_median"] <= 10.0)
    )


    df_real = df[mask].copy()
    df_real.to_csv(realistic_csv, index=False)
    print(f"[realism] {len(df_real)}/{len(df)} runs passed realism gates → {realistic_csv}")

    if df_real.empty and len(df):
        df_real = df.sort_values("axis_progress_expr", ascending=False).head(3)
        print("[realism] no runs passed thresholds; using top-3 by axis_progress_expr as fallback")
    # --- build DE_real once (eval split), so concordance has a reference ---
    real_src = real_h_eval
    real_tgt = real_t_eval
    # besser: concat nur source/target:
    A = ad.concat([real_src.copy(), real_tgt.copy()], join="inner", merge="same")
    A.obs["grp"] = ["source"]*real_src.n_obs + ["target"]*real_tgt.n_obs
    if args.de_log_norm:
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)



    # rank genes: target vs source
    sc.tl.rank_genes_groups(A, groupby="grp", groups=["target"], reference="source", method="wilcoxon",
                            use_raw=False, pts=True, tie_correct=True)
    r = A.uns["rank_genes_groups"]
    df_real_df = pd.DataFrame({
        "gene": np.array(r["names"]["target"]).astype(str),
        "logfc": np.array(r["logfoldchanges"]["target"]).astype(float),
        "pval_adj": np.array(r["pvals_adj"]["target"]).astype(float),
        "score": np.array(r["scores"]["target"]).astype(float),
    })
    df_real_df.to_csv(os.path.join(args.outdir, "DE_real_target_vs_source.csv"), index=False)


    # Ensure DE_real exists (otherwise concordance step will crash/skip silently)
    de_real_path = os.path.join(args.outdir, "DE_real_target_vs_source.csv")
    de_real = None
    if os.path.exists(de_real_path):
        de_real = pd.read_csv(de_real_path)
    else:
        print(f"[warn] Missing {de_real_path} -> de_concordance will be skipped.")

    # DE requires decoded subsets; skip runs without them
    all_de_src, all_de_tgt = [], []

    # Pre-create columns in df so they exist even if skipped
    for col in ["de_concordance_spearman", "de_concordance_pval", "de_concordance_n"]:
        if col not in df.columns:
            df[col] = np.nan

    if "_load_decoded_subset" not in globals():
        print("[warn] _load_decoded_subset not defined; skipping DE.")
    else:
        for _, row in df.iterrows():
            if not bool(row.get("decoded_available", True)):
                continue

            method = str(row.get("method", "zigzag"))
            mode = str(row["mode"])
            direction = str(row["direction"])
            t_param = int(row["t_param"])
            s_val = float(row["guidance_s"])

            if method == "guided_baseline":
                run_dir = os.path.join(args.outdir, f"guided_{direction}", f"t{t_param}_s{s_val}")
            else:
                if mode == "multi":
                    rounds = int(row.get("multi_pass_rounds", 0))
                    run_dir = os.path.join(args.outdir, f"{mode}_{direction}", f"r{rounds}_t{t_param}_s{s_val}")
                else:
                    run_dir = os.path.join(args.outdir, f"{mode}_{direction}", f"t{t_param}_s{s_val}")


            dec_path = os.path.join(run_dir, "decoded_subset.h5ad")
            if not os.path.exists(dec_path):
                # not fatal; just skip this run
                continue

            fake = _load_decoded_subset(dec_path)

            real_src = real_h_eval if direction == "healthy2tumor" else real_t_eval
            real_tgt = real_t_eval if direction == "healthy2tumor" else real_h_eval

            if args.sparsity_project:
                fake = project_sparsity_gene_wise(
                    fake, real_tgt,
                    min_detect_rate=args.sparsity_min_detect_rate,
                    max_detect_rate=args.sparsity_max_detect_rate
                )

            A_de = _prep_concat_for_de(real_src, real_tgt, fake, do_log_norm=bool(args.de_log_norm))


            if args.marker_pos:
                _geneset_score(A_de, args.marker_pos, outcol="marker_pos_score")
            if args.marker_neg:
                _geneset_score(A_de, args.marker_neg, outcol="marker_neg_score")

            de_vs_tgt = _rank_genes(
                A_de, groups="edited", reference="target",
                top_n=args.de_top_n,
                outfile=os.path.join(run_dir, "DE_edited_vs_target.csv"),
            )
            rounds_val = int(row.get("multi_pass_rounds", 0)) if mode == "multi" else 0

            de_vs_tgt["mode"] = mode
            de_vs_tgt["direction"] = direction
            de_vs_tgt["t_param"] = t_param
            de_vs_tgt["guidance_s"] = s_val
            de_vs_tgt["multi_pass_rounds"] = rounds_val

            all_de_tgt.append(de_vs_tgt)

            de_vs_src = _rank_genes(
                A_de, groups="edited", reference="source",
                top_n=args.de_top_n,
                outfile=os.path.join(run_dir, "DE_edited_vs_source.csv"),
            )
            rounds_val = int(row.get("multi_pass_rounds", 0)) if mode == "multi" else 0

            de_vs_src["mode"] = mode
            de_vs_src["direction"] = direction
            de_vs_src["t_param"] = t_param
            de_vs_src["guidance_s"] = s_val
            de_vs_src["multi_pass_rounds"] = rounds_val

            all_de_src.append(de_vs_src)

            # Concordance: FIX = update df (not some stale run_eval dict)
            if de_real is not None:
                de_edit = pd.read_csv(os.path.join(run_dir, "DE_edited_vs_source.csv"))
                de_conc = de_logfc_concordance(de_real=de_real, de_edit=de_edit, top_n=200)

                rounds_val = int(row.get("multi_pass_rounds", 0)) if mode == "multi" else 0
                nrev_val = int(row.get("n_reverse_steps", t_param))

                sel = (
                    (df["method"].astype(str) == method) &
                    (df["mode"].astype(str) == mode) &
                    (df["direction"].astype(str) == direction) &
                    (df["t_param"].astype(int) == int(t_param)) &
                    (df["guidance_s"].astype(float) == float(s_val)) &
                    (df["multi_pass_rounds"].fillna(0).astype(int) == rounds_val) &
                    (df["n_reverse_steps"].fillna(0).astype(int) == nrev_val)
                )

                if "eta" in df.columns and "eta" in row.index:
                    sel = sel & (df["eta"].astype(float) == float(row["eta"]))
                df.loc[sel, "de_concordance_spearman"] = float(de_conc["rho"])
                df.loc[sel, "de_concordance_pval"] = float(de_conc["pval"])
                df.loc[sel, "de_concordance_n"] = int(de_conc["n_genes"])

    # Write back updated summary (now includes concordance cols if computed)
    df.to_csv(summary_csv, index=False)

    if all_de_tgt:
        de_tgt_all = pd.concat(all_de_tgt, ignore_index=True)
        de_tgt_all.to_csv(os.path.join(consensus_dir, "all_DE_edited_vs_target.csv"), index=False)
        near_target = (
            de_tgt_all.assign(abs_logfc=lambda d: d["logfc"].abs())
                    .groupby("gene")["abs_logfc"].median()
                    .sort_values().head(100)
        )
        near_target.to_csv(os.path.join(consensus_dir, "consensus_near_target_genes.csv"))

    if all_de_src:
        de_src_all = pd.concat(all_de_src, ignore_index=True)
        de_src_all.to_csv(os.path.join(consensus_dir, "all_DE_edited_vs_source.csv"), index=False)
        consistent_shift = (
            de_src_all.groupby("gene")["logfc"].median()
                    .sort_values(ascending=False).head(200)
        )
        consistent_shift.to_csv(os.path.join(consensus_dir, "consensus_shift_genes.csv"))

    if all_de_tgt or all_de_src:
        print("[de] consensus tables written in:", consensus_dir)
    else:
        print("[de] no DE tables written (no decoded runs passed / decoded subsets missing).")

def build_posthoc_parser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--de_log_norm', action='store_true', default=True)
    ap.add_argument('--eval_min_knn', type=float, default=0.20)
    ap.add_argument('--eval_min_centroid', type=float, default=0.55)
    ap.add_argument('--de_top_n', type=int, default=50)
    ap.add_argument('--marker_pos', nargs='*', default=None)
    ap.add_argument('--marker_neg', nargs='*', default=None)
    ap.add_argument('--sparsity_project', action='store_true')
    ap.add_argument('--sparsity_min_detect_rate', type=float, default=0.0)
    ap.add_argument('--sparsity_max_detect_rate', type=float, default=1.0)
    return ap
