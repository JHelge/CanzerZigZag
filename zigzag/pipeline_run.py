#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .common import *
from .common import (
    _latent_centroid_score,
    _subsample_rows,
    _prep_concat_for_bio,
    _load_decoded_subset,
    _prep_concat_for_de,
    _geneset_score,
    _rank_genes,
    plot_umap_expr_seed_gallery,
    plot_umap_expr_all_samples,
    plot_best_vs_identity,
)

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

def generate_from_start(
    z_start_t: torch.Tensor,
    diffusion,
    model,
    clf,
    t_param: int,
    guidance_s: float,
    target_id: int,
    device,
    multi_pass_rounds: int,
    eta: float,
    centroid=None,
    centroid_scale: float = 0.0,
):
    """
    Continue generation from an arbitrary latent start point.
    This supports both:
      - restart from original seed
      - continuation from an intermediate generated sample
    """
    z_curr = z_start_t.clone()

    if multi_pass_rounds <= 0:
        xt = damped_q_sample(diffusion, z_curr, torch.full((z_curr.size(0),), int(t_param), device=device, dtype=torch.long), eta_noise=eta)
        z_out = guided_reverse_pass(
            model=model,
            diffusion=diffusion,
            clf=clf,
            x_start=xt,
            t_start=int(t_param),
            target_id=target_id,
            s=float(guidance_s),
            device=device,
            centroid=centroid,
            centroid_scale=float(centroid_scale),
            step_scale=float(eta),
        )
        return z_out

    for _ in range(int(multi_pass_rounds)):
        tt = torch.full((z_curr.size(0),), int(t_param), device=device, dtype=torch.long)
        xt = damped_q_sample(diffusion, z_curr, tt, eta_noise=eta)
        z_curr = guided_reverse_pass(
            model=model,
            diffusion=diffusion,
            clf=clf,
            x_start=xt,
            t_start=int(t_param),
            target_id=target_id,
            s=float(guidance_s),
            device=device,
            centroid=centroid,
            centroid_scale=float(centroid_scale),
            step_scale=float(eta),
        )

    return z_curr

def _ensure_dense_X(A, name="adata", layer_fallback=None):
    if A is None:
        return A
    if A.X is Ellipsis:
        if layer_fallback is not None and layer_fallback in A.layers:
            A.X = A.layers[layer_fallback]
        else:
            raise ValueError(
                f"{name}.X is Ellipsis (no matrix stored). "
                f"Available layers: {list(A.layers.keys())}"
            )
    A.X = to_numpy_dense(A.X).astype(np.float32, copy=False)
    return A


def _safe_rule_name(x):
    return (
        str(x)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "p")
        .replace(">=", "ge")
        .replace("<=", "le")
        .replace(">", "gt")
        .replace("<", "lt")
    )


def _load_gmt(path):
    """
    Minimal GMT reader:
    pathway_name <tab> description <tab> gene1 <tab> gene2 ...
    """
    gene_sets = {}
    if path is None or not os.path.exists(path):
        return gene_sets

    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gene_sets[parts[0]] = [g for g in parts[2:] if g]

    return gene_sets


def _de_concordance_selected(de_real, de_selected, top_n=200):
    """
    Compare real target-vs-source DE with selected edited-vs-source DE.
    Both tables need columns: gene, logfc.
    """
    out = {
        "n_genes": 0,
        "pearson_all": np.nan,
        "spearman_all": np.nan,
        "pearson_top": np.nan,
        "spearman_top": np.nan,
        "top50_overlap": np.nan,
        "top100_overlap": np.nan,
        "top200_overlap": np.nan,
    }

    if de_real is None or de_selected is None:
        return out
    if "gene" not in de_real.columns or "logfc" not in de_real.columns:
        return out
    if "gene" not in de_selected.columns or "logfc" not in de_selected.columns:
        return out

    a = de_real[["gene", "logfc"]].rename(columns={"logfc": "logfc_real"})
    b = de_selected[["gene", "logfc"]].rename(columns={"logfc": "logfc_selected"})
    m = a.merge(b, on="gene", how="inner").dropna()

    out["n_genes"] = int(len(m))
    if len(m) < 5:
        return out

    out["pearson_all"] = float(m["logfc_real"].corr(m["logfc_selected"], method="pearson"))
    out["spearman_all"] = float(m["logfc_real"].corr(m["logfc_selected"], method="spearman"))

    real_ranked = (
        de_real.assign(abs_logfc=lambda d: d["logfc"].abs())
        .sort_values("abs_logfc", ascending=False)
    )
    sel_ranked = (
        de_selected.assign(abs_logfc=lambda d: d["logfc"].abs())
        .sort_values("abs_logfc", ascending=False)
    )

    top_genes = set(real_ranked.head(int(top_n))["gene"].astype(str))
    mt = m[m["gene"].astype(str).isin(top_genes)].copy()

    if len(mt) >= 5:
        out["pearson_top"] = float(mt["logfc_real"].corr(mt["logfc_selected"], method="pearson"))
        out["spearman_top"] = float(mt["logfc_real"].corr(mt["logfc_selected"], method="spearman"))

    def _overlap(k):
        r = set(real_ranked.head(k)["gene"].astype(str))
        s = set(sel_ranked.head(k)["gene"].astype(str))
        return float(len(r & s) / max(1, len(r)))

    out["top50_overlap"] = _overlap(50)
    out["top100_overlap"] = _overlap(100)
    out["top200_overlap"] = _overlap(200)
    return out


def _score_pathways_selected(A, gene_sets, outfile):
    """
    Scores pathways on source / target / edited cells.
    Uses simple gene-wise centered mean score over pathway genes.
    """
    rows = []

    if not gene_sets:
        return pd.DataFrame(rows)

    for pname, genes in gene_sets.items():
        genes_present = [g for g in genes if g in A.var_names]
        if len(genes_present) < 3:
            continue

        X = A[:, genes_present].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        score = X.mean(axis=1)

        tmp = pd.DataFrame({
            "grp": A.obs["grp"].astype(str).values,
            "score": score,
        })
        means = tmp.groupby("grp")["score"].mean().to_dict()

        source = float(means.get("source", np.nan))
        target = float(means.get("target", np.nan))
        edited = float(means.get("edited", np.nan))

        rows.append({
            "pathway": pname,
            "n_genes": int(len(genes_present)),
            "mean_source": source,
            "mean_target": target,
            "mean_edited": edited,
            "delta_real_target_vs_source": target - source,
            "delta_edited_vs_source": edited - source,
            "edited_fraction_of_real_shift": abs(edited - source) / (abs(target - source) + 1e-9),
            "distance_edited_to_target": abs(edited - target),
            "direction_agree": int(np.sign(target - source) == np.sign(edited - source)),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(
            ["direction_agree", "edited_fraction_of_real_shift"],
            ascending=[False, False],
        )
    df.to_csv(outfile, index=False)
    return df


def _cos_np(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def _write_selected_candidate_biology(
    out_base,
    selected_fake_all,
    real_h_eval,
    real_t_eval,
    direction,
    args,
):
    """
    Runs DE/pathway analysis on selected candidates, not on all generated samples.
    Main rule defaults to args.selected_main_rule.
    """
    try:
        main_rule = str(getattr(args, "selected_main_rule", "proba_ge_0.7_min_identity"))
        rule_safe = _safe_rule_name(main_rule)

        if "selection_rule" not in selected_fake_all.obs.columns:
            print("[selected-bio] skip: selected_fake.obs['selection_rule'] missing")
            return {}

        selected_fake = selected_fake_all[
            selected_fake_all.obs["selection_rule"].astype(str).values == main_rule
        ].copy()

        if selected_fake.n_obs < 2:
            print(f"[selected-bio] skip: too few selected candidates for {main_rule}: n={selected_fake.n_obs}")
            return {}

        real_src = real_h_eval if direction == "healthy2tumor" else real_t_eval
        real_tgt = real_t_eval if direction == "healthy2tumor" else real_h_eval

        A_de = _prep_concat_for_de(
            real_src,
            real_tgt,
            selected_fake,
            do_log_norm=bool(args.de_log_norm),
        )

        # Optional marker scores
        if args.marker_pos:
            _geneset_score(A_de, args.marker_pos, outcol="marker_pos_score")
        if args.marker_neg:
            _geneset_score(A_de, args.marker_neg, outcol="marker_neg_score")

        # Real target-vs-source DE in same concatenated object
        de_real = _rank_genes(
            A_de,
            groups="target",
            reference="source",
            top_n=int(getattr(args, "selected_de_top_n", 2000)),
            outfile=os.path.join(out_base, f"DE_real_target_vs_source_for_selected_{rule_safe}.csv"),
        )

        # Selected edited-vs-source DE
        de_sel_vs_src = _rank_genes(
            A_de,
            groups="edited",
            reference="source",
            top_n=int(getattr(args, "selected_de_top_n", 2000)),
            outfile=os.path.join(out_base, f"DE_selected_{rule_safe}_vs_source.csv"),
        )

        # Selected edited-vs-real-target DE
        de_sel_vs_tgt = _rank_genes(
            A_de,
            groups="edited",
            reference="target",
            top_n=int(getattr(args, "selected_de_top_n", 2000)),
            outfile=os.path.join(out_base, f"DE_selected_{rule_safe}_vs_target.csv"),
        )

        conc = _de_concordance_selected(
            de_real=de_real,
            de_selected=de_sel_vs_src,
            top_n=200,
        )

        summary = {
            "selection_rule": main_rule,
            "n_selected": int(selected_fake.n_obs),
            **conc,
        }

        # Marker summaries if present
        marker_rows = []
        for score_col in ["marker_pos_score", "marker_neg_score"]:
            if score_col in A_de.obs.columns:
                tmp = pd.DataFrame({
                    "grp": A_de.obs["grp"].astype(str).values,
                    "score": A_de.obs[score_col].astype(float).values,
                })
                stats = tmp.groupby("grp")["score"].agg(["mean", "median"]).reset_index()
                stats["score_name"] = score_col
                marker_rows.append(stats)

                means = tmp.groupby("grp")["score"].mean().to_dict()
                summary[f"{score_col}_source_mean"] = float(means.get("source", np.nan))
                summary[f"{score_col}_target_mean"] = float(means.get("target", np.nan))
                summary[f"{score_col}_edited_mean"] = float(means.get("edited", np.nan))

        if marker_rows:
            pd.concat(marker_rows, ignore_index=True).to_csv(
                os.path.join(out_base, f"selected_marker_scores_summary_{rule_safe}.csv"),
                index=False,
            )

        # Optional pathway scoring
        gene_sets = _load_gmt(getattr(args, "pathway_gmt", None))
        if gene_sets:
            pathway_df = _score_pathways_selected(
                A_de,
                gene_sets,
                outfile=os.path.join(out_base, f"pathway_scores_selected_{rule_safe}.csv"),
            )
            if len(pathway_df) > 0:
                summary["pathway_n_tested"] = int(len(pathway_df))
                summary["pathway_direction_agree_frac"] = float(pathway_df["direction_agree"].mean())
                summary["pathway_median_fraction_real_shift"] = float(pathway_df["edited_fraction_of_real_shift"].median())

        pd.DataFrame([summary]).to_csv(
            os.path.join(out_base, f"selected_candidates_de_summary_{rule_safe}.csv"),
            index=False,
        )
        save_json(summary, os.path.join(out_base, f"selected_candidates_de_summary_{rule_safe}.json"))

        print("[selected-bio] wrote:", os.path.join(out_base, f"selected_candidates_de_summary_{rule_safe}.csv"))
        return summary

    except Exception as e:
        print(f"[warn] selected-candidate biology failed: {e}")
        return {}


def _write_selected_successful_trajectories(
    out_base,
    vae,
    device,
    genes,
    selected_df,
    decoded_metrics,
    X_seed_dec_aligned,
    prog_scaler,
    prog_pca,
    prog_w,
    prog_b,
    ref_raw_h,
    ref_raw_t,
    tumor_scaler,
    tumor_pca,
    tumor_clf,
    args,
    direction=None,          # <-- NEU
    real_h_eval=None,        # <-- NEU
    real_t_eval=None,        # <-- NEU
):
    """
    Scores only trajectories belonging to selected candidates.
    Successful trajectories are selected candidates with tumor_proba >= args.selected_success_proba.
    Requires trajectories_latent_all_reps.npz produced by --save_all_rep_trajectories.
    """
    traj_path = os.path.join(out_base, "trajectories_latent_all_reps.npz")
    if not os.path.exists(traj_path):
        print(f"[selected-traj] skip: missing {traj_path}")
        return {}

    try:
        main_rule = str(getattr(args, "selected_main_rule", "proba_ge_0.7_min_identity"))
        success_thr = float(getattr(args, "selected_success_proba", 0.70))
        rule_safe = _safe_rule_name(main_rule)

        zdat = np.load(traj_path)
        Z_traj = zdat["Z_traj"]          # [n_generated, n_states, latent_dim]
        cell_idx_arr = zdat["cell_idx"].astype(int)
        rep_id_arr = zdat["rep_id"].astype(int)

        if "selection_rule" not in selected_df.columns:
            print("[selected-traj] skip: selected_df lacks selection_rule")
            return {}

        selected_main = selected_df[selected_df["selection_rule"].astype(str) == main_rule].copy()
        if selected_main.empty:
            print(f"[selected-traj] skip: no rows for selection rule {main_rule}")
            return {}

        selected_main["selected_success"] = selected_main["tumor_proba"].astype(float) >= success_thr

        key_to_sel = {
            (int(r.cell_idx), int(r.rep_id)): {
                "selected_success": bool(r.selected_success),
                "final_tumor_logit": float(r.tumor_logit),
                "final_tumor_proba": float(r.tumor_proba),
                "final_progress01": float(r.progress01),
                "final_id_dist_expr": float(r.id_dist_expr),
            }
            for r in selected_main.itertuples(index=False)
        }

        keep = [
            i for i in range(len(cell_idx_arr))
            if (int(cell_idx_arr[i]), int(rep_id_arr[i])) in key_to_sel
        ]

        if len(keep) == 0:
            print("[selected-traj] skip: no selected trajectories matched cell_idx/rep_id")
            return {}

        Z_sel = Z_traj[keep].astype(np.float32)  # [n_selected, n_states, latent_dim]
        cell_sel = cell_idx_arr[keep]
        rep_sel = rep_id_arr[keep]
        n_sel, n_states, latent_dim = Z_sel.shape

        # Decode every selected trajectory state
        # Decode every selected trajectory state
        Z_flat = Z_sel.reshape(n_sel * n_states, latent_dim)
        X_flat = decode_latents_in_batches(
            vae,
            Z_flat,
            device,
            batch_size=int(args.batch),
        ).astype(np.float32)

        # IMPORTANT:
        # Match the exact expression-space regime used for selected_candidates_eval.csv.
        # In the main decoded evaluation, X_fake is sparsity-projected before scoring.
        # Therefore, trajectory states must be projected before tumor_logit/proba,
        # progress01, and id_dist_expr are computed.
        if bool(getattr(args, "sparsity_project", False)):
            if direction is None or real_h_eval is None or real_t_eval is None:
                raise ValueError(
                    "Trajectory sparsity projection needs direction, real_h_eval, and real_t_eval."
                )

            target_for_sparsity = real_t_eval if direction == "healthy2tumor" else real_h_eval

            traj_ad = ad.AnnData(
                X=X_flat,
                var=pd.DataFrame(index=genes),
            )

            traj_ad = project_sparsity_gene_wise(
                traj_ad,
                target_for_sparsity,
                min_detect_rate=args.sparsity_min_detect_rate,
                max_detect_rate=args.sparsity_max_detect_rate,
            )

            X_flat = to_numpy_dense(traj_ad.X).astype(np.float32)

        # Score tumor-likeness for every intermediate state
        tumor_logit, tumor_proba = tumor_logreg_score(
            X_query=X_flat,
            scaler=tumor_scaler,
            pca=tumor_pca,
            clf=tumor_clf,
        )

        prog_raw, prog01 = progress_score_01_decoder(
            X_flat,
            prog_scaler,
            prog_pca,
            prog_w,
            prog_b,
            ref_raw_h,
            ref_raw_t,
        )

        # No external seed-expression lookup:
        # for trajectory analysis, identity is measured relative to the decoded
        # round-0 trajectory state. This guarantees:
        # round 0 id_dist_expr = 0.
        #
        # This trajectory-native identity distance can differ slightly from
        # selected_candidates_eval.csv, where identity is computed against the
        # separately evaluated seed expression.
        rows = []
        for i in range(n_sel):
            sid = int(cell_sel[i])
            rid = int(rep_sel[i])
            key = (sid, rid)
            meta = key_to_sel[key]

            z_seq = Z_sel[i]
            z_seed = z_seq[0]
            z_final = z_seq[-1]
            final_vec = z_final - z_seed
            prev_step = None

            x_seed_round0 = X_flat[i * n_states].astype(np.float32)

            for rr in range(n_states):
                flat_idx = i * n_states + rr
                z_curr = z_seq[rr]
                edit_vec = z_curr - z_seed

                if rr == 0:
                    step_vec = np.zeros_like(edit_vec)
                else:
                    step_vec = z_seq[rr] - z_seq[rr - 1]

                cos_final = np.nan if rr == 0 else _cos_np(edit_vec, final_vec)
                cos_prev = np.nan if rr <= 1 or prev_step is None else _cos_np(step_vec, prev_step)

                id_expr_from_round0 = float(np.linalg.norm(X_flat[flat_idx] - x_seed_round0))

                rows.append({
                    "cell_idx": sid,
                    "rep_id": rid,
                    "selection_rule": main_rule,
                    "selected_success": bool(meta["selected_success"]),
                    "round": int(rr),
                    "round_frac": float(rr / max(1, n_states - 1)),
                    "tumor_logit": float(tumor_logit[flat_idx]),
                    "tumor_proba": float(tumor_proba[flat_idx]),
                    "progress_raw": float(prog_raw[flat_idx]),
                    "progress01": float(prog01[flat_idx]),
                    "id_dist_expr": id_expr_from_round0,
                    "id_dist_expr_from_round0": id_expr_from_round0,
                    "latent_dist_from_seed": float(np.linalg.norm(edit_vec)),
                    "latent_step_size": float(np.linalg.norm(step_vec)),
                    "cosine_to_final_direction": cos_final,
                    "cosine_to_previous_step": cos_prev,
                    "final_tumor_logit": float(meta["final_tumor_logit"]),
                    "final_tumor_proba": float(meta["final_tumor_proba"]),
                    "final_progress01": float(meta["final_progress01"]),
                    "final_id_dist_expr": float(meta["final_id_dist_expr"]),
                })

                prev_step = step_vec.copy()

        traj = pd.DataFrame(rows)
        # Sanity check: final round should match selected final candidate approximately
        final_rows = traj[traj["round"] == traj["round"].max()].copy()

        sanity = final_rows[[
            "cell_idx",
            "rep_id",
            "tumor_logit",
            "tumor_proba",
            "progress01",
            "id_dist_expr",
            "final_tumor_logit",
            "final_tumor_proba",
            "final_progress01",
            "final_id_dist_expr",
        ]].copy()

        sanity["absdiff_tumor_logit"] = (
            sanity["tumor_logit"] - sanity["final_tumor_logit"]
        ).abs()

        sanity["absdiff_tumor_proba"] = (
            sanity["tumor_proba"] - sanity["final_tumor_proba"]
        ).abs()

        sanity["absdiff_progress01"] = (
            sanity["progress01"] - sanity["final_progress01"]
        ).abs()

        sanity["absdiff_id_dist_expr"] = (
            sanity["id_dist_expr"] - sanity["final_id_dist_expr"]
        ).abs()

        sanity.to_csv(
            os.path.join(out_base, f"selected_trajectory_final_sanity_{rule_safe}.csv"),
            index=False,
        )

        print(
            "[selected-traj sanity] mean absdiff tumor_proba:",
            float(sanity["absdiff_tumor_proba"].mean()),
        )
        # Pareto per selected trajectory over true intermediate tumor_logit vs id_dist_expr
        pareto_parts = []
        for (sid, rid), g in traj.groupby(["cell_idx", "rep_id"], sort=False):
            pm = pareto_front(
                conversion=g["tumor_logit"].values,
                identity_dist=g["id_dist_expr"].values,
            )
            pareto_parts.append(pd.Series(pm.astype(int), index=g.index))
        traj["trajectory_pareto"] = pd.concat(pareto_parts).sort_index().values

        traj.to_csv(os.path.join(out_base, f"selected_trajectory_metrics_{rule_safe}.csv"), index=False)

        round_summary = (
            traj
            .groupby(["selection_rule", "selected_success", "round"])
            .agg(
                n_states=("cell_idx", "size"),
                n_seeds=("cell_idx", "nunique"),
                mean_tumor_logit=("tumor_logit", "mean"),
                median_tumor_logit=("tumor_logit", "median"),
                mean_tumor_proba=("tumor_proba", "mean"),
                median_tumor_proba=("tumor_proba", "median"),
                mean_progress01=("progress01", "mean"),
                median_progress01=("progress01", "median"),
                mean_id_dist_expr=("id_dist_expr", "mean"),
                median_id_dist_expr=("id_dist_expr", "median"),
                mean_latent_dist_from_seed=("latent_dist_from_seed", "mean"),
                mean_cosine_to_final_direction=("cosine_to_final_direction", "mean"),
                median_cosine_to_final_direction=("cosine_to_final_direction", "median"),
                mean_cosine_to_previous_step=("cosine_to_previous_step", "mean"),
                median_cosine_to_previous_step=("cosine_to_previous_step", "median"),
                pareto_frac=("trajectory_pareto", "mean"),
            )
            .reset_index()
        )
        round_summary.to_csv(os.path.join(out_base, f"selected_trajectory_round_summary_{rule_safe}.csv"), index=False)

        seed_summary = (
            traj
            .groupby(["selection_rule", "selected_success", "cell_idx", "rep_id"])
            .agg(
                start_tumor_logit=("tumor_logit", "first"),
                end_tumor_logit=("tumor_logit", "last"),
                max_tumor_logit=("tumor_logit", "max"),
                start_tumor_proba=("tumor_proba", "first"),
                end_tumor_proba=("tumor_proba", "last"),
                max_tumor_proba=("tumor_proba", "max"),
                start_progress01=("progress01", "first"),
                end_progress01=("progress01", "last"),
                max_progress01=("progress01", "max"),
                start_id_dist_expr=("id_dist_expr", "first"),
                end_id_dist_expr=("id_dist_expr", "last"),
                max_id_dist_expr=("id_dist_expr", "max"),
                mean_cosine_to_final_direction=("cosine_to_final_direction", "mean"),
                median_cosine_to_final_direction=("cosine_to_final_direction", "median"),
                mean_cosine_to_previous_step=("cosine_to_previous_step", "mean"),
                median_cosine_to_previous_step=("cosine_to_previous_step", "median"),
                pareto_frac=("trajectory_pareto", "mean"),
            )
            .reset_index()
        )

        seed_summary["gain_tumor_logit"] = seed_summary["end_tumor_logit"] - seed_summary["start_tumor_logit"]
        seed_summary["gain_tumor_proba"] = seed_summary["end_tumor_proba"] - seed_summary["start_tumor_proba"]
        seed_summary["gain_progress01"] = seed_summary["end_progress01"] - seed_summary["start_progress01"]
        seed_summary["gain_id_dist_expr"] = seed_summary["end_id_dist_expr"] - seed_summary["start_id_dist_expr"]

        seed_summary.to_csv(os.path.join(out_base, f"selected_trajectory_seed_summary_{rule_safe}.csv"), index=False)

        succ = seed_summary[seed_summary["selected_success"].astype(bool)].copy()
        summary = {
            "selection_rule": main_rule,
            "success_proba_threshold": success_thr,
            "n_selected_trajectories": int(seed_summary.shape[0]),
            "n_successful_selected_trajectories": int(succ.shape[0]),
            "frac_successful_selected_trajectories": float(succ.shape[0] / max(1, seed_summary.shape[0])),
            "successful_mean_end_tumor_logit": float(succ["end_tumor_logit"].mean()) if len(succ) else np.nan,
            "successful_mean_end_tumor_proba": float(succ["end_tumor_proba"].mean()) if len(succ) else np.nan,
            "successful_mean_gain_tumor_logit": float(succ["gain_tumor_logit"].mean()) if len(succ) else np.nan,
            "successful_mean_gain_progress01": float(succ["gain_progress01"].mean()) if len(succ) else np.nan,
            "successful_mean_gain_id_dist_expr": float(succ["gain_id_dist_expr"].mean()) if len(succ) else np.nan,
            "successful_mean_cosine_to_final_direction": float(succ["mean_cosine_to_final_direction"].mean()) if len(succ) else np.nan,
        }
        save_json(summary, os.path.join(out_base, f"selected_trajectory_summary_{rule_safe}.json"))

        print("[selected-traj] wrote:", os.path.join(out_base, f"selected_trajectory_metrics_{rule_safe}.csv"))
        return summary

    except Exception as e:
        print(f"[warn] selected trajectory analysis failed: {e}")
        return {}

def damped_q_sample(diffusion, x_start, t, eta_noise: float):
    """
    Create a partially noised state.

    eta_noise=1.0:
        old behavior (full q_sample)

    eta_noise<1.0:
        interpolate between x_start and the fully noised proposal.
    """
    noise = torch.randn_like(x_start)
    xt_full = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
    if eta_noise < 1.0:
        return x_start + eta_noise * (xt_full - x_start)
    return xt_full

def run_continuation_analysis(
    args,
    out_base,
    device,
    diffusion,
    model,
    clf,
    vae,
    genes,
    seed_ids_sub,
    rep_ids_sub,
    X_fake,
    decoded_metrics,
    Z_edit_np,
    Z_seed_lat_aligned,
    X_seed_dec_aligned,
    ref_raw_h,
    ref_raw_t,
    prog_scaler,
    prog_pca,
    prog_w,
    prog_b,
    X_ref_h_eval,
    X_ref_t_eval,
    direction,
    mode,
    t_param,
    guidance_s,
    eta,
    rounds,
    target_id,
    tumor_scaler=None,
    tumor_pca=None,
    tumor_clf=None,
):
    rng = np.random.default_rng(args.seed)

    uniq_seeds = pd.unique(decoded_metrics["cell_idx"])
    if len(uniq_seeds) == 0:
        return

    n_take = min(int(args.continuation_n_seeds), len(uniq_seeds))
    chosen_seeds = rng.choice(uniq_seeds, size=n_take, replace=False)

    rows = []

    for sid in chosen_seeds:
        g = decoded_metrics[decoded_metrics["cell_idx"] == sid].copy()
        if g.empty:
            continue

        seed_row_idx = np.where(seed_ids_sub == sid)[0]
        if len(seed_row_idx) == 0:
            continue
        seed_row_idx = int(seed_row_idx[0])

        # original seed
        z_seed0 = Z_seed_lat_aligned[seed_row_idx:seed_row_idx+1].astype(np.float32)
        x_seed0 = X_seed_dec_aligned[seed_row_idx:seed_row_idx+1].astype(np.float32)

        candidates = []

        # best sample by progress01
        best_idx_global = int(g["tumor_logit"].idxmax())
        candidates.append(("best", best_idx_global))

        # mid sample
        g_mid = g[
            (g["progress01"] >= float(args.continuation_pick_mid_min)) &
            (g["progress01"] <= float(args.continuation_pick_mid_max))
        ].copy()
        if not g_mid.empty:
            mid_idx_global = int(g_mid["id_dist_expr"].idxmin())
            candidates.append(("mid", mid_idx_global))

        # identity-preserving sample
        id_idx_global = int(g["id_dist_expr"].idxmin())
        candidates.append(("identity", id_idx_global))

        seen = set()
        dedup_candidates = []
        for cname, idxg in candidates:
            if idxg not in seen:
                dedup_candidates.append((cname, idxg))
                seen.add(idxg)

        for cand_type, idxg in dedup_candidates:
            row_local = decoded_metrics.index.get_loc(idxg)
            z0 = Z_edit_np[row_local:row_local+1].astype(np.float32)
            x0 = X_fake[row_local:row_local+1].astype(np.float32)

            # step 0 = chosen generated start point
            expr_ratio0 = expr_pca_reference_distance_ratio_score(
                X_query=x0,
                X_ref_healthy=X_ref_h_eval,
                X_ref_tumor=X_ref_t_eval,
                k=50,
                n_pcs=50,
            )
            d_h0 = float(expr_ratio0["dist_h_per_cell"][0])
            d_t0 = float(expr_ratio0["dist_t_per_cell"][0])
            tm0 = float(d_h0 - d_t0)

            pr_raw0, pr010 = progress_score_01_decoder(
                x0, prog_scaler, prog_pca, prog_w, prog_b, ref_raw_h, ref_raw_t
            )
            id0 = float(np.linalg.norm(x0[0] - x_seed0[0]))

            if tumor_scaler is not None and tumor_pca is not None and tumor_clf is not None:
                tl0, tp0 = tumor_logreg_score(
                    X_query=x0,
                    scaler=tumor_scaler,
                    pca=tumor_pca,
                    clf=tumor_clf,
                )
                tl0 = float(tl0[0])
                tp0 = float(tp0[0])
            else:
                tl0 = float("nan")
                tp0 = float("nan")
            rows.append({
                "cell_idx": int(sid),
                "candidate_type": cand_type,
                "step_idx": 0,
                "mode": str(mode),
                "direction": str(direction),
                "t_param": int(t_param),
                "guidance_s": float(guidance_s),
                "eta": float(eta),
                "multi_pass_rounds": int(rounds),
                "progress_raw": float(pr_raw0[0]),
                "progress01": float(pr010[0]),
                "id_dist_expr": float(id0),
                "dist_to_healthy_expr": d_h0,
                "dist_to_tumor_expr": d_t0,
                "tumor_margin": tm0,
                "tumor_logit": tl0,
                "tumor_proba": tp0,

            })

            z_curr = torch.from_numpy(z0).float().to(device)

            for step in range(1, int(args.continuation_steps) + 1):
                with torch.no_grad():
                    z_next = generate_from_start(
                        z_start_t=z_curr,
                        diffusion=diffusion,
                        model=model,
                        clf=clf,
                        t_param=int(t_param),
                        guidance_s=float(guidance_s),
                        target_id=int(target_id),
                        device=device,
                        multi_pass_rounds=int(rounds) if mode == "multi" else 0,
                        eta=float(eta),
                        centroid=None,
                        centroid_scale=0.0,
                    )

                z_next_np = z_next.detach().cpu().numpy().astype(np.float32)
                x_next = decode_latents_in_batches(vae, z_next_np, device, batch_size=1).astype(np.float32)

                pr_raw, pr01 = progress_score_01_decoder(
                    x_next, prog_scaler, prog_pca, prog_w, prog_b, ref_raw_h, ref_raw_t
                )

                expr_ratio = expr_pca_reference_distance_ratio_score(
                    X_query=x_next,
                    X_ref_healthy=X_ref_h_eval,
                    X_ref_tumor=X_ref_t_eval,
                    k=50,
                    n_pcs=50,
                )
                d_h = float(expr_ratio["dist_h_per_cell"][0])
                d_t = float(expr_ratio["dist_t_per_cell"][0])
                tumor_margin = float(d_h - d_t)

                id_dist = float(np.linalg.norm(x_next[0] - x_seed0[0]))
                if tumor_scaler is not None and tumor_pca is not None and tumor_clf is not None:
                    tl, tp = tumor_logreg_score(
                        X_query=x_next,
                        scaler=tumor_scaler,
                        pca=tumor_pca,
                        clf=tumor_clf,
                    )
                    tl = float(tl[0])
                    tp = float(tp[0])
                else:
                    tl = float("nan")
                    tp = float("nan")

                rows.append({
                    "cell_idx": int(sid),
                    "candidate_type": cand_type,
                    "step_idx": int(step),
                    "mode": str(mode),
                    "direction": str(direction),
                    "t_param": int(t_param),
                    "guidance_s": float(guidance_s),
                    "eta": float(eta),
                    "multi_pass_rounds": int(rounds),
                    "progress_raw": float(pr_raw[0]),
                    "progress01": float(pr01[0]),
                    "id_dist_expr": float(id_dist),
                    "dist_to_healthy_expr": d_h,
                    "dist_to_tumor_expr": d_t,
                    "tumor_margin": tumor_margin,
                    "tumor_logit": tl,
                    "tumor_proba": tp,
                })

                z_curr = z_next

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_base, "continuation_chain.csv"),
            index=False
        )
        print("[continuation] wrote:", os.path.join(out_base, "continuation_chain.csv"))

def run_pipeline(args):

    # ------------------------- Mode resolution -------------------------
    two_file_mode = (args.h5ad_train is not None) and (args.h5ad_test is not None)
    if two_file_mode:
        if not (os.path.exists(args.h5ad_train) and os.path.exists(args.h5ad_test)):
            raise FileNotFoundError(f"Missing train/test file: {args.h5ad_train} / {args.h5ad_test}")
    else:
        if args.h5ad is None:
            raise ValueError("Provide either --h5ad (Option A) OR both --h5ad_train and --h5ad_test (2-file mode).")
        if not os.path.exists(args.h5ad):
            raise FileNotFoundError(f"--h5ad not found: {args.h5ad}")

    # Enforce: ZigZag unguided (classifier guidance disabled for zigzag grid)
    args.guidance_s = [0.0]

    set_seeds(args.seed)
    device = dist_util.dev()
    ensure_dir(args.outdir)

    # ------------------------- Load + preprocess -------------------------
    if two_file_mode:
        adata_train = sc.read_h5ad(args.h5ad_train)
        adata_test  = sc.read_h5ad(args.h5ad_test)
        print("[data] TRAIN:", adata_train.shape, "TEST:", adata_test.shape)
        print("[layers TRAIN]", list(adata_train.layers.keys()))
        print("[layers TEST ]", list(adata_test.layers.keys()))
        print("X is Ellipsis?", adata_train.X is Ellipsis)

        A_train = adata_train
        A_test  = adata_test

        # Sanity: same feature space
        if list(map(str, adata_train.var_names)) != list(map(str, adata_test.var_names)):
            raise ValueError("TRAIN and TEST var_names differ. They must be aligned (same genes, same order).")

        # label col in both
        if args.label_col not in adata_train.obs or args.label_col not in adata_test.obs:
            raise ValueError(f"obs['{args.label_col}'] must exist in BOTH train and test h5ads.")

        #print("[preproc] normalize_total + log1p on TRAIN + TEST (for VAE & classifier)")
        #sc.pp.normalize_total(adata_train, target_sum=1e4); sc.pp.log1p(adata_train)
        #sc.pp.normalize_total(adata_test,  target_sum=1e4); sc.pp.log1p(adata_test)

        save_json(
            {"two_file_mode": True, "h5ad_train": args.h5ad_train, "h5ad_test": args.h5ad_test},
            os.path.join(args.outdir, "data_mode.json"),
        )
    else:
        adata = sc.read_h5ad(args.h5ad)
        print("[data] SINGLE:", adata.shape)

        if args.label_col not in adata.obs:
            raise ValueError(f"obs['{args.label_col}'] not found in {args.h5ad}")

        print("[preproc] normalize_total + log1p on adata.X (for VAE & classifier)")
        #sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

        save_json(
            {"two_file_mode": False, "h5ad": args.h5ad, "split_col": args.split_col, "test_frac": args.test_frac, "seed": args.seed},
            os.path.join(args.outdir, "data_mode.json"),
        )

    # ------------------------- Masks / labels -------------------------
    if two_file_mode:
        mask_h_tr, mask_t_tr, info_tr = make_masks(adata_train, args.label_col)
        mask_h_te, mask_t_te, info_te = make_masks(adata_test,  args.label_col)

        print("[labels TRAIN]", info_tr)
        print("[labels TEST ]", info_te)

        # keep names from TRAIN report (assumed same mapping)
        cls0_name = info_tr["class0"]
        cls1_name = info_tr["class1"]
    else:
        mask_h, mask_t, info = make_masks(adata, args.label_col)
        print("[labels]", info)
        cls0_name = info["class0"]
        cls1_name = info["class1"]

    # ------------------------- Split / indices (NO leakage) -------------------------
    rng = np.random.default_rng(args.seed)

    if two_file_mode:
        # local indices per adata
        h_tr = np.where(mask_h_tr.values)[0]
        t_tr = np.where(mask_t_tr.values)[0]
        h_te = np.where(mask_h_te.values)[0]
        t_te = np.where(mask_t_te.values)[0]

        split_info = {
            "mode": "two_file",
            "split_col": "external_files",
            "n_train_h": int(len(h_tr)), "n_train_t": int(len(t_tr)),
            "n_test_h":  int(len(h_te)), "n_test_t":  int(len(t_te)),
            "h5ad_train": args.h5ad_train,
            "h5ad_test":  args.h5ad_test,
        }
        save_json(split_info, os.path.join(args.outdir, "reference_split.json"))
        print("[split two-file]", split_info)

        # REAL refs (eval always from TEST)
        real_h_eval  = adata_test[h_te].copy()
        real_t_eval  = adata_test[t_te].copy()
        real_h_train = adata_train[h_tr].copy()
        real_t_train = adata_train[t_tr].copy()

        # pools + seeds come from Test
        h_pool_idx = h_te
        t_pool_idx = t_te

        seed_h_idx = rng.choice(h_te, size=min(args.n_per_dir, len(h_te)), replace=False)
        seed_t_idx = rng.choice(t_te, size=min(args.n_per_dir, len(t_te)), replace=False)

        A_seed = adata_test
        A_eval = adata_test

    else:
        # groupwise split (Option A)
        h_tr, h_te, t_tr, t_te, split_info = groupwise_split_indices(
            adata, mask_h, mask_t, split_col=args.split_col, test_frac=args.test_frac, seed=args.seed
        )
        save_json(split_info, os.path.join(args.outdir, "reference_split.json"))
        print("[split]", split_info)

        # eval refs (fallback if tiny)
        real_h_train = adata[h_tr].copy()
        real_t_train = adata[t_tr].copy()
        real_h_eval  = adata[h_te].copy() if len(h_te) > 10 else real_h_train
        real_t_eval  = adata[t_te].copy() if len(t_te) > 10 else real_t_train

        # pools (prefer eval pool if exists; else train)
        h_pool_idx = h_te if len(h_te) > 10 else h_tr
        t_pool_idx = t_te if len(t_te) > 10 else t_tr

        seed_h_idx = rng.choice(h_tr, size=min(args.n_per_dir, len(h_tr)), replace=False)
        seed_t_idx = rng.choice(t_tr, size=min(args.n_per_dir, len(t_tr)), replace=False)

        A_seed = adata
        A_eval = adata

    # ------------------------- VAE gene order handling -------------------------
    gene_order_path = args.vae_gene_order_tsv
    if gene_order_path is None:
        model_dir = args.vae_ckpt if os.path.isdir(args.vae_ckpt) else os.path.dirname(args.vae_ckpt)
        cand = os.path.join(model_dir, "gene_order.tsv")
        if os.path.exists(cand):
            gene_order_path = cand

    genes_model = None
    var_names_seed = list(map(str, A_seed.var_names))

    if gene_order_path is not None and os.path.exists(gene_order_path):
        genes_model = pd.read_csv(gene_order_path, sep="\t", header=None)[0].astype(str).values.tolist()
        missing_genes = sorted(set(genes_model) - set(var_names_seed))
        present = len(genes_model) - len(missing_genes)
        print(f"[VAE input] using gene_order.tsv ({len(genes_model)} genes). present={present} missing_filled0={len(missing_genes)}")
        save_json(
            {"gene_order_tsv": gene_order_path, "n_model_genes": len(genes_model), "n_present": present,
             "n_missing": len(missing_genes), "missing_first50": missing_genes[:50]},
            os.path.join(args.outdir, "vae_gene_order_report.json"),
        )
    else:
        if args.require_gene_order:
            raise RuntimeError("require_gene_order=True but gene_order.tsv not found. Provide --vae_gene_order_tsv.")
        print("[VAE input] gene_order.tsv not found -> using adata.var order as-is (MAKE SURE THIS IS CORRECT).")

    n_vae_genes = len(genes_model) if genes_model is not None else A_seed.n_vars
    genes = genes_model if genes_model is not None else list(map(str, A_seed.var_names))

    # ------------------------- VAE -------------------------
    vae = load_VAE(args.vae_ckpt, num_gene=n_vae_genes, hidden_dim=args.latent_dim).eval().to(device)
    robust_load_vae(vae, args.vae_ckpt)

    # ------------------------- Diffusion -------------------------
    diff_cfg = model_and_diffusion_defaults()
    model, diffusion = create_model_and_diffusion(**diff_cfg)
    model.load_state_dict(torch.load(args.diff_ckpt, map_location="cpu"))
    model.eval().to(device)
    print(f"✅ Diffusion model ready | steps={diffusion.num_timesteps}")

    # ------------------------- Classifier -------------------------
    clf_path = resolve_classifier_ckpt(args.clf_ckpt)
    clf_cfg = classifier_and_diffusion_defaults()
    clf_cfg.update(dict(input_dim=args.latent_dim, num_class=2))
    clf, _ = create_classifier_and_diffusion(**clf_cfg)

    sd = torch.load(clf_path, map_location="cpu")
    if isinstance(sd, dict) and "param_groups" in sd and "state" in sd:
        raise RuntimeError(f"Provided checkpoint is an optimizer state, not model weights: {clf_path}")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = clf.load_state_dict(sd, strict=False)
    if missing:
        print(f"[clf] warning: missing keys: {len(missing)} -> {missing[:5]}")
    if unexpected:
        print(f"[clf] warning: unexpected keys: {len(unexpected)} -> {unexpected[:5]}")
    clf.eval().to(device)
    manifest = get_runtime_manifest(args, A_seed, device)
    save_json(manifest, os.path.join(args.outdir, "runtime_manifest.json"))

    # ------------------------- Encode latents -------------------------
    if two_file_mode:
        # IMPORTANT: use the SAME preprocessing as used in your training loader
        # If your training loader normalizes+log1p, set these True here too.
        Z_tr_all, info_trZ = load_latents_from_h5ad(
            h5ad_path=args.h5ad_train,
            vae_ckpt=args.vae_ckpt,
            hidden_dim=args.latent_dim,
            normalize_total=True,
            target_sum=1e4,
            log1p=True,
            layer=None,          # or "counts" if you want to encode from counts layer
            encode_batch=4096,
            return_obs=False,
            return_var_names=False,
        )

        Z_te_all, info_teZ = load_latents_from_h5ad(
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

        save_json(
            {"Z_train_shape": [int(Z_tr_all.shape[0]), int(Z_tr_all.shape[1])],
            "Z_test_shape":  [int(Z_te_all.shape[0]), int(Z_te_all.shape[1])]},
            os.path.join(args.outdir, "latent_shapes.json"),
        )

        Z_train_all = Z_tr_all
        Z_test_all  = Z_te_all

        Z_seed_all = Z_test_all
        Z_eval_all = Z_test_all

    else:
        Z_all, info_Z = load_latents_from_h5ad(
            h5ad_path=args.h5ad,
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

        save_json({"Z_shape": [int(Z_all.shape[0]), int(Z_all.shape[1])]},
                os.path.join(args.outdir, "latent_shapes.json"))

        Z_seed_all = Z_all
        Z_eval_all = Z_all



    # Helpers: consistent eval refs (latent) without leaking
    def get_eval_ref_latents(direction: str):
        """
        Returns (Z_ref_h, Z_ref_t) in latent space for evaluation.
        In two-file-mode: always from TEST.
        In single-file: prefer held-out test split if non-tiny, else train.
        """
        if two_file_mode:
            return Z_eval_all[h_te], Z_eval_all[t_te]
        else:
            hh = h_te if len(h_te) > 10 else h_tr
            tt = t_te if len(t_te) > 10 else t_tr
            return Z_eval_all[hh], Z_eval_all[tt]

    # ------------------------- Pools (latent+decoded) -------------------------
    Z_pool_h_lat = Z_seed_all[h_pool_idx]
    Z_pool_t_lat = Z_seed_all[t_pool_idx]

    X_pool_h_dec = decode_latents_in_batches(vae, Z_pool_h_lat, device, batch_size=args.batch).astype(np.float32)
    X_pool_t_dec = decode_latents_in_batches(vae, Z_pool_t_lat, device, batch_size=args.batch).astype(np.float32)

    if args.sparsity_project:
        poolH_ad = ad.AnnData(X=X_pool_h_dec, var=pd.DataFrame(index=genes))
        poolT_ad = ad.AnnData(X=X_pool_t_dec, var=pd.DataFrame(index=genes))

        poolH_ad = project_sparsity_gene_wise(
            poolH_ad, A_seed[h_pool_idx],
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )
        poolT_ad = project_sparsity_gene_wise(
            poolT_ad, A_seed[t_pool_idx],
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )

        X_pool_h_dec = to_numpy_dense(poolH_ad.X).astype(np.float32)
        X_pool_t_dec = to_numpy_dense(poolT_ad.X).astype(np.float32)

    # ------------------------- Seeds subsets (TEST seeds only) -------------------------
    Z_seed_h = Z_seed_all[seed_h_idx]
    Z_seed_t = Z_seed_all[seed_t_idx]
    Z_h_all = Z_seed_h
    Z_t_all = Z_seed_t

    # Seeds subsets (TEST seeds only)
    real_h_subset = A_seed[seed_h_idx].copy()
    real_t_subset = A_seed[seed_t_idx].copy()

    real_h_subset = _ensure_dense_X(real_h_subset, "real_h_subset", layer_fallback=None)
    real_t_subset = _ensure_dense_X(real_t_subset, "real_t_subset", layer_fallback=None)



    # ------------------------- Classifier orientation (robust) -------------------------
    probe_h = torch.from_numpy(Z_seed_h).float().to(device)
    probe_t = torch.from_numpy(Z_seed_t).float().to(device)
    t0_h = torch.zeros(probe_h.size(0), dtype=torch.long, device=device)
    t0_t = torch.zeros(probe_t.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        lh = clf(probe_h, t0_h).mean(dim=0).cpu().numpy()
        lt = clf(probe_t, t0_t).mean(dim=0).cpu().numpy()

    delta = lt - lh
    id_tumor = int(np.argmax(delta))
    id_healthy = 1 - id_tumor
    print("[clf orient] mean logits healthy:", lh, "tumor:", lt)
    print(f"[clf orient] chose id_tumor={id_tumor}, id_healthy={id_healthy}, delta={delta}")

    with torch.no_grad():
        p_h = torch.softmax(clf(probe_h, t0_h), dim=1)[:, id_tumor].mean().item()
        p_t = torch.softmax(clf(probe_t, t0_t), dim=1)[:, id_tumor].mean().item()
        mean_logit_h = clf(probe_h, t0_h)[:, id_tumor].mean().item()
        mean_logit_t = clf(probe_t, t0_t)[:, id_tumor].mean().item()
    print(f"[clf sanity @t=0] mean P(tumor) | healthy={p_h:.4f} | tumor={p_t:.4f}")
    print(f"[clf sanity @t=0] mean tumor-logit | healthy={mean_logit_h:.4f} | tumor={mean_logit_t:.4f}")

    # ------------------------- Decode TRAIN refs for progress axis / ref fits -------------------------
    Z_ref_h_lat_train = Z_train_all[h_tr]
    Z_ref_t_lat_train = Z_train_all[t_tr]

    print(f"[ref-train] decoding real refs: healthy={Z_ref_h_lat_train.shape[0]}, tumor={Z_ref_t_lat_train.shape[0]}")

    X_ref_h_dec = decode_latents_in_batches(vae, Z_ref_h_lat_train, device, batch_size=args.batch).astype(np.float32)
    X_ref_t_dec = decode_latents_in_batches(vae, Z_ref_t_lat_train, device, batch_size=args.batch).astype(np.float32)

    refH_ad = ad.AnnData(X=X_ref_h_dec, var=pd.DataFrame(index=genes))
    refT_ad = ad.AnnData(X=X_ref_t_dec, var=pd.DataFrame(index=genes))

    if args.sparsity_project:
        refH_ad = project_sparsity_gene_wise(
            refH_ad, A_train[h_tr],
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )
        refT_ad = project_sparsity_gene_wise(
            refT_ad, A_train[t_tr],
            min_detect_rate=args.sparsity_min_detect_rate,
            max_detect_rate=args.sparsity_max_detect_rate,
        )

    X_ref_h_dec = to_numpy_dense(refH_ad.X).astype(np.float32)
    X_ref_t_dec = to_numpy_dense(refT_ad.X).astype(np.float32)

    mu_h = X_pool_h_dec.mean(axis=0).astype(np.float32)
    mu_t = X_pool_t_dec.mean(axis=0).astype(np.float32)

    # ------------------------- Fit progress axis (decoder-space, TRAIN refs only) -------------------------
    Xh_prog = _subsample_rows(X_ref_h_dec, max_n=20000, seed=args.seed)
    Xt_prog = _subsample_rows(X_ref_t_dec, max_n=20000, seed=args.seed)
    prog_scaler, prog_pca, prog_w, prog_b, ref_raw_h, ref_raw_t = fit_progress_axis_decoder_space(
        Xh_prog, Xt_prog, n_pcs=128
    )
    print("[progress-axis] fitted on decoder-space real refs (TRAIN)")

    def autoencode_subset(vae_obj, adata_subset, device_):
        # .X ist bei dir vorhanden (kein Ellipsis)
        if genes_model is not None:
            X = build_vae_input_from_adata(adata_subset, genes_model).astype(np.float32)
        else:
            X = to_numpy_dense(adata_subset.X).astype(np.float32)

        Z = encode_vae(vae_obj, X, device_)
        Z_t = torch.from_numpy(Z).float().to(device_)
        with torch.no_grad():
            X_rec = vae_obj.decoder(Z_t).detach().cpu().numpy().astype(np.float32, copy=False)

        return ad.AnnData(X=X_rec, var=pd.DataFrame(index=genes))


    # ------------------------- Latent centroids (based on TRAIN seeds) -------------------------
    c_h = torch.from_numpy(Z_h_all.mean(axis=0, keepdims=True)).float().to(device)
    c_t = torch.from_numpy(Z_t_all.mean(axis=0, keepdims=True)).float().to(device)

    # ------------------------- Directions + configs -------------------------
    directions = ["healthy2tumor"] if args.only_healthy2tumor else ["healthy2tumor", "tumor2healthy"]
    configs = []

    for d in directions:
        for t0 in args.t_start:
            for s in args.guidance_s:
                configs.append(("single", d, t0, s, None))

    if args.multi_pass_rounds > 0 or args.multi_pass_rounds_grid is not None or args.multi_pass_t_grid is not None:
        rounds_list = args.multi_pass_rounds_grid if args.multi_pass_rounds_grid is not None else [args.multi_pass_rounds]
        tpass_list = args.multi_pass_t_grid if args.multi_pass_t_grid is not None else args.multi_pass_t

        for d in directions:
            for r in rounds_list:
                if r <= 0:
                    continue
                for t_pass in tpass_list:
                    for s in args.guidance_s:
                        configs.append(("multi", d, int(t_pass), float(s), int(r)))

    run_evals = []

    # ------------------------- Guided baseline (optional; safe for both modes) -------------------------
    if args.run_guided_baseline:
        # seeds from TEST (or adata in single-file)
        if directions and "healthy2tumor" in directions:
            run_guided_baseline_for_direction(
                direction="healthy2tumor",
                seed_idx=seed_h_idx,            # indices into A_seed / Z_seed_all
                target_id=id_tumor,
                out_root=args.outdir,
                args=args,
                device=device,
                model=model,
                diffusion=diffusion,
                clf=clf,
                Z_seed_source_all=Z_seed_all,
                Z_eval_ref_all=Z_eval_all,
                Z_seed_src=Z_seed_all[seed_h_idx],   # <-- ADD
                h_tr=h_tr, t_tr=t_tr,
                h_te=h_te, t_te=t_te,
                vae=vae,
                genes=genes,
                real_h_eval=real_h_eval,        # eval always from TEST in 2-file
                real_t_eval=real_t_eval,
            )

        if (not args.only_healthy2tumor) and ("tumor2healthy" in directions):
            run_guided_baseline_for_direction(
                direction="tumor2healthy",
                seed_idx=seed_t_idx,
                target_id=id_healthy,
                out_root=args.outdir,
                args=args,
                device=device,
                model=model,
                diffusion=diffusion,
                clf=clf,
                Z_seed_source_all=Z_seed_all,
                Z_eval_ref_all=Z_eval_all,
                Z_seed_src=Z_seed_all[seed_t_idx],
                h_tr=h_tr, t_tr=t_tr,
                h_te=h_te, t_te=t_te,
                vae=vae,
                genes=genes,
                real_h_eval=real_h_eval,
                real_t_eval=real_t_eval,
            )

        # Collect guided baseline run_eval.json files ONCE
        guided_eval_paths = glob.glob(os.path.join(args.outdir, "guided_*", "**", "run_eval.json"), recursive=True)
        for pth in guided_eval_paths:
            try:
                with open(pth, "r") as f:
                    run_evals.append(json.load(f))
            except Exception as e:
                print(f"[warn] could not read {pth}: {e}")

    # -------------------------------------------------------------------
    # ... ab hier kommt dein GRID-RUNS Loop (den hast du schon)
    # WICHTIG für später im Loop:
    #   - überall Z_all -> Z_seed_all (für seeds/pools/edits)
    #   - eval refs latent via get_eval_ref_latents()
    #   - A_seed/A_eval statt adata
    # -------------------------------------------------------------------


    # ============================================================
    # ------------------------ GRID-RUNS -------------------------
    # ============================================================
    for mode, direction, t_param, s, rounds in configs:

        # --- pretty desc + outdir ---
        if mode == "multi":
            assert rounds is not None
            t_desc = f"{rounds}x{int(t_param)}"
            out_base = ensure_dir(os.path.join(args.outdir, f"{mode}_{direction}", f"r{int(rounds)}_t{int(t_param)}_s{float(s)}"))
            t_pass = int(t_param)
        else:
            t_desc = f"{int(t_param)}"
            out_base = ensure_dir(os.path.join(args.outdir, f"{mode}_{direction}", f"t{int(t_param)}_s{float(s)}"))
            t_pass = None  # not used for single

        print(f"\n[run] mode={mode} | {direction} | t={t_desc} | s={s}")

# --- pick seeds/targets/centroids ---
# two_file_mode: seeds come from TEST (generalization), pools for baselines can come from TRAIN/TEST depending on baseline
        if direction == "healthy2tumor":
            seed_idx = seed_h_idx              # indices into A_seed / Z_seed_all
            Z_seed  = Z_seed_h                 # TEST seed latents (ordered like seed_idx)
            target_id = id_tumor
            centroid  = c_t
            mu_src, mu_tgt = mu_h, mu_t
            src_label_qc, tgt_label_qc = "healthy_real", "tumor_real"
            Z_src_seed = Z_h_all
            Z_tgt_seed = Z_t_all
        else:
            seed_idx = seed_t_idx
            Z_seed  = Z_seed_t
            target_id = id_healthy
            centroid  = c_h
            mu_src, mu_tgt = mu_t, mu_h
            src_label_qc, tgt_label_qc = "tumor_real", "healthy_real"
            Z_src_seed = Z_t_all
            Z_tgt_seed = Z_h_all

        # --- eval refs (latent) helper ---
        # returns (Z_ref_h_lat_eval, Z_ref_t_lat_eval) WITHOUT leakage:
        #   - 2-file: always TEST
        #   - 1-file: prefer held-out test split if non-tiny else train
        Z_ref_h_lat_eval, Z_ref_t_lat_eval = get_eval_ref_latents(direction)

        # --- tracking (only makes sense for multi + healthy2tumor) ---
        track_this_run = (mode == "multi" and args.track_seeds > 0 and direction == "healthy2tumor")
        traj_dump = None
        traj_seed_global = None
        traj_rounds = None

        if track_this_run:
            # pool among TRAIN seeds you are editing (latent space)
            Z_pool = Z_seed_all[seed_idx]

            pick_local = pick_boundary_central_outlier_seeds(
                Z_pool=Z_pool,
                Z_ref_h=Z_ref_h_lat_eval,
                Z_ref_t=Z_ref_t_lat_eval,
                k=50,
            )

            traj_seed_global = seed_idx[pick_local]  # global indices into A_seed / Z_seed_all
            traj_rounds = min(int(rounds), int(args.track_rounds_max))
            traj_dump = []  # list of latent states (round 0..R)

            z_track = torch.from_numpy(Z_seed_all[traj_seed_global]).float().to(device)
            z_curr_t = z_track.clone()
            traj_dump.append(z_curr_t.detach().cpu().numpy().astype(np.float32))

            for rr in range(traj_rounds):
                eta = float(getattr(args, "eta", 1.0))
                t = torch.full((z_curr_t.size(0),), int(t_param), device=device, dtype=torch.long)

                # Damped noising
                xt = damped_q_sample(diffusion, z_curr_t, t, eta_noise=eta)

                # Damped reverse pass
                z_curr_t = guided_reverse_pass(
                    model, diffusion, clf,
                    xt, int(t_param), target_id, float(s), device,
                    centroid=centroid,
                    centroid_scale=args.centroid_guidance_scale,
                    step_scale=eta,
                    debug=False,
                )

                traj_dump.append(z_curr_t.detach().cpu().numpy().astype(np.float32))
        # ============================================================
        # --------------------- generate edits -----------------------
        # ============================================================
        cosine_sim, l2_dist = [], []
        edited_latents, edited_seed_ids, edited_rep_ids = [], [], []
        save_all_rep_trajectories = (
            bool(getattr(args, "save_all_rep_trajectories", False))
            and mode == "multi"
            and direction == "healthy2tumor"
        )
        all_rep_traj_chunks = []
        all_rep_traj_seed_ids = []
        all_rep_traj_rep_ids = []
        for rep in range(args.n_reps_per_seed):
            for start in range(0, len(seed_idx), args.batch):
                sl = slice(start, min(len(seed_idx), start + args.batch))
                batch_seed_ids = seed_idx[sl]
                z_orig = torch.from_numpy(Z_seed_all[batch_seed_ids]).float().to(device)  # TEST seeds

                z_curr = z_orig.clone()
                debug_this_batch = (start == 0 and rep == 0)

                if mode == "single":
                    t0 = int(t_param)
                    if t0 <= 0:
                        x_final = z_curr
                    else:
                        eta = float(getattr(args, "eta", 1.0))
                        t = torch.full((z_curr.size(0),), t0, device=device, dtype=torch.long)

                        # Damped noising
                        xt = damped_q_sample(diffusion, z_curr, t, eta_noise=eta)

                        # Damped reverse pass
                        x_final = guided_reverse_pass(
                            model, diffusion, clf,
                            xt, t0, target_id, float(s), device,
                            centroid=centroid,
                            centroid_scale=args.centroid_guidance_scale,
                            step_scale=eta,
                            debug=debug_this_batch,
                        )
                
                else:
                    # multi: repeated damped (q_sample + reverse)
                    assert rounds is not None
                    eta = float(getattr(args, "eta", 1.0))

                    if save_all_rep_trajectories:
                        traj_states = [z_curr.detach().cpu().numpy().astype(np.float32)]
                    else:
                        traj_states = None

                    for _rr in range(int(rounds)):
                        t = torch.full((z_curr.size(0),), int(t_param), device=device, dtype=torch.long)

                        # Damped noising
                        xt = damped_q_sample(diffusion, z_curr, t, eta_noise=eta)

                        # Damped reverse pass
                        z_curr = guided_reverse_pass(
                            model, diffusion, clf,
                            xt, int(t_param), target_id, float(s), device,
                            centroid=centroid,
                            centroid_scale=args.centroid_guidance_scale,
                            step_scale=eta,
                            debug=False,
                        )

                        if save_all_rep_trajectories:
                            traj_states.append(z_curr.detach().cpu().numpy().astype(np.float32))

                    x_final = z_curr

                    if save_all_rep_trajectories:
                        # list of [batch, latent_dim] -> [batch, rounds+1, latent_dim]
                        traj_batch = np.stack(traj_states, axis=1).astype(np.float32)
                        all_rep_traj_chunks.append(traj_batch)
                        all_rep_traj_seed_ids.append(batch_seed_ids.copy())
                        all_rep_traj_rep_ids.append(np.full((batch_seed_ids.shape[0],), rep, dtype=np.int32))

                x_np = x_final.detach().cpu().numpy().astype(np.float32, copy=False)
                edited_latents.append(x_np)
                edited_seed_ids.append(batch_seed_ids.copy())
                edited_rep_ids.append(np.full((batch_seed_ids.shape[0],), rep, dtype=np.int32))

                z0_np = z_orig.detach().cpu().numpy().astype(np.float32, copy=False)
                denom = (np.linalg.norm(z0_np, axis=1) * np.linalg.norm(x_np, axis=1) + 1e-9)
                cosine_sim.extend(((z0_np * x_np).sum(1) / denom).tolist())
                l2_dist.extend((np.linalg.norm(x_np - z0_np, axis=1)).tolist())

        rep_ids = np.concatenate(edited_rep_ids, axis=0)
        Z_edit = np.vstack(edited_latents).astype(np.float32, copy=False)
        seed_ids = np.concatenate(edited_seed_ids, axis=0).astype(int, copy=False)
        if save_all_rep_trajectories and all_rep_traj_chunks:
            Z_traj_all = np.concatenate(all_rep_traj_chunks, axis=0).astype(np.float32)
            traj_cell_idx = np.concatenate(all_rep_traj_seed_ids, axis=0).astype(np.int32)
            traj_rep_id = np.concatenate(all_rep_traj_rep_ids, axis=0).astype(np.int32)

            np.savez_compressed(
                os.path.join(out_base, "trajectories_latent_all_reps.npz"),
                Z_traj=Z_traj_all,
                cell_idx=traj_cell_idx,
                rep_id=traj_rep_id,
                r=int(rounds) if mode == "multi" else 0,
                t_param=int(t_param),
                eta=float(getattr(args, "eta", 1.0)),
                guidance_s=float(s),
            )
            print("[trajectory-all] wrote:", os.path.join(out_base, "trajectories_latent_all_reps.npz"), Z_traj_all.shape)
        # ============================================================
        # -------- seed-specificity metrics (paired per seed) ---------
        # ============================================================
        first_row_for_seed = {}
        for i, sid in enumerate(seed_ids):
            if sid not in first_row_for_seed:
                first_row_for_seed[sid] = i
        paired_rows = np.array([first_row_for_seed[sid] for sid in seed_idx if sid in first_row_for_seed], dtype=int)
        Z_edit_paired = Z_edit[paired_rows]

        seed_pres = seed_nn_preservation(Z_src=Z_seed, Z_edit=Z_edit_paired, k=30)
        seed_rank = seed_retrieval_rank(Z_src=Z_seed, Z_edit=Z_edit_paired)

        src_pw = pairwise_dist_summary(Z_seed, seed=args.seed)
        edt_pw = pairwise_dist_summary(Z_edit, seed=args.seed)
        collapse_ratio = (edt_pw["pairwise_dist_mean"] / (src_pw["pairwise_dist_mean"] + 1e-9))

        # latent ref-only kNN (eval refs as defined above)
        knn_ref_mean, _ = real_reference_knn_target_fraction(
            Z_query=Z_edit,
            Z_ref_healthy=Z_ref_h_lat_eval,
            Z_ref_tumor=Z_ref_t_lat_eval,
            k=50,
        )
        desired_latent_knn = float(knn_ref_mean) if direction == "healthy2tumor" else float(1.0 - knn_ref_mean)
        latent_knn = desired_latent_knn

        seed_spec = {}
        seed_spec["realref_knn_class1_frac_mean"] = float(knn_ref_mean)
        seed_spec["realref_knn_desired_frac_mean"] = float(desired_latent_knn)
        seed_spec.update(seed_pres)
        seed_spec.update(seed_rank)
        seed_spec.update({
            "src_pairwise_dist_mean": float(src_pw["pairwise_dist_mean"]),
            "edit_pairwise_dist_mean": float(edt_pw["pairwise_dist_mean"]),
            "pairwise_dist_ratio_edit_over_seed": float(collapse_ratio),
        })
        print("[seed-spec]", seed_spec)

        # latent UMAP
        try:
            #plot_umap_latent(Z_h_all, Z_t_all, Z_edit, os.path.join(out_base, "umap_latent_real_vs_edited.png"))
            plot_umap_latent(
                Z_ref_h_lat_eval, 
                Z_ref_t_lat_eval, 
                Z_edit, 
                os.path.join(out_base, "umap_latent_real_vs_edited.png")
            )
            plot_umap_latent(Z_ref_h_lat_eval, Z_ref_t_lat_eval, Z_edit,
                 os.path.join(out_base, "umap_latent_refs_vs_edited.png"))

            plot_umap_latent(Z_seed_h, Z_seed_t, Z_edit,
                 os.path.join(out_base, "umap_latent_seeds_vs_edited.png"))


        except Exception as e:
            print(f"[warn] latent UMAP plotting failed: {e}")

        # classifier prob at t=0
        with torch.no_grad():
            Z_edit_t = torch.from_numpy(Z_edit).float().to(device)
            zeros_t = torch.zeros(Z_edit_t.size(0), dtype=torch.long, device=device)
            p = torch.softmax(clf(Z_edit_t, zeros_t), dim=1).cpu().numpy()
        p_target = p[:, target_id]

        # Save latent outputs
        np.savez(os.path.join(out_base, "edited_latents.npz"), Z_edit=Z_edit, seed_ids=seed_ids)
        pd.DataFrame({
            "cell_idx": seed_ids,
            "mode": mode,
            "direction": direction,
            "t_param": int(t_param),
            "guidance_s": float(s),
            "rep": rep_ids,
            "p_target": p_target,
            "cosine_to_seed": cosine_sim,
            "l2_to_seed": l2_dist
        }).to_csv(os.path.join(out_base, "metrics.csv"), index=False)

        # latent centroid score (latent-only)
        if direction == "healthy2tumor":
            latent_centroid = _latent_centroid_score(Z_ref_h_lat_eval, Z_ref_t_lat_eval, Z_edit)
        else:
            latent_centroid = _latent_centroid_score(Z_ref_t_lat_eval, Z_ref_h_lat_eval, Z_edit)



        # ============================================================
        # ------------------- DECODE SUBSET (opt) --------------------
        # ============================================================
        n_decode = int(round(Z_edit.shape[0] * max(0.0, min(1.0, args.decode_frac))))
        decoded_available = n_decode > 0

        boot_ci = {}
        selected_method_summary = {}
        zig_eval = None  # set only if decoded_available

        if decoded_available:
            pick = np.random.default_rng(0).choice(Z_edit.shape[0], size=n_decode, replace=False)
            Z_sub = torch.from_numpy(Z_edit[pick]).float().to(device)
            seed_ids_sub = seed_ids[pick]  # indices into A_seed / Z_seed_all
            rep_ids_sub = rep_ids[pick]
            # decode
            if args.project_before_decode:
                Z_proj = Z_sub.clone()
                for _ in range(max(1, args.project_n_steps)):
                    with torch.no_grad():
                        X_tmp = vae.decoder(Z_proj).cpu().numpy()
                    Z_np = encode_vae(vae, X_tmp.astype(np.float32), device)
                    Z_proj = torch.from_numpy(Z_np).float().to(device)
                with torch.no_grad():
                    X_dec = vae.decoder(Z_proj).cpu().numpy()
            else:
                with torch.no_grad():
                    X_dec = vae.decoder(Z_sub).cpu().numpy()

            fake = ad.AnnData(X=X_dec, var=pd.DataFrame(index=genes))

            # sparsity projection FIRST (target distribution depends on direction; use TRAIN subset for target sparsity)
            if args.sparsity_project:
                target_for_sparsity = real_t_eval if direction == "healthy2tumor" else real_h_eval
                fake = project_sparsity_gene_wise(
                    fake, target_for_sparsity,
                    min_detect_rate=args.sparsity_min_detect_rate,
                    max_detect_rate=args.sparsity_max_detect_rate
                )
            X_fake = to_numpy_dense(fake.X).astype(np.float32)

            # seed recon in SAME eval space (decode the corresponding original seeds)
            Z_seed_sub = Z_seed_all[seed_ids_sub]
            X_seed_dec = decode_latents_in_batches(vae, Z_seed_sub, device, batch_size=args.batch).astype(np.float32)

            # eval refs decoded (ALWAYS from eval-latents chosen earlier)
            X_ref_h_eval = decode_latents_in_batches(vae, Z_ref_h_lat_eval, device, batch_size=args.batch).astype(np.float32)
            X_ref_t_eval = decode_latents_in_batches(vae, Z_ref_t_lat_eval, device, batch_size=args.batch).astype(np.float32)

            # encode candidates back to latent (sym eval)
            Z_fake_lat = encode_vae(vae, X_fake.astype(np.float32), device).astype(np.float32)

            # build per-candidate mapped seed IDs in fixed order of seed_idx
            seed_idx_list = seed_idx.tolist()
            pos = {int(gid): i for i, gid in enumerate(seed_idx_list)}  # global id -> 0..n_seed-1
            seed_ids_mapped = np.array([pos[int(g)] for g in seed_ids_sub], dtype=int)

            # unique seed arrays (same order as seed_idx)
            Z_seed_lat_unique = Z_seed.copy().astype(np.float32)  # already ordered
            X_seed_dec_unique = decode_latents_in_batches(vae, Z_seed_lat_unique, device, batch_size=args.batch).astype(np.float32)

            # per-candidate aligned seed decode (already aligned)
            X_seed_dec_aligned = X_seed_dec.astype(np.float32)

            fake_for_sig = ad.AnnData(X=X_fake.astype(np.float32), var=pd.DataFrame(index=genes))

            # ========== Evaluate ZIGZAG (sym) ==========
            eval_rows = []
            zig_eval, zig_intra_df, zig_iddist = evaluate_method_sym(
                method_name="zigzag",
                direction=direction,
                X_cand_dec=X_fake.astype(np.float32),
                Z_cand_lat=Z_fake_lat.astype(np.float32),
                seed_ids_global=seed_ids_sub.astype(int),
                seed_ids_mapped=seed_ids_mapped,
                X_seed_dec=X_seed_dec_aligned,
                X_seed_dec_unique=X_seed_dec_unique,
                Z_seed_lat_unique=Z_seed_lat_unique,
                Z_ref_h_lat_eval=Z_ref_h_lat_eval,
                Z_ref_t_lat_eval=Z_ref_t_lat_eval,
                X_ref_h_eval=X_ref_h_eval,
                X_ref_t_eval=X_ref_t_eval,
                real_h_eval=real_h_eval,
                real_t_eval=real_t_eval,
                fake_eval=fake_for_sig,
                src_label_qc=src_label_qc,
                tgt_label_qc=tgt_label_qc,
                Z_src_seed=Z_src_seed,
                Z_tgt_seed=Z_tgt_seed,
            )
            eval_rows.append(zig_eval)

            # ========== Baselines (same eval function) ==========
            n_pick = 3

            # random tumor baseline (one per candidate)
            X_rand, rand_idx = baseline_random_tumor(X_pool_t_dec, n_total=len(seed_ids_sub), rng_seed=0)
            Z_rand = encode_vae(vae, X_rand.astype(np.float32), device).astype(np.float32)
            base, _, _ = evaluate_method_sym(
                method_name="baseline_random_tumor",
                direction=direction,
                X_cand_dec=X_rand.astype(np.float32),
                Z_cand_lat=Z_rand.astype(np.float32),
                seed_ids_global=seed_ids_sub.astype(int),
                seed_ids_mapped=seed_ids_mapped,
                X_seed_dec=X_seed_dec_aligned,
                X_seed_dec_unique=X_seed_dec_unique,
                Z_seed_lat_unique=Z_seed_lat_unique,
                Z_ref_h_lat_eval=Z_ref_h_lat_eval,
                Z_ref_t_lat_eval=Z_ref_t_lat_eval,
                X_ref_h_eval=X_ref_h_eval,
                X_ref_t_eval=X_ref_t_eval,
            )
            eval_rows.append(base)

            # nearest tumor 1 baseline
            X_nn1, nn1_idx = baseline_nearest_tumor_1(X_seed_dec_aligned, X_pool_t_dec, n_pcs=50, rng_seed=0)
            Z_nn1 = encode_vae(vae, X_nn1.astype(np.float32), device).astype(np.float32)
            base, _, _ = evaluate_method_sym(
                method_name="baseline_nearest_tumor_1",
                direction=direction,
                X_cand_dec=X_nn1.astype(np.float32),
                Z_cand_lat=Z_nn1.astype(np.float32),
                seed_ids_global=seed_ids_sub.astype(int),
                seed_ids_mapped=seed_ids_mapped,
                X_seed_dec=X_seed_dec_aligned,
                X_seed_dec_unique=X_seed_dec_unique,
                Z_seed_lat_unique=Z_seed_lat_unique,
                Z_ref_h_lat_eval=Z_ref_h_lat_eval,
                Z_ref_t_lat_eval=Z_ref_t_lat_eval,
                X_ref_h_eval=X_ref_h_eval,
                X_ref_t_eval=X_ref_t_eval,
            )
            eval_rows.append(base)
            A_pool_train = adata_train


            seed_adata_sub = A_seed[seed_ids_sub].copy()          # TEST seeds

            tumor_pool_adata = A_seed[t_pool_idx].copy()          # TEST tumor pool
            X_pool_t_dec_for_match = X_pool_t_dec                 # aligned by construction

            # ===== matched tumor baseline (TEST pool, aligned by construction) =====
            X_match, match_ref_idx, seed_ids_match = baseline_matched_tumor(
                adata_obj=A_seed,                      # in 2-file mode: TEST adata
                seed_ids_sub=seed_ids_sub.astype(int), # indices into A_seed
                idx_tumor_pool=t_pool_idx,             # tumor pool indices into SAME A_seed
                X_ref_t_dec=X_pool_t_dec,              # decoded tumor pool aligned with A_seed[t_pool_idx]
                n_pick=n_pick,
                match_cols=[c for c in ["cell_type","celltype","patient","donor","batch","sample"]
                            if c in A_seed.obs.columns],
                n_bins=10,
                rng_seed=0,
            )


            assert X_pool_t_dec_for_match.shape[0] == tumor_pool_adata.n_obs
            assert X_match.shape[0] == seed_ids_match.shape[0]



            seed_ids_match = seed_ids_match.astype(int)
            seed_ids_match_mapped = np.array([pos[int(g)] for g in seed_ids_match], dtype=int)
            X_seed_match_aligned = decode_latents_in_batches(vae, Z_seed_all[seed_ids_match], device, batch_size=args.batch).astype(np.float32)

            Z_match = encode_vae(vae, X_match.astype(np.float32), device).astype(np.float32)
            base, _, _ = evaluate_method_sym(
                method_name="baseline_matched_tumor",
                direction=direction,
                X_cand_dec=X_match.astype(np.float32),
                Z_cand_lat=Z_match.astype(np.float32),
                seed_ids_global=seed_ids_match,
                seed_ids_mapped=seed_ids_match_mapped,
                X_seed_dec=X_seed_match_aligned,
                X_seed_dec_unique=X_seed_dec_unique,
                Z_seed_lat_unique=Z_seed_lat_unique,
                Z_ref_h_lat_eval=Z_ref_h_lat_eval,
                Z_ref_t_lat_eval=Z_ref_t_lat_eval,
                X_ref_h_eval=X_ref_h_eval,
                X_ref_t_eval=X_ref_t_eval,
            )
            eval_rows.append(base)

            # linear shift baseline (alpha grid)
            for alpha in [0.25, 0.5, 0.75, 1.0]:
                X_lin = baseline_linear_shift(X_seed_dec_aligned, mu_h, mu_t, alpha=float(alpha))
                Z_lin = encode_vae(vae, X_lin.astype(np.float32), device).astype(np.float32)
                base, _, _ = evaluate_method_sym(
                    method_name=f"baseline_linear_shift_a{alpha}",
                    direction=direction,
                    X_cand_dec=X_lin.astype(np.float32),
                    Z_cand_lat=Z_lin.astype(np.float32),
                    seed_ids_global=seed_ids_sub.astype(int),
                    seed_ids_mapped=seed_ids_mapped,
                    X_seed_dec=X_seed_dec_aligned,
                    X_seed_dec_unique=X_seed_dec_unique,
                    Z_seed_lat_unique=Z_seed_lat_unique,
                    Z_ref_h_lat_eval=Z_ref_h_lat_eval,
                    Z_ref_t_lat_eval=Z_ref_t_lat_eval,
                    X_ref_h_eval=X_ref_h_eval,
                    X_ref_t_eval=X_ref_t_eval,
                )
                eval_rows.append(base)

            # gaussian editor baseline (K per seed)
            for alpha in [0.5, 1.0]:
                for eps_scale in [0.5, 1.0]:
                    X_g_list, seed_g_list = [], []
                    for r in range(n_pick):
                        X_g = baseline_latent_gaussian_editor(
                            X_seed_dec=X_seed_dec_aligned,
                            mu_h=mu_h, mu_t=mu_t,
                            X_ref_t_dec=X_pool_t_dec,
                            alpha=float(alpha),
                            cov_mode="diag",
                            eps_scale=float(eps_scale),
                            rng_seed=1000 + r,
                        )
                        X_g_list.append(X_g)
                        seed_g_list.append(seed_ids_sub.astype(int))
                    X_gauss = np.vstack(X_g_list).astype(np.float32)
                    seed_ids_gauss = np.concatenate(seed_g_list).astype(int)
                    seed_ids_gauss_mapped = np.array([pos[int(g)] for g in seed_ids_gauss], dtype=int)

                    X_seed_gauss_aligned = np.repeat(X_seed_dec_aligned, n_pick, axis=0)
                    Z_gauss = encode_vae(vae, X_gauss.astype(np.float32), device).astype(np.float32)

                    base, _, _ = evaluate_method_sym(
                        method_name=f"baseline_gaussian_a{alpha}_e{eps_scale}",
                        direction=direction,
                        X_cand_dec=X_gauss.astype(np.float32),
                        Z_cand_lat=Z_gauss.astype(np.float32),
                        seed_ids_global=seed_ids_gauss,
                        seed_ids_mapped=seed_ids_gauss_mapped,
                        X_seed_dec=X_seed_gauss_aligned,
                        X_seed_dec_unique=X_seed_dec_unique,
                        Z_seed_lat_unique=Z_seed_lat_unique,
                        Z_ref_h_lat_eval=Z_ref_h_lat_eval,
                        Z_ref_t_lat_eval=Z_ref_t_lat_eval,
                        X_ref_h_eval=X_ref_h_eval,
                        X_ref_t_eval=X_ref_t_eval,
                    )
                    eval_rows.append(base)

            # write symmetric comparison table
            sym_df = pd.DataFrame(eval_rows)
            sym_df["mode"] = mode
            sym_df["t_param"] = int(t_param)
            sym_df["guidance_s"] = float(s)
            sym_df["multi_pass_rounds"] = int(rounds) if mode == "multi" else 0
            sym_df.to_csv(os.path.join(out_base, "symmetric_method_comparison.csv"), index=False)
            print("[sym-eval] wrote:", os.path.join(out_base, "symmetric_method_comparison.csv"))

            # ensure seed decoded subset is in same sparsity regime as X_fake for progress/identity
            if args.sparsity_project:
                seed_ad = ad.AnnData(X=X_seed_dec_aligned, var=pd.DataFrame(index=genes))
                seed_ad = project_sparsity_gene_wise(
                    seed_ad, A_seed[seed_ids_sub],
                    min_detect_rate=args.sparsity_min_detect_rate,
                    max_detect_rate=args.sparsity_max_detect_rate
                )
                X_seed_dec_aligned = to_numpy_dense(seed_ad.X).astype(np.float32)

            # ============================================================
            # Progress + identity (decoded)

            # ============================================================
            #prog_raw, prog01 = progress_score_01_decoder(
            #    X_fake, prog_scaler, prog_pca, prog_w, prog_b, ref_raw_h, ref_raw_t
            #)
            #bins = stage_bins(prog01)
            #iddist_expr = np.linalg.norm(X_fake - X_seed_dec_aligned, axis=1).astype(np.float32)
            #pareto_mask = pareto_front(conversion=prog01, identity_dist=iddist_expr)
            prog_raw, prog01 = progress_score_01_decoder(
                X_fake, prog_scaler, prog_pca, prog_w, prog_b, ref_raw_h, ref_raw_t
            )
            bins = stage_bins(prog01)
            iddist_expr = np.linalg.norm(X_fake - X_seed_dec_aligned, axis=1).astype(np.float32)
            # Linear healthy-vs-tumor score in expression PCA space
            tumor_scaler, tumor_pca, tumor_clf = fit_tumor_logreg_axis(
                X_ref_healthy=X_ref_h_eval,
                X_ref_tumor=X_ref_t_eval,
                n_pcs=50,
                random_state=int(args.seed),
            )
            tumor_logit, tumor_proba = tumor_logreg_score(
                X_query=X_fake,
                scaler=tumor_scaler,
                pca=tumor_pca,
                clf=tumor_clf,
            )
            expr_ratio = expr_pca_reference_distance_ratio_score(
                X_query=X_fake,
                X_ref_healthy=X_ref_h_eval,
                X_ref_tumor=X_ref_t_eval,
                k=50,
                n_pcs=50
            )
            dist_h_expr = expr_ratio["dist_h_per_cell"].astype(np.float32)
            dist_t_expr = expr_ratio["dist_t_per_cell"].astype(np.float32)
            tumor_margin = (dist_h_expr - dist_t_expr).astype(np.float32)

            pareto_mask = pareto_front(conversion=tumor_logit, identity_dist=iddist_expr)
            decoded_metrics = pd.DataFrame({
                "cell_idx": seed_ids_sub.astype(int),
                "rep_id": rep_ids_sub.astype(int),
                "direction": direction,
                "mode": mode,
                "t_param": int(t_param),
                "guidance_s": float(s),
                "eta": float(getattr(args, "eta", 1.0)),
                "progress_raw": prog_raw,
                "progress01": prog01,
                "stage_bin": bins,
                "id_dist_expr": iddist_expr,
                "dist_to_healthy_expr": dist_h_expr,
                "dist_to_tumor_expr": dist_t_expr,
                "tumor_margin": tumor_margin,
                "pareto": pareto_mask.astype(int),
                "tumor_logit": tumor_logit,
                "tumor_proba": tumor_proba,
            })
        

            # Identity threshold: erstmal median-basiert, später ggf. auf Validation fixieren
            tau_id = float(np.nanmedian(decoded_metrics["id_dist_expr"].values))

            decoded_metrics["soft_success"] = (
                (decoded_metrics["tumor_logit"] > 0.0) &
                (decoded_metrics["id_dist_expr"] <= tau_id)
            ).astype(int)

            decoded_metrics["strict_success"] = (
                (decoded_metrics["tumor_logit"] > 0.0) &
                (decoded_metrics["tumor_proba"] >= 0.70) &
                (decoded_metrics["id_dist_expr"] <= tau_id)
            ).astype(int)
            decoded_metrics.to_csv(os.path.join(out_base, "decoded_progress_identity.csv"), index=False)

            # ------------------------------------------------------------
            # Seed-wise selected candidates: actual stochastic search output
            # ------------------------------------------------------------
            # Important: decoded_metrics describes the raw candidate distribution.
            # For ZigZag-as-candidate-discovery, method performance should also be
            # evaluated after a predefined per-seed selection rule.
            selected_rows = []

            def _append_selected(row_like, rule_name: str, fallback_used: bool = False):
                row = row_like.copy()
                row["selection_rule"] = str(rule_name)
                row["fallback_used"] = int(bool(fallback_used))
                row["decoded_index"] = int(row.name)
                selected_rows.append(row)

            for sid, g in decoded_metrics.groupby("cell_idx", sort=False):
                # 1) MaxTumor: how tumor-like can this seed become under the budget?
                idx_max = g["tumor_logit"].idxmax()
                _append_selected(
                    g.loc[idx_max],
                    rule_name="max_tumor_logit",
                    fallback_used=False,
                )

                # 2) TumorThenIdentity: first require clear tumor evidence, then preserve seed identity.
                gt = g[g["tumor_proba"] >= 0.70]
                if len(gt) > 0:
                    idx_sel = gt["id_dist_expr"].idxmin()
                    _append_selected(
                        g.loc[idx_sel],
                        rule_name="proba_ge_0.7_min_identity",
                        fallback_used=False,
                    )
                else:
                    _append_selected(
                        g.loc[idx_max],
                        rule_name="proba_ge_0.7_min_identity",
                        fallback_used=True,
                    )

                # 3) Pareto-knee: balanced compromise on the per-seed Pareto front.
                # Maximize tumor_logit and minimize id_dist_expr with equal weight after min-max scaling.
                gp = g[g["pareto"].astype(int) == 1].copy()
                if len(gp) == 0:
                    gp = g.copy()
                    pareto_fallback = True
                else:
                    pareto_fallback = False

                tl = gp["tumor_logit"].astype(float).values
                idd = gp["id_dist_expr"].astype(float).values
                tl_norm = (tl - np.nanmin(tl)) / (np.nanmax(tl) - np.nanmin(tl) + 1e-12)
                id_norm = (np.nanmax(idd) - idd) / (np.nanmax(idd) - np.nanmin(idd) + 1e-12)
                gp["pareto_knee_score"] = 0.5 * tl_norm + 0.5 * id_norm
                idx_pareto = gp["pareto_knee_score"].idxmax()
                row_p = g.loc[idx_pareto].copy()
                row_p["pareto_knee_score"] = float(gp.loc[idx_pareto, "pareto_knee_score"])
                _append_selected(
                    row_p,
                    rule_name="pareto_knee_tumor_identity",
                    fallback_used=pareto_fallback,
                )

            selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
            selected_df.to_csv(os.path.join(out_base, "selected_candidates_eval.csv"), index=False)

            selected_summary = (
                selected_df.groupby("selection_rule")
                .agg(
                    n_selected=("cell_idx", "size"),
                    n_seeds=("cell_idx", "nunique"),
                    fallback_frac=("fallback_used", "mean"),
                    mean_tumor_logit=("tumor_logit", "mean"),
                    median_tumor_logit=("tumor_logit", "median"),
                    mean_tumor_proba=("tumor_proba", "mean"),
                    median_tumor_proba=("tumor_proba", "median"),
                    mean_id_dist_expr=("id_dist_expr", "mean"),
                    median_id_dist_expr=("id_dist_expr", "median"),
                    mean_progress01=("progress01", "mean"),
                    median_progress01=("progress01", "median"),
                    frac_tumor_logit_gt0=("tumor_logit", lambda x: float(np.mean(np.asarray(x) > 0.0))),
                    frac_tumor_proba_ge_0_5=("tumor_proba", lambda x: float(np.mean(np.asarray(x) >= 0.50))),
                    frac_tumor_proba_ge_0_7=("tumor_proba", lambda x: float(np.mean(np.asarray(x) >= 0.70))),
                    frac_tumor_proba_ge_0_9=("tumor_proba", lambda x: float(np.mean(np.asarray(x) >= 0.90))),
                )
                .reset_index()
            )
            selected_summary.to_csv(os.path.join(out_base, "selected_candidates_summary.csv"), index=False)

            # Compact JSON version for run_eval.json and aggregation.
            selected_method_summary = {}
            for _, rr in selected_summary.iterrows():
                prefix = "selected_" + str(rr["selection_rule"])
                for col in selected_summary.columns:
                    if col == "selection_rule":
                        continue
                    val = rr[col]
                    if pd.isna(val):
                        selected_method_summary[f"{prefix}_{col}"] = float("nan")
                    elif isinstance(val, (np.integer, int)):
                        selected_method_summary[f"{prefix}_{col}"] = int(val)
                    else:
                        selected_method_summary[f"{prefix}_{col}"] = float(val)
            save_json(selected_method_summary, os.path.join(out_base, "selected_candidates_summary.json"))

            # Save selected expression profiles for downstream DE / marker / pathway analyses.
            selected_fake_for_analysis = None
            try:
                sel_idx = selected_df["decoded_index"].astype(int).values
                selected_fake = fake[sel_idx].copy()
                selected_fake.obs = selected_df.astype({
                    "cell_idx": str,
                    "rep_id": str,
                    "selection_rule": str,
                }).copy()
                selected_fake_for_analysis = selected_fake
                selected_fake.write_h5ad(os.path.join(out_base, "selected_candidates_expr.h5ad"))

                # UMAP overview for selected outputs only. If multiple rules are present,
                # this plot intentionally shows all selected outputs together.
                plot_umap_expr_all_samples(
                    real_h_eval=real_h_eval,
                    real_t_eval=real_t_eval,
                    fake=selected_fake,
                    seed_ids=selected_df["cell_idx"].astype(int).values,
                    out_png=os.path.join(out_base, "umap_selected_candidates_expr.png"),
                )
            except Exception as e:
                print(f"[warn] selected candidate h5ad/umap failed: {e}")

            # ------------------------------------------------------------
            # Selected-candidate biology: DE / marker / pathway
            # This is intentionally outside the UMAP try/except, so biology
            # still runs even if plotting fails.
            # ------------------------------------------------------------
            if selected_fake_for_analysis is not None:
                selected_bio_summary = _write_selected_candidate_biology(
                    out_base=out_base,
                    selected_fake_all=selected_fake_for_analysis,
                    real_h_eval=real_h_eval,
                    real_t_eval=real_t_eval,
                    direction=direction,
                    args=args,
                )

                if selected_bio_summary:
                    selected_method_summary.update({
                        f"selected_bio_{k}": v
                        for k, v in selected_bio_summary.items()
                        if isinstance(v, (int, float, np.integer, np.floating))
                    })

            # ------------------------------------------------------------
            # Selected successful trajectories
            # Requires --save_all_rep_trajectories and the matching npz.
            # ------------------------------------------------------------
            selected_traj_summary = _write_selected_successful_trajectories(
                out_base=out_base,
                vae=vae,
                device=device,
                genes=genes,
                selected_df=selected_df,
                decoded_metrics=decoded_metrics,
                X_seed_dec_aligned=X_seed_dec_aligned,
                prog_scaler=prog_scaler,
                prog_pca=prog_pca,
                prog_w=prog_w,
                prog_b=prog_b,
                ref_raw_h=ref_raw_h,
                ref_raw_t=ref_raw_t,
                tumor_scaler=tumor_scaler,
                tumor_pca=tumor_pca,
                tumor_clf=tumor_clf,
                args=args,
                direction=direction,          # <-- NEU
                real_h_eval=real_h_eval,      # <-- NEU
                real_t_eval=real_t_eval,      # <-- NEU
            )

            if selected_traj_summary:
                selected_method_summary.update({
                    f"selected_traj_{k}": v
                    for k, v in selected_traj_summary.items()
                    if isinstance(v, (int, float, np.integer, np.floating))
                })

            # Re-write compact JSON after adding optional selected biology/trajectory metrics.
            save_json(selected_method_summary, os.path.join(out_base, "selected_candidates_summary.json"))

            print("[selected] wrote:", os.path.join(out_base, "selected_candidates_eval.csv"))
            print("[selected] wrote:", os.path.join(out_base, "selected_candidates_summary.csv"))

            # ------------------------------------------------------------
            # Per-seed best-of-k / median-of-k summary over repetitions
            # ------------------------------------------------------------
            seed_group = decoded_metrics.groupby("cell_idx", sort=False)

            per_seed_best = (
                seed_group.apply(lambda g: g.loc[g["tumor_logit"].idxmax()])
                .reset_index(drop=True)
                .copy()
            )
            per_seed_best["tumor_like_best"] = (per_seed_best["progress01"] >= 0.5).astype(int)
            per_seed_median = (
                seed_group[["progress01", "tumor_logit", "tumor_proba", "id_dist_expr", "soft_success", "strict_success"]]
                .median()
                .reset_index()
                .rename(columns={
                    "progress01": "progress01_median",
                    "tumor_logit": "tumor_logit_median",
                    "tumor_proba": "tumor_proba_median",
                    "id_dist_expr": "id_dist_expr_median",
                    "soft_success": "soft_success_frac",
                    "strict_success": "strict_success_frac",
                })
            )

            per_seed_any_tumor_like = (
                seed_group["progress01"]
                .apply(lambda x: bool(np.any(x >= 0.5)))
                .reset_index(name="any_tumor_like")
            )
            per_seed_any_soft_success = (
                seed_group["soft_success"]
                .max()
                .reset_index(name="any_soft_success")
            )

            per_seed_any_strict_success = (
                seed_group["strict_success"]
                .max()
                .reset_index(name="any_strict_success")
            )

            per_seed_frac_success = (
                decoded_metrics.groupby("cell_idx", sort=False)["soft_success"]
                .mean()
                .reset_index(name="frac_success")
            )

            per_seed_any_success = (
                decoded_metrics.groupby("cell_idx", sort=False)["soft_success"]
                .max()
                .reset_index(name="any_success")
            )

            per_seed_summary = per_seed_best.merge(per_seed_median, on="cell_idx", how="left")
            per_seed_summary = per_seed_summary.merge(per_seed_any_tumor_like, on="cell_idx", how="left")
            per_seed_summary = per_seed_summary.merge(per_seed_frac_success, on="cell_idx", how="left")
            per_seed_summary = per_seed_summary.merge(per_seed_any_success, on="cell_idx", how="left")
            per_seed_summary = per_seed_summary.merge(per_seed_any_soft_success, on="cell_idx", how="left")
            per_seed_summary = per_seed_summary.merge(per_seed_any_strict_success, on="cell_idx", how="left")
            per_seed_summary.to_csv(
                os.path.join(out_base, "per_seed_best_of_k.csv"),
                index=False
            )

            best_of_k_summary = {
                "n_unique_seeds": int(per_seed_summary["cell_idx"].nunique()),
                "mean_best_progress01": float(per_seed_summary["progress01"].mean()),
                "median_best_progress01": float(per_seed_summary["progress01"].median()),
                "mean_best_id_dist_expr": float(per_seed_summary["id_dist_expr"].mean()),
                "median_best_id_dist_expr": float(per_seed_summary["id_dist_expr"].median()),
                "mean_best_tumor_margin": float(per_seed_summary["tumor_margin"].mean()),
                "median_best_tumor_margin": float(per_seed_summary["tumor_margin"].median()),
                "frac_seeds_any_tumor_like_progress01_ge_0_5": float(per_seed_summary["any_tumor_like"].mean()),
                "frac_seeds_best_tumor_like_progress01_ge_0_5": float(per_seed_summary["tumor_like_best"].mean()),
                "frac_seeds_any_soft_success": float(per_seed_summary["any_success"].mean()),
                "mean_frac_success_per_seed": float(per_seed_summary["frac_success"].mean()),
                "median_frac_success_per_seed": float(per_seed_summary["frac_success"].median()),
                "mean_best_tumor_logit": float(per_seed_summary["tumor_logit"].mean()),
                "median_best_tumor_logit": float(per_seed_summary["tumor_logit"].median()),
                "mean_best_tumor_proba": float(per_seed_summary["tumor_proba"].mean()),
                "median_best_tumor_proba": float(per_seed_summary["tumor_proba"].median()),
                "tau_id": float(tau_id),
                "frac_seeds_any_strict_success": float(per_seed_summary["any_strict_success"].mean()),
                "mean_soft_success_frac_per_seed": float(per_seed_summary["soft_success_frac"].mean()),
                "mean_strict_success_frac_per_seed": float(per_seed_summary["strict_success_frac"].mean()),
            }

            save_json(
                best_of_k_summary,
                os.path.join(out_base, "best_of_k_summary.json")
            )
            # ------------------------------------------------------------
            # Direction consistency per seed
            # ------------------------------------------------------------
            direction_rows = []

            for sid, g in decoded_metrics.groupby("cell_idx", sort=False):
                idxs = g.index.to_numpy()
                if len(idxs) < 2:
                    continue

                X_seed_one = X_seed_dec_aligned[idxs[0]:idxs[0]+1]
                V = X_fake[idxs] - X_seed_one

                norms = np.linalg.norm(V, axis=1) + 1e-9
                Vn = V / norms[:, None]

                cos_mat = Vn @ Vn.T
                upper = cos_mat[np.triu_indices_from(cos_mat, k=1)]

                direction_rows.append({
                    "cell_idx": int(sid),
                    "n_samples": int(len(idxs)),
                    "mean_pairwise_cosine": float(np.mean(upper)),
                    "median_pairwise_cosine": float(np.median(upper)),
                    "std_pairwise_cosine": float(np.std(upper)),
                    "mean_tumor_logit": float(g["tumor_logit"].mean()),
                    "best_tumor_logit": float(g["tumor_logit"].max()),
                    "mean_progress01": float(g["progress01"].mean()),
                    "best_progress01": float(g["progress01"].max()),
                })

            direction_df = pd.DataFrame(direction_rows)
            direction_df.to_csv(
                os.path.join(out_base, "direction_consistency.csv"),
                index=False,
            )

            save_json(
                {
                    "mean_pairwise_cosine_across_seeds": float(direction_df["mean_pairwise_cosine"].mean()) if len(direction_df) else np.nan,
                    "median_pairwise_cosine_across_seeds": float(direction_df["median_pairwise_cosine"].median()) if len(direction_df) else np.nan,
                },
                os.path.join(out_base, "direction_consistency_summary.json"),
            )
            # ------------------------------------------------------------
            # Success vs budget using tumor_logit + identity
            # ------------------------------------------------------------
            budget_rows = []

            budgets = [1, 2, 5, 10, 20, 50, 100]
            budgets = [b for b in budgets if b <= decoded_metrics["rep_id"].nunique()]

            rng_budget = np.random.default_rng(int(args.seed))

            for B in budgets:
                seed_success = []

                for sid, g in decoded_metrics.groupby("cell_idx", sort=False):
                    g = g.copy()

                    # deterministic: use lowest rep_ids up to budget
                    gB = g.sort_values("rep_id").head(B)

                    seed_success.append({
                        "cell_idx": int(sid),
                        "budget": int(B),
                        "any_soft_success": int(gB["soft_success"].max()),
                        "any_strict_success": int(gB["strict_success"].max()),
                        "best_tumor_logit": float(gB["tumor_logit"].max()),
                        "best_tumor_proba": float(gB["tumor_proba"].max()),
                        "best_progress01": float(gB["progress01"].max()),
                        "min_id_dist_expr": float(gB["id_dist_expr"].min()),
                    })

                tmp = pd.DataFrame(seed_success)

                budget_rows.append({
                    "budget": int(B),
                    "frac_seeds_any_soft_success": float(tmp["any_soft_success"].mean()),
                    "frac_seeds_any_strict_success": float(tmp["any_strict_success"].mean()),
                    "mean_best_tumor_logit": float(tmp["best_tumor_logit"].mean()),
                    "median_best_tumor_logit": float(tmp["best_tumor_logit"].median()),
                    "mean_best_tumor_proba": float(tmp["best_tumor_proba"].mean()),
                    "median_best_tumor_proba": float(tmp["best_tumor_proba"].median()),
                    "mean_best_progress01": float(tmp["best_progress01"].mean()),
                    "median_best_progress01": float(tmp["best_progress01"].median()),
                })

            pd.DataFrame(budget_rows).to_csv(
                os.path.join(out_base, "success_vs_budget_tumorlogit.csv"),
                index=False,
            )
            # ------------------------------------------------------------
            # Visualizations for repeated samples per seed
            # ------------------------------------------------------------
            try:
                plot_umap_expr_seed_gallery(
                    real_h_eval=real_h_eval,
                    real_t_eval=real_t_eval,
                    real_seed_adata=A_seed[seed_ids_sub].copy(),
                    fake=fake,
                    seed_ids=seed_ids_sub,
                    rep_ids=rep_ids_sub,
                    out_png=os.path.join(out_base, "umap_seed_gallery.png"),
                    n_show=12,
                    select_by="best_progress",
                    seed_scores=per_seed_summary,
                )
            except Exception as e:
                print(f"[warn] seed gallery plot failed: {e}")

            try:
                plot_umap_expr_all_samples(
                    real_h_eval=real_h_eval,
                    real_t_eval=real_t_eval,
                    fake=fake,
                    seed_ids=seed_ids_sub,
                    out_png=os.path.join(out_base, "umap_all_samples.png"),
                )
            except Exception as e:
                print(f"[warn] global all-samples UMAP failed: {e}")

            try:
                plot_best_vs_identity(
                    per_seed_summary=per_seed_summary,
                    out_png=os.path.join(out_base, "best_vs_identity.png"),
                    x_col="progress01",
                    y_col="id_dist_expr",
                )
                plot_best_vs_identity(
                    per_seed_summary=per_seed_summary,
                    out_png=os.path.join(out_base, "best_tumorlogit_vs_identity.png"),
                    x_col="tumor_logit",
                    y_col="id_dist_expr",
                )
            except Exception as e:
                print(f"[warn] best-vs-identity plot failed: {e}")
            pareto_summary = {
                "pareto_frac": float(np.mean(pareto_mask)),
                "pareto_progress01_mean": float(np.mean(prog01[pareto_mask])) if np.any(pareto_mask) else np.nan,
                "pareto_id_dist_expr_mean": float(np.mean(iddist_expr[pareto_mask])) if np.any(pareto_mask) else np.nan,
            }
            chains = pick_chain_per_seed(seed_ids_sub, prog01, iddist_expr, n_chain=5)
            rows_chain = []
            for sid, idxs in chains.items():
                for step, ridx in enumerate(idxs):
                    rows_chain.append({
                        "cell_idx": int(sid),
                        "chain_step": int(step),
                        "row_in_subset": int(ridx),
                        "progress01": float(prog01[ridx]),
                        "tumor_logit": float(tumor_logit[ridx]),
                        "tumor_proba": float(tumor_proba[ridx]),
                        "id_dist_expr": float(iddist_expr[ridx]),
                        "soft_success": int(decoded_metrics.iloc[ridx]["soft_success"]),
                        "strict_success": int(decoded_metrics.iloc[ridx]["strict_success"]),
                        "stage_bin": int(bins[ridx]),
                    })
            pd.DataFrame(rows_chain).to_csv(os.path.join(out_base, "per_seed_chain.csv"), index=False)

            boot_ci = {
                "progress01": bootstrap_ci(prog01, n_boot=500, seed=args.seed),
                "tumor_logit": bootstrap_ci(tumor_logit, n_boot=500, seed=args.seed),
                "tumor_proba": bootstrap_ci(tumor_proba, n_boot=500, seed=args.seed),
                "id_dist_expr": bootstrap_ci(iddist_expr, n_boot=500, seed=args.seed),
                "soft_success": bootstrap_ci(decoded_metrics["soft_success"].values, n_boot=500, seed=args.seed),
                "strict_success": bootstrap_ci(decoded_metrics["strict_success"].values, n_boot=500, seed=args.seed),
            }
            save_json(boot_ci, os.path.join(out_base, "bootstrap_ci_decoded.json"))
            # -----------------------------
            # SUCCESS VS BUDGET ANALYSIS
            # -----------------------------

            if getattr(args, "run_continuation", False):
                try:
                    run_continuation_analysis(
                        args=args,
                        out_base=out_base,
                        device=device,
                        diffusion=diffusion,
                        model=model,
                        clf=clf,
                        vae=vae,
                        genes=genes,
                        seed_ids_sub=seed_ids_sub,
                        rep_ids_sub=rep_ids_sub,
                        X_fake=X_fake,
                        decoded_metrics=decoded_metrics,
                        Z_edit_np=Z_edit.astype(np.float32),
                        Z_seed_lat_aligned=Z_seed.astype(np.float32),
                        X_seed_dec_aligned=X_seed_dec_aligned.astype(np.float32),
                        ref_raw_h=ref_raw_h,
                        ref_raw_t=ref_raw_t,
                        prog_scaler=prog_scaler,
                        prog_pca=prog_pca,
                        prog_w=prog_w,
                        prog_b=prog_b,
                        X_ref_h_eval=X_ref_h_eval.astype(np.float32),
                        X_ref_t_eval=X_ref_t_eval.astype(np.float32),
                        direction=direction,
                        mode=mode,
                        t_param=t_param,
                        guidance_s=s,
                        eta=float(getattr(args, "eta", 1.0)),
                        rounds=rounds,
                        target_id=target_id,
                    )
                except Exception as e:
                    print(f"[warn] continuation analysis failed: {e}")
            # expr PCA ref-kNN (evaluation refs in expression space)
            try:
                X_ref_h_sub = to_numpy_dense(real_h_eval.X).astype(np.float32)
                X_ref_t_sub = to_numpy_dense(real_t_eval.X).astype(np.float32)
                knn_expr_ref_mean, _ = expr_pca_reference_knn_target_fraction(
                    X_query=X_fake,
                    X_ref_healthy=X_ref_h_sub,
                    X_ref_tumor=X_ref_t_sub,
                    k=50,
                    n_pcs=50
                )
                knn_frac = float(knn_expr_ref_mean) if direction == "healthy2tumor" else float(1.0 - knn_expr_ref_mean)
            except Exception as e:
                print(f"[warn] expr PCA ref-kNN failed: {e}")
                knn_frac = np.nan



            # save decoded subset + combined h5ad + UMAP
            try:
                fake.write_h5ad(os.path.join(out_base, "decoded_subset.h5ad"), compression="lzf")
            except Exception as e:
                print(f"[warn] writing decoded_subset.h5ad failed: {e}")

            try:
                plot_umap_expr(real_h_eval, real_t_eval, fake, os.path.join(out_base, "umap_expr_real_vs_edited.png"))
            except Exception as e:
                print(f"[warn] expr UMAP plotting failed: {e}")

            try:
#                real_t = real_t_subset.copy()
#                real_h = real_h_subset.copy()
                real_t = real_t_eval.copy()
                real_h = real_h_eval.copy()
                real_t.obs["source"] = "real_tumor"
                real_h.obs["source"] = "real_healthy"
                fake2 = fake.copy()
                fake2.obs["source"] = "generated"
                combined = ad.concat([real_h, real_t, fake2], join="inner", merge="same")
                combined.write_h5ad(os.path.join(out_base, "combined_real_generated.h5ad"), compression="lzf")
                print(f"[combined] wrote {os.path.join(out_base, 'combined_real_generated.h5ad')}")
            except Exception as e:
                print(f"[warn] writing combined_real_generated.h5ad failed: {e}")

        else:
            save_json(
                {"decoded_available": False, "decode_frac": float(args.decode_frac), "n_decode": int(n_decode)},
                os.path.join(out_base, "decoded_skipped.json")
            )
            print("[decode] decode_frac=0 -> skipping decoded QC/eval for this run (latent-only eval will be aggregated).")

        # ============================================================
        # ------------------- aggregate per-run metrics --------------
        # ============================================================
        if decoded_available and zig_eval is not None:
            run_eval = {
                "method": "zigzag",
                "mode": mode,
                "direction": direction,
                "t_param": int(t_param),
                "guidance_s": float(s),
                "eta": float(getattr(args, "eta", 1.0)),
                "multi_pass_rounds": int(rounds) if mode == "multi" else 0,
                "n_reverse_steps": int(rounds * int(t_param)) if mode == "multi" else int(t_param),
                "decoded_available": True,
                "n_decode": int(n_decode),

                # MAIN (sym)
                "knn_desired_lat": float(zig_eval["knn_desired_lat"]),
                "knn_desired_expr": float(zig_eval["knn_desired_expr"]),
                "id_dist_expr_mean": float(zig_eval["id_dist_expr_mean"]),
                "intra_seed_div_mean": float(zig_eval["intra_seed_div_mean"]),
                "cand_seed_rank_median": float(zig_eval["cand_seed_rank_median"]),
                "cand_seed_top1": float(zig_eval["cand_seed_top1"]),
                "pairwise_dist_ratio_edit_over_seed": float(zig_eval["pairwise_dist_ratio_edit_over_seed"]),
                "latent_centroid_score": float(zig_eval.get("latent_centroid_score", np.nan)),
                "dist_ratio_lat": float(zig_eval["dist_ratio_lat"]),
                "dist_ratio_expr": float(zig_eval["dist_ratio_expr"]),
                "dist_to_healthy_lat": float(zig_eval["dist_to_healthy_lat"]),
                "dist_to_tumor_lat": float(zig_eval["dist_to_tumor_lat"]),
                "dist_to_healthy_expr": float(zig_eval["dist_to_healthy_expr"]),
                "dist_to_tumor_expr": float(zig_eval["dist_to_tumor_expr"]),
                "axis_progress_lat": float(zig_eval["axis_progress_lat"]),
                "axis_progress_expr": float(zig_eval["axis_progress_expr"]),
                "de_logfc_pearson": float(zig_eval["de_logfc_pearson"]),
                "de_logfc_spearman": float(zig_eval["de_logfc_spearman"]),
                "de_top50_overlap": float(zig_eval["de_top50_overlap"]),
                "de_top100_overlap": float(zig_eval["de_top100_overlap"]),
            }
        else:
            run_eval = {
                "method": "zigzag",
                "mode": mode,
                "direction": direction,
                "t_param": int(t_param),
                "guidance_s": float(s),
                "multi_pass_rounds": int(rounds) if mode == "multi" else 0,
                "n_reverse_steps": int(rounds * int(t_param)) if mode == "multi" else int(t_param),
                "decoded_available": False,
                "n_decode": int(n_decode),

                # latent-only fallback
                "knn_desired_lat": float(latent_knn),
                "knn_desired_expr": float("nan"),
                "id_dist_expr_mean": float("nan"),
                "intra_seed_div_mean": float("nan"),
                "cand_seed_rank_median": float("nan"),
                "cand_seed_top1": float("nan"),
                "pairwise_dist_ratio_edit_over_seed": float(seed_spec.get("pairwise_dist_ratio_edit_over_seed", np.nan)),
                "latent_centroid_score": float(latent_centroid),
                "dist_ratio_lat": float("nan"),
                "dist_ratio_expr": float("nan"),
                "dist_to_healthy_lat": float("nan"),
                "dist_to_tumor_lat": float("nan"),
                "dist_to_healthy_expr": float("nan"),
                "dist_to_tumor_expr": float("nan"),
                "axis_progress_lat": float("nan"),
                "axis_progress_expr": float("nan"),
                "de_logfc_pearson": float("nan"),
                "de_logfc_spearman": float("nan"),
                "de_top50_overlap": float("nan"),
                "de_top100_overlap": float("nan"),
            }

        run_eval.update(seed_spec)
        run_eval.update(pareto_summary)
        if selected_method_summary:
            run_eval.update(selected_method_summary)
        if boot_ci:
            run_eval["bootstrap_ci_decoded"] = boot_ci

        run_evals.append(run_eval)
        save_json(run_eval, os.path.join(out_base, "run_eval.json"))

        print("[eval]",
            f"centroid={latent_centroid:.3f} latentKNN={latent_knn:.3f}",
            f"decoded={decoded_available} n_decode={n_decode}")
        if decoded_available and zig_eval is not None:
            print("[eval-decoded]",
                f"exprKNN={zig_eval['knn_desired_expr']:.3f}",
                f"id={zig_eval['id_dist_expr_mean']:.3f}",
                f"rank_med={zig_eval['cand_seed_rank_median']:.1f}",
                f"collapse={zig_eval['pairwise_dist_ratio_edit_over_seed']:.3f}")
        # ---- Config dump ----
        save_json(vars(args), os.path.join(out_base, "config.json"))

        # ---- trajectories dump + plots ----
        if track_this_run and traj_dump is not None:
            traj_arr = np.stack(traj_dump, axis=0)  # (traj_rounds+1, n_track, latent_dim)

            np.savez(
                os.path.join(out_base, "trajectories_latent.npz"),
                seed_global=traj_seed_global.astype(np.int32),
                Z_traj=traj_arr,
                t_pass=int(t_param),
                n_steps=int(traj_rounds),
                n_states=int(traj_rounds + 1),
                guidance_s=float(s),
            )

        if track_this_run and os.path.exists(os.path.join(out_base, "trajectories_latent.npz")):
            try:
                # latent refs (eval refs already computed)
                plot_latent_trajectories_pca(
                    Z_ref_h=Z_ref_h_lat_eval,
                    Z_ref_t=Z_ref_t_lat_eval,
                    traj_npz=os.path.join(out_base, "trajectories_latent.npz"),
                    out_png=os.path.join(out_base, "trajectories_latent_pca.png"),
                    max_ref=5000,
                    ref_s=6, ref_alpha=0.12,
                    line_w=2.2,
                    start_s=45, end_s=55,
                )

                # decode eval refs RAW (NO sparsity projection here)
                X_ref_h_dec_eval_raw = decode_latents_in_batches(
                    vae, _subsample_rows(Z_ref_h_lat_eval, max_n=8000, seed=0), device, batch_size=args.batch
                ).astype(np.float32)
                X_ref_t_dec_eval_raw = decode_latents_in_batches(
                    vae, _subsample_rows(Z_ref_t_lat_eval, max_n=8000, seed=1), device, batch_size=args.batch
                ).astype(np.float32)

                # 1) raw decoded reference-fit+project
                plot_expression_trajectories_pca_from_latent_traj(
                    vae=vae,
                    device=device,
                    X_ref_h_dec=X_ref_h_dec_eval_raw,
                    X_ref_t_dec=X_ref_t_dec_eval_raw,
                    traj_npz=os.path.join(out_base, "trajectories_latent.npz"),
                    out_png=os.path.join(out_base, "trajectories_expression_pca_rawdecoded.png"),
                    batch_size=args.batch,
                    top_var_genes=2000,
                    max_ref=8000,
                    zscore=True,
                    genes=genes,
                    sparsity_project=False,
                    sparsity_target_adata=None,
                )

                # 2) sparsity-projected plot (if enabled): project BOTH refs and traj to target sparsity (eval tumor)
                if args.sparsity_project:
                    sparsity_target = real_t_eval
                    plot_expression_trajectories_pca_from_latent_traj(
                        vae=vae,
                        device=device,
                        X_ref_h_dec=X_ref_h_dec_eval_raw,
                        X_ref_t_dec=X_ref_t_dec_eval_raw,
                        traj_npz=os.path.join(out_base, "trajectories_latent.npz"),
                        out_png=os.path.join(out_base, "trajectories_expression_pca_sparsityprojected.png"),
                        batch_size=args.batch,
                        top_var_genes=2000,
                        max_ref=8000,
                        zscore=True,
                        genes=genes,
                        sparsity_project=True,
                        sparsity_target_adata=sparsity_target,
                    )

            except Exception as e:
                print(f"[warn] expression trajectory PCA failed: {e}")

    # --- persist eval reference sets so posthoc can be re-run later ---
    try:
        real_h_eval.write_h5ad(os.path.join(args.outdir, "real_h_eval.h5ad"), compression="lzf")
        real_t_eval.write_h5ad(os.path.join(args.outdir, "real_t_eval.h5ad"), compression="lzf")
        print("[posthoc] saved eval refs: real_h_eval.h5ad / real_t_eval.h5ad")
    except Exception as e:
        print(f"[warn] could not save eval refs for posthoc: {e}")

    if getattr(args, 'skip_posthoc', False):
        return

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

    if df_real.empty:
        print("[realism] no runs passed thresholds; keeping realistic_runs.csv empty")
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
                print(f"[de] skip missing decoded subset: {dec_path}")
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
