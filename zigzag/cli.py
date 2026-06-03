#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def build_run_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # ------------------------- Realism gates -------------------------
    ap.add_argument("--de_log_norm", action="store_true", default=True,
                help="Use normalize_total+log1p before DE (recommended).")

    ap.add_argument("--eval_min_p_target", type=float, default=0.75)
    ap.add_argument("--eval_min_knn", type=float, default=0.20)
    ap.add_argument("--eval_min_centroid", type=float, default=0.55)
    ap.add_argument("--eval_max_l2", type=float, default=0.80)
    ap.add_argument("--de_top_n", type=int, default=50)

    # ------------------------- Core -------------------------
    ap.add_argument("--diff_ckpt", required=True, help="diffusion backbone checkpoint")
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--clf_ckpt", required=True, help="time-conditioned latent classifier checkpoint or folder")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--n_per_dir", type=int, default=500)
    ap.add_argument("--t_start", nargs="+", type=int, default=[50, 100, 200])
    ap.add_argument("--guidance_s", nargs="+", type=float, default=[0.5, 1.0, 1.5, 3.0])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--decode_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--label_col", type=str, default="cnv_status")
    ap.add_argument("--latent_dim", type=int, default=128)

    ap.add_argument("--marker_pos", nargs="*", default=None)
    ap.add_argument("--marker_neg", nargs="*", default=None)
    # ------------------------- Selected candidate biology -------------------------
    ap.add_argument(
        "--pathway_gmt",
        type=str,
        default=None,
        help="Optional GMT file for selected-candidate pathway scoring."
    )
    ap.add_argument(
        "--selected_main_rule",
        type=str,
        default="proba_ge_0.7_min_identity",
        help="Main selected-candidate rule used for biology and trajectory analyses."
    )
    ap.add_argument(
        "--selected_success_proba",
        type=float,
        default=0.70,
        help="Tumor-probability threshold used to define successful selected trajectories."
    )
    ap.add_argument(
        "--selected_de_top_n",
        type=int,
        default=2000,
        help="Number of genes written for selected-candidate DE tables."
    )

    # ------------------------- Trajectory storage -------------------------
    ap.add_argument(
        "--save_all_rep_trajectories",
        action="store_true",
        help=(
            "Store latent trajectory for every generated replicate. "
            "Needed for selected-successful trajectory analysis. "
            "Use mainly for final paper runs because files can become large."
        )
    )
    # ------------------------- Multi-pass -------------------------
    ap.add_argument("--multi_pass_rounds", type=int, default=0)
    ap.add_argument("--multi_pass_t", nargs="+", type=int, default=[50])
    ap.add_argument(
        "--eta", type=float, default=1.0,
        help="Partial update factor for each reverse edit step. "
            "1.0 = full step (old behavior), <1.0 = damped step."
    )
    ap.add_argument("--centroid_guidance_scale", type=float, default=0.0)

    ap.add_argument("--project_before_decode", action="store_true")
    ap.add_argument("--project_n_steps", type=int, default=3)

    ap.add_argument("--sparsity_project", action="store_true")
    ap.add_argument("--sparsity_min_detect_rate", type=float, default=0.0)
    ap.add_argument("--sparsity_max_detect_rate", type=float, default=1.0)

    ap.add_argument("--only_healthy2tumor", action="store_true",
                    help="If set, run only healthy->tumor direction.")
    ap.add_argument("--n_reps_per_seed", type=int, default=1)

    # ------------------------- Nature-methods fixes -------------------------
    ap.add_argument("--vae_gene_order_tsv", type=str, default=None,
                    help="Optional path to gene_order.tsv. If not set, try to find next to vae_ckpt.")
    ap.add_argument("--require_gene_order", action="store_true",
                    help="Fail if gene_order.tsv cannot be found.")
    ap.add_argument("--split_col", type=str, default=None,
                    help="obs column for group-wise split (patient/donor/sample) to reduce leakage.")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--multi_pass_rounds_grid", nargs="+", type=int, default=None)
    ap.add_argument("--multi_pass_t_grid", nargs="+", type=int, default=None)
    ap.add_argument("--track_seeds", type=int, default=3)
    ap.add_argument("--track_rounds_max", type=int, default=50)
    ap.add_argument("--run_continuation", action="store_true",
                    help="Run continuation analysis from selected generated samples.")
    ap.add_argument("--continuation_n_seeds", type=int, default=20)
    ap.add_argument("--continuation_steps", type=int, default=4)
    ap.add_argument("--continuation_pick_mid_min", type=float, default=0.35)
    ap.add_argument("--continuation_pick_mid_max", type=float, default=0.55)
    # ------------------------- Guided baseline -------------------------
    ap.add_argument("--run_guided_baseline", action="store_true",
                    help="Run guided diffusion baseline (single pass) in addition to zigzag.")
    ap.add_argument("--guided_s", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--guided_t", nargs="+", type=int, default=[50, 100, 200])

    # ------------------------- Data mode -------------------------
    ap.add_argument("--h5ad", default=None, help="Single H5AD (Option A). If train/test provided, this is ignored.")
    ap.add_argument("--h5ad_train", default=None, help="TRAIN H5AD (2-file mode).")
    ap.add_argument("--h5ad_test", default=None, help="TEST H5AD (2-file mode).")

    return ap
