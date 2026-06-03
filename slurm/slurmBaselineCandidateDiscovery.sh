#!/bin/bash -l
#SBATCH --job-name=CRC_baseline_v2
#SBATCH --output=/prj/ml-ident-canc/original_codes/scDiffusion/output%x_%A.out
#SBATCH --error=/prj/ml-ident-canc/original_codes/scDiffusion/error%x_%A.err
#SBATCH --partition=gds
#SBATCH --nodelist=erlenbach
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de

set -euo pipefail

eval "$(/vol/cluster-data/johannes/miniconda3/bin/conda shell.bash hook)"
conda activate myenv_scDiffusion
cd /prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/

SCRIPT=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/analysis/baseline_candidate_discovery.py
REPO_DIR=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat

# CRC final: r10,t75
H5_TRAIN=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_CRC_aligned_PATIENTSPLIT_seed0_TRAIN.h5ad
H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_CRC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_crc_train.pt
OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/crc_train_test_package_split_correct_eta_0_1_samples_100
RUN_DIR=${OUT_ROOT}/status/multi_healthy2tumor/r10_t75_s0.0
SINGLE_CYCLE_RUN_DIR=${OUT_ROOT}/status/multi_healthy2tumor/r1_t75_s0.0

python -u "$SCRIPT" \
  --run_dir "$RUN_DIR" \
  --single_cycle_run_dir "$SINGLE_CYCLE_RUN_DIR" \
  --h5ad_train "$H5_TRAIN" \
  --h5ad_test "$H5_TEST" \
  --vae_ckpt "$VAE" \
  --repo_dir "$REPO_DIR" \
  --label_col status \
  --healthy_label healthy \
  --tumor_label tumor \
  --latent_dim 128 \
  --batch 64 \
  --budget 100 \
  --interp_alphas 0.0 0.1 0.2 0.35 0.5 0.75 1.0 1.25 1.5 \
  --n_tumor_clusters 8 \
  --sparsity_project \
  --sparsity_min_detect_rate 0.0 \
  --sparsity_max_detect_rate 1.0 \
  --max_ref_cells 5000 \
  --n_perm 1000 \
  --seed 17 \
  --tumor_score_col progress01_geom \
  --tumor_thresholds 0.7 1.0 \
  --distance_cols id_dist_expr resid_id_dist_expr \
  --knn_k 15 \
  --realism_quantile 0.95 \
  --out_prefix baseline_valid


echo "DONE: $RUN_DIR/baseline_v2_selected_summary.csv"
