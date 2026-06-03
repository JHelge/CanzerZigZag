#!/bin/bash -l

#SBATCH --job-name=RCC_resid_seed_id
#SBATCH --output=/prj/ml-ident-canc/original_codes/scDiffusion/output%x_%A.out
#SBATCH --error=/prj/ml-ident-canc/original_codes/scDiffusion/error%x_%A.err
#SBATCH --partition=gds
#SBATCH --nodelist=erlenbach
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de

set -euo pipefail

eval "$(/vol/cluster-data/johannes/miniconda3/bin/conda shell.bash hook)"
conda activate myenv_scDiffusion

cd /prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/

# ============================================================
# Dataset selection
# Uncomment exactly one dataset block.
# ============================================================

# ----------------------------
# BRCA
# ----------------------------
#H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_BRCA_aligned_PATIENTSPLIT_seed0_TEST.h5ad
#VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_brca_train.pt
#OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/brca_train_test_package_split_correct_eta_0_1_samples_100

# ----------------------------
# CRC
# ----------------------------
#H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_CRC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
#VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_crc_train.pt
#OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/crc_train_test_package_split_correct_eta_0_1_samples_100

# ----------------------------
# LC
# ----------------------------
#H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_LC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
#VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_lc_train.pt
#OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/lc_train_test_package_split_correct_eta_0_1_samples_100

# ----------------------------
# RCC
# ----------------------------
H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_RCC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_rcc_train.pt
OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/rcc_train_test_package_split_correct_eta_0_1_samples_100

LABEL_COL=status
OUT_DIR=${OUT_ROOT}/${LABEL_COL}
GRID_DIR=${OUT_DIR}/multi_healthy2tumor

SCRIPT=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/analysis/residual_seed_identity_grid.py
REPO_DIR=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Residualized seed-identity grid analysis"
echo "============================================================"
echo "H5_TEST: $H5_TEST"
echo "VAE: $VAE"
echo "OUT_ROOT: $OUT_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "GRID_DIR: $GRID_DIR"
echo "SCRIPT: $SCRIPT"
echo "REPO_DIR: $REPO_DIR"
echo "============================================================"

if [ ! -d "$GRID_DIR" ]; then
  echo "ERROR: GRID_DIR does not exist: $GRID_DIR"
  exit 1
fi
if [ ! -f "$H5_TEST" ]; then
  echo "ERROR: H5_TEST does not exist: $H5_TEST"
  exit 1
fi
if [ ! -f "$VAE" ]; then
  echo "ERROR: VAE checkpoint does not exist: $VAE"
  exit 1
fi
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: residual_seed_identity_grid.py not found: $SCRIPT"
  exit 1
fi

python -u "$SCRIPT" \
  --grid_dir "$GRID_DIR" \
  --h5ad_test "$H5_TEST" \
  --vae_ckpt "$VAE" \
  --repo_dir "$REPO_DIR" \
  --selection_rule proba_ge_0.7_min_identity \
  --label_col "$LABEL_COL" \
  --source_label healthy \
  --target_label tumor \
  --latent_dim 128 \
  --batch 64 \
  --sparsity_project \
  --sparsity_min_detect_rate 0.0 \
  --sparsity_max_detect_rate 1.0 \
  --axis_projection_mode target_for_both \
  --max_ref_cells 5000 \
  --max_decode_candidates 2000 \
  --n_perm 1000 \
  --cloud_pairs 20000 \
  --min_bin_candidates 50 \
  --tumor_bins 0.0 0.5 0.6 0.7 0.8 0.9 1.01 \
  --selection_thresholds 0.6 0.7 0.8 \
  --seed 17 \
  --out_csv "$OUT_DIR/residual_seed_identity_grid_summary.csv"

echo "============================================================"
echo "DONE"
echo "Wrote:"
echo "$OUT_DIR/residual_seed_identity_grid_summary.csv"
echo "============================================================"
