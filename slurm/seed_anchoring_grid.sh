#!/bin/bash -l

#SBATCH --job-name=BRCA_seed_anchoring_grid
#SBATCH --output=/prj/ml-ident-canc/original_codes/scDiffusion/output%x_%A.out
#SBATCH --error=/prj/ml-ident-canc/original_codes/scDiffusion/error%x_%A.err
#SBATCH --partition=gds
#SBATCH --nodelist=erlenbach
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de

set -euo pipefail

eval "$(/vol/cluster-data/johannes/miniconda3/bin/conda shell.bash hook)"
conda activate myenv_scDiffusion

cd /prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/

# ============================================================
# Dataset selection
# ============================================================

# ----------------------------
# BRCA
# ----------------------------
H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_BRCA_aligned_PATIENTSPLIT_seed0_TEST.h5ad
VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_brca_train.pt
OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/brca_train_test_package_split_correct_eta_0_1_samples_100

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
#H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_RCC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
#VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_rcc_train.pt
#OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/rcc_train_test_package_split_correct_eta_0_1_samples_100


# ============================================================
# Fixed settings
# ============================================================

LABEL_COL=status
OUT_DIR=${OUT_ROOT}/${LABEL_COL}
GRID_DIR=${OUT_DIR}/multi_healthy2tumor

SEED_ANCHORING_SCRIPT=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/analysis/selected_seed_anchoring.py
GRID_SCRIPT=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/analysis/run_seed_anchoring_grid.py

REPO_DIR=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat

SELECTION_RULE=proba_ge_0.7_min_identity

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Seed anchoring grid analysis"
echo "============================================================"
echo "H5_TEST: $H5_TEST"
echo "VAE: $VAE"
echo "OUT_ROOT: $OUT_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "GRID_DIR: $GRID_DIR"
echo "SEED_ANCHORING_SCRIPT: $SEED_ANCHORING_SCRIPT"
echo "GRID_SCRIPT: $GRID_SCRIPT"
echo "REPO_DIR: $REPO_DIR"
echo "============================================================"

if [ ! -d "$GRID_DIR" ]; then
  echo "ERROR: GRID_DIR does not exist:"
  echo "$GRID_DIR"
  exit 1
fi

if [ ! -f "$H5_TEST" ]; then
  echo "ERROR: H5_TEST does not exist:"
  echo "$H5_TEST"
  exit 1
fi

if [ ! -e "$VAE" ]; then
  echo "ERROR: VAE checkpoint does not exist:"
  echo "$VAE"
  exit 1
fi

if [ ! -f "$SEED_ANCHORING_SCRIPT" ]; then
  echo "ERROR: selected_seed_anchoring.py not found:"
  echo "$SEED_ANCHORING_SCRIPT"
  exit 1
fi

if [ ! -f "$GRID_SCRIPT" ]; then
  echo "ERROR: run_seed_anchoring_grid.py not found:"
  echo "$GRID_SCRIPT"
  exit 1
fi


# ============================================================
# Run selected-candidate seed anchoring for all r/t grid runs
# ============================================================

python -u "$GRID_SCRIPT" \
  --grid_dir "$GRID_DIR" \
  --h5ad_seed "$H5_TEST" \
  --vae_ckpt "$VAE" \
  --seed_anchoring_script "$SEED_ANCHORING_SCRIPT" \
  --repo_dir "$REPO_DIR" \
  --latent_dim 128 \
  --batch 64 \
  --selection_rule "$SELECTION_RULE" \
  --label_col "$LABEL_COL" \
  --healthy_label healthy \
  --tumor_label tumor \
  --n_perm 1000 \
  --n_random_tumor 1000 \
  --seed 17 \
  --out_csv "$OUT_DIR/selected_seed_anchoring_grid_summary.csv"

echo "============================================================"
echo "DONE"
echo "Wrote:"
echo "$OUT_DIR/selected_seed_anchoring_grid_summary.csv"
echo "============================================================"