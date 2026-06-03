#!/bin/bash -l

#SBATCH --job-name=CRC_fair_baseline_eval
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

SCRIPT=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/analysis/fair_baseline_evaluator.py
REPO_DIR=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat

# CRC final
H5_TRAIN=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_CRC_aligned_PATIENTSPLIT_seed0_TRAIN.h5ad
H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_CRC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_crc_train.pt
RUN_DIR=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/crc_train_test_package_split_correct_eta_0_1_samples_100/status/multi_healthy2tumor/r10_t75_s0.0

python -u "$SCRIPT" \
  --run_dir "$RUN_DIR" \
  --h5ad_train "$H5_TRAIN" \
  --h5ad_test "$H5_TEST" \
  --vae_ckpt "$VAE" \
  --repo_dir "$REPO_DIR" \
  --label_col status \
  --healthy_label healthy \
  --tumor_label tumor \
  --latent_dim 128 \
  --batch 64 \
  --eval_pcs 50 \
  --max_ref_cells 5000 \
  --success_proba 0.7 \
  --n_perm 1000 \
  --cloud_pairs 20000 \
  --knn_k 15 \
  --sparsity_project \
  --sparsity_min_detect_rate 0.0 \
  --sparsity_max_detect_rate 1.0 \
  --pathway_gmt /prj/ml-ident-canc/original_codes/data/h.all.v2026.1.Hs.symbols.gmt \
  --out_prefix fair_baseline \
  --seed 17
