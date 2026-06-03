#!/bin/bash -l

#SBATCH --job-name=CRC_eval_baseline_pools
#SBATCH --output=/prj/ml-ident-canc/original_codes/scDiffusion/output%x_%A.out
#SBATCH --error=/prj/ml-ident-canc/original_codes/scDiffusion/error%x_%A.err
#SBATCH --partition=gds
#SBATCH --nodelist=erlenbach
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de

set -euo pipefail

eval "$(/vol/cluster-data/johannes/miniconda3/bin/conda shell.bash hook)"
conda activate myenv_scDiffusion

cd /prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/

SCRIPT=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/analysis/evaluate_baseline_pools_with_pipeline.py
REPO_DIR=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat

# CRC final
RUN_DIR=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/crc_train_test_package_split_correct_eta_0_1_samples_100/status/multi_healthy2tumor/r10_t75_s0.0

python -u "$SCRIPT" \
  --run_dir "$RUN_DIR" \
  --repo_dir "$REPO_DIR" \
  --run_posthoc \
  --posthoc_script posthoc.py \
  --selection_rule proba_ge_0.7_min_identity
