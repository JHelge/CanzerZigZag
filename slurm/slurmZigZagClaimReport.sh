#!/bin/bash -l

#SBATCH --job-name=CRC_claim_report
#SBATCH --output=/prj/ml-ident-canc/original_codes/scDiffusion/output%x_%A.out
#SBATCH --error=/prj/ml-ident-canc/original_codes/scDiffusion/error%x_%A.err
#SBATCH --partition=gds
#SBATCH --nodelist=erlenbach
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de

set -euo pipefail

eval "$(/vol/cluster-data/johannes/miniconda3/bin/conda shell.bash hook)"
conda activate myenv_scDiffusion

cd /prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/

SCRIPT=/prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/analysis/zigzag_claim_aligned_report.py

CRC_RUN=/prj/ml-ident-canc/original_codes/scDiffusion/output/continuation/crc_train_test_package_split_correct_eta_0_1_samples_100/status/multi_healthy2tumor/r10_t75_s0.0

python -u "$SCRIPT" \
  --dataset CRC:"$CRC_RUN" \
  --outdir "$CRC_RUN/claim_aligned_report"
