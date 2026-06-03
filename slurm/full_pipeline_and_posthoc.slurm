#!/bin/bash -l

#SBATCH --job-name=LC_zigzag_full_pipeline
#SBATCH --output=/prj/ml-ident-canc/original_codes/scDiffusion/output%x_%A.out
#SBATCH --error=/prj/ml-ident-canc/original_codes/scDiffusion/error%x_%A.err
#SBATCH --partition=gds
#SBATCH --nodelist=wachbach
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de

set -euo pipefail

eval "$(/vol/cluster-data/johannes/miniconda3/bin/conda shell.bash hook)"
conda activate myenv_scDiffusion

cd /prj/ml-ident-canc/original_codes/scDiffusion/zigzag_refactored_safe_chat/

CLF_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/classifiers

#BRCA
#H5_TRAIN=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_BRCA_aligned_PATIENTSPLIT_seed0_TRAIN.h5ad
#H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_BRCA_aligned_PATIENTSPLIT_seed0_TEST.h5ad
#DIFF=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/backbone/finetuned/brca_train/open_problem/model010000.pt
#VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_brca_train.pt
#OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/afinal/brca_train_test_package_split_correct_eta_0_1_samples_100

#CRC
#H5_TRAIN=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_CRC_aligned_PATIENTSPLIT_seed0_TRAIN.h5ad
#H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_CRC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
#DIFF=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/backbone/finetuned/crc_train/open_problem/model010000.pt
#VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_crc_train.pt
#OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/afinal/crc_train_test_package_split_correct_eta_0_1_samples_100

#LC
H5_TRAIN=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_LC_aligned_PATIENTSPLIT_seed0_TRAIN.h5ad
H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_LC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
DIFF=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/backbone/finetuned/lc_train/open_problem/model010000.pt
VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_lc_train.pt
OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/afinal/lc_train_test_package_split_correct_eta_0_1_samples_100

#RCC
#H5_TRAIN=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_RCC_aligned_PATIENTSPLIT_seed0_TRAIN.h5ad
#H5_TEST=/prj/ml-ident-canc/scDiffusion/data/KangData/atlas_dataset/aligned_by_celltype/Epithelial_RCC_aligned_PATIENTSPLIT_seed0_TEST.h5ad
#DIFF=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/backbone/finetuned/rcc_train/open_problem/model010000.pt
#VAE=/prj/ml-ident-canc/original_codes/scDiffusion/output/checkpoint/vae_finetuned_rcc_train.pt
#OUT_ROOT=/prj/ml-ident-canc/original_codes/scDiffusion/output/afinal/rcc_train_test_package_split_correct_eta_0_1_samples_100



mkdir -p "$OUT_ROOT"

LABELS=("status")
CLF_NAMES=("label_cnv")

# Controls:
#   RUN_PIPELINE=1 RUN_POSTHOC=0  -> full run (incl posthoc unless you add --skip_posthoc)
#   RUN_PIPELINE=1 RUN_POSTHOC=0 + add --skip_posthoc -> run only (no posthoc)
#   RUN_PIPELINE=0 RUN_POSTHOC=1  -> posthoc only (requires outdir already exists with saved eval refs + summary csv)
RUN_PIPELINE=${RUN_PIPELINE:-1}
RUN_POSTHOC=${RUN_POSTHOC:-0}

for IDX in "${!LABELS[@]}"; do
  LABEL_COL=${LABELS[$IDX]}
  CLF_NAME=${CLF_NAMES[$IDX]}
  CLF_DIR=${CLF_ROOT}/${CLF_NAME}
  OUT_DIR=${OUT_ROOT}/${LABEL_COL}
  mkdir -p "$OUT_DIR"


  if [ "$RUN_PIPELINE" -eq 1 ]; then
    python -u run.py \
      --h5ad_train "$H5_TRAIN" \
      --h5ad_test "$H5_TEST" \
      --diff_ckpt "$DIFF" \
      --vae_ckpt "$VAE" \
      --clf_ckpt "$CLF_DIR" \
      --outdir "$OUT_DIR" \
      --n_per_dir 100 \
      --t_start 0 \
      --guidance_s 0.0 \
      --eta 0.1 \
      --n_reps_per_seed 10 \
      --multi_pass_rounds_grid 25 \
      --multi_pass_t_grid 100 \
      --batch 64 \
      --decode_frac 1.0 \
      --label_col "$LABEL_COL" \
      --latent_dim 128 \
      --seed 17 \
      --eval_min_knn 0.0 \
      --eval_min_centroid 0.0 \
      --centroid_guidance_scale 0.0 \
      --sparsity_project \
      --sparsity_min_detect_rate 0.0 \
      --sparsity_max_detect_rate 1.0 \
      --only_healthy2tumor \
      --track_seeds 4 \
      --track_rounds_max 100 \
      --run_continuation \
      --continuation_n_seeds 10 \
      --continuation_steps 4 \
      --continuation_pick_mid_min 0.35 \
      --continuation_pick_mid_max 0.55 \
      --save_all_rep_trajectories \
      --selected_main_rule proba_ge_0.7_min_identity \
      --selected_success_proba 0.7 \
      --pathway_gmt /prj/ml-ident-canc/original_codes/data/h.all.v2026.1.Hs.symbols.gmt
  fi

  if [ "$RUN_POSTHOC" -eq 1 ]; then
    python -u posthoc.py --outdir "$OUT_DIR"
  fi
done

