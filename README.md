# CancerZigZag

**Iterative seed-anchored diffusion for exploratory generation of tumor-associated single-cell candidate clouds from unpaired epithelial cell populations.**

CancerZigZag is a diffusion-based framework for exploring tumor-associated candidate states from healthy-like single-cell expression profiles when paired healthy--tumor measurements or longitudinal trajectories are unavailable. Instead of predicting one deterministic tumor counterpart for each starting cell, CancerZigZag generates a stochastic candidate cloud initialized from each healthy-like seed cell.

The framework was developed and evaluated in four epithelial cancer contexts:

- Colorectal cancer (CRC)
- Breast cancer (BRCA)
- Lung cancer (LC)
- Renal cell carcinoma (RCC)

## Concept

CancerZigZag separates model learning from seed-initialized generation:

1. A variational autoencoder (VAE) provides a latent representation of single-cell expression profiles.
2. A diffusion model is trained exclusively on tumor-derived epithelial cells from a patient-wise training cohort.
3. Held-out healthy-like epithelial cells are encoded as seed states.
4. Repeated partial latent-space perturbation and tumor-trained reverse diffusion generate stochastic candidate clouds from each seed.
5. Generated candidates are evaluated post hoc relative to held-out healthy-like and tumor-derived reference populations.

CancerZigZag generation is performed without paired healthy--tumor observations and without classifier-based or centroid-based guidance during generation.

## Scope and interpretation

CancerZigZag is intended as an **exploratory candidate-generation framework**. Generated outputs are not interpreted as:

- predictions of a unique future tumor state;
- temporal or lineage-resolved biological trajectories;
- deterministic healthy-to-tumor transformations;
- independently validated malignant cell states.

The reported evaluation is reference-conditioned: held-out reference populations are used for post hoc candidate scoring, expression-space sparsity adjustment, residual seed-structure analysis, and transcriptional-shift concordance.

## Workflow

### 1. Data preparation

Single-cell RNA-sequencing profiles are processed separately for each cancer context. Epithelial cells are separated into:

- **healthy-like cells**: non-malignant epithelial cells;
- **tumor-derived cells**: malignant epithelial cells.

Train/test splitting is performed at the patient level to prevent patient overlap between model training and held-out evaluation.

### 2. VAE fine-tuning

A pretrained scDiffusion/scimilarity-based VAE is fine-tuned on epithelial cells from the training cohort and then kept fixed for downstream diffusion training and candidate generation.

### 3. Tumor-only diffusion training

For each cancer context, a diffusion model is trained only on latent representations of tumor-derived epithelial cells from the training cohort.

### 4. CancerZigZag generation

For each held-out healthy-like seed cell, CancerZigZag repeatedly applies:

1. partial latent-space perturbation to a selected diffusion time index `t`;
2. reverse diffusion using the corresponding tumor-trained diffusion model.

The number of repeated cycles is controlled by `r`, while the damping parameter `eta` controls update strength.

The investigated parameter grid was:

```text
r ∈ {1, 10, 25, 50, 75, 100}
t ∈ {1, 10, 25, 50, 75, 100}
eta = 0.1
```

### 5. Post hoc evaluation

The analysis pipeline includes:

- classifier-estimated tumor-association scoring;
- representative-candidate selection under a predefined success threshold;
- residual representative-candidate anchoring;
- residual candidate-cloud seed structure;
- targeted controls:
  - single-cycle ablation (`r = 1`);
  - matched Gaussian perturbation;
  - linear interpolation toward tumor-cluster centroids;
- latent-space generated-path geometry;
- reference-conditioned transcriptional-shift concordance;
- MSigDB Hallmark pathway-direction agreement.

## Reported configurations

The detailed downstream analyses in the manuscript use the following reference-informed configurations:

| Dataset | ZigZag cycles (`r`) | Perturbation depth (`t`) |
|---|---:|---:|
| CRC | 10 | 75 |
| BRCA | 50 | 75 |
| LC | 25 | 100 |
| RCC | 75 | 100 |

For the detailed reported analyses, 100 stochastic candidates were evaluated for each of 10 held-out healthy-like seeds per cancer context.

## Main findings

Across the reported configurations:

- each of the ten downstream-evaluated held-out seeds per cancer context yielded at least one candidate reaching the predefined post hoc reference-classifier threshold;
- residual representative-candidate anchoring was strongest in CRC and more modest in LC;
- residual organization of complete candidate clouds was weak overall;
- multi-round CancerZigZag yielded threshold-reaching candidates more consistently than single-cycle and matched Gaussian controls in the analysed seed subsets;
- generated latent-space paths were not reducible to direct interpolation toward evaluated tumor-cluster reference modes;
- representative candidates showed positive, context-dependent transcriptional-shift concordance with held-out tumor-associated reference differences.

## Repository structure

Update this section to match the final public repository layout:

```text
.
├── analysis/                # Downstream evaluation and targeted-control scripts
├── configs/                 # Run configurations, if provided
├── scripts/                 # Generation/training scripts or SLURM wrappers
├── figures/                 # Figure-generation scripts or exported figures, if provided
├── README.md
└── LICENSE
```

## Installation

Create a Python environment compatible with the original scDiffusion implementation and install the dependencies required by the repository. Core dependencies are expected to include:

```text
python
numpy
pandas
scipy
scikit-learn
scanpy
anndata
torch
matplotlib
```

CancerZigZag builds on the scDiffusion framework. The scDiffusion codebase and required pretrained VAE resources must be accessible in the environment used for training, generation, and evaluation.

## Data availability

The analyses use epithelial-cell single-cell transcriptomic data from the pan-cancer tumor--normal atlas by Kang et al. The associated public dataset is available through the Zenodo resource cited in the manuscript.

This repository does not redistribute the complete source atlas data. Users should obtain the data from the original public resource and prepare cancer-context-specific train/test inputs according to the preprocessing workflow described in the manuscript.

## Running the workflow

The full workflow consists of:

```text
1. Prepare patient-wise train/test data for each cancer context.
2. Fine-tune the VAE on the training cohort.
3. Train a tumor-only diffusion model in the VAE latent space.
4. Run CancerZigZag candidate generation.
5. Run pipeline-compatible downstream evaluation and targeted controls.
```

Add the exact executable commands from the final repository here before release. At minimum, document:

```bash
# Replace these placeholders with the exact repository commands.
python <vae_finetuning_script>.py --help
python <diffusion_training_script>.py --help
python <cancerzigzag_generation_script>.py --help
python <evaluation_script>.py --help
```

For cluster-based execution, include the SLURM scripts used for generation and downstream evaluation together with the required input paths, environment name, expected output directories, and hardware requirements.

## Evaluation notes

### Reference-conditioned expression-space evaluation

For healthy-to-tumor generation, VAE-decoded generated candidates are subjected to a gene-wise sparsity projection guided by detection rates in the held-out tumor-derived epithelial reference population before expression-space evaluation. Decoded originating seed states used for expression-space seed-distance calculations are separately sparsity-projected according to the held-out healthy-like seed subset.

Consequently, expression-space classifier scores, seed-distance summaries, residual expression-space structure, and transcriptional-shift analyses are post hoc, reference-conditioned summaries rather than analyses of an unmodified common decoded expression representation.

### Unguided generation

The post hoc reference classifier is used only for evaluation and representative-candidate selection. It does not guide CancerZigZag generation.

## Citation

A manuscript describing CancerZigZag is currently in preparation/submission. Please cite the accompanying manuscript once final citation details are available.

```bibtex
@article{CancerZigZag,
  title   = {CancerZigZag: Iterative Seed-Anchored Diffusion for Generative Modeling of Single-Cell State Transitions},
  author  = {Schl{\\"u}ter, Johannes and Sch{\\"o}nhuth, Alexander},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

## License

Add the selected software license before public release and include the corresponding `LICENSE` file in the repository.

## Contact

**Johannes Schlüter**  
Faculty of Technology, Bielefeld University  
Contact details are provided in the accompanying manuscript.
