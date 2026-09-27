# NPS Spectral Intelligence Platform

A reproducible, end-to-end mass-spectrometry toolkit for detecting and categorizing new psychoactive substances (NPS). The project combines SimCLR representation pretraining, supervised binary and multi-class classifiers, and FlashEntropySearch library matching in a Streamlit application.

> **Reviewer note:** The source tree is directly inspectable, but the large source/reference spectra, preprocessed Parquet files, and prebuilt entropy-search index are intentionally excluded by `.gitignore`. Complete instructions for placing or rebuilding these artifacts are provided below. A release archive intended for reviewers should include the source archive plus the model/data artifacts listed in [Required runtime artifacts](#required-runtime-artifacts). A ready-to-run Docker image is published on Docker Hub and is the recommended way to obtain a working deployment in all cases (see [Quick start with Docker](#quick-start-with-docker)).

## Contents

- [Quick start with Docker](#quick-start-with-docker)
- [Scientific workflow](#scientific-workflow)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Required runtime artifacts](#required-runtime-artifacts)
- [Build the library-search index](#build-the-library-search-index)
- [Run the Streamlit application](#run-the-streamlit-application)
- [Run inference from Python](#run-inference-from-python)
- [Input format and example](#input-format-and-example)
- [Expected outputs and benchmark results](#expected-outputs-and-benchmark-results)
- [Reproduce preprocessing and training](#reproduce-preprocessing-and-training)
- [Docker](#docker)
- [Verification checklist for reviewers](#verification-checklist-for-reviewers)
- [Reproducibility notes and limitations](#reproducibility-notes-and-limitations)
- [Data privacy, citation, and license](#data-privacy-citation-and-license)

## Quick start with Docker

**Docker deployment is recommended in all cases.** A prebuilt, deployable image is published at [hub.docker.com/r/molyleaf/cannabinoids](https://hub.docker.com/repository/docker/molyleaf/cannabinoids). It pins the Python version and every dependency, and bundles the application code, the model weights, and the prebuilt entropy-search index, so no artifact staging or local environment setup is required. The currently recommended tag is `1.0.6`; all published tags are versioned (there is no `latest` tag).

```bash
docker pull molyleaf/cannabinoids:1.0.6
docker run --rm -p 8001:8001 molyleaf/cannabinoids:1.0.6
```

Open `http://localhost:8001/cannabinoids` and upload an MSP file.

To substitute newer checkpoints or a rebuilt library index, mount them read-only over the in-image copies:

```bash
docker run --rm -p 8001:8001 \
  -v "$(pwd)/app/models:/srv/cannabinoids/app/models:ro" \
  -v "$(pwd)/app/cache/positive_idx:/srv/cannabinoids/app/cache/positive_idx:ro" \
  molyleaf/cannabinoids:1.0.6
```

See [Docker](#docker) for further details, and [Installation](#installation) only if Docker is unavailable on your host or you need a native environment for development, training, and analysis.

## Scientific workflow

1. **Parsing and cleaning.** MSP-style records are parsed into `(m/z, intensity)` peak lists. `ms-entropy.clean_spectrum` removes outlier/duplicate peaks and normalizes spectra.
2. **Feature construction.** Peaks in `m/z = 40` through `600` are binned at 1 Da resolution. The result is a 561-dimensional vector, followed by total-ion-current normalization and square-root scaling.
3. **Self-supervised pretraining.** A one-dimensional convolutional encoder is trained with SimCLR using the NT-Xent loss, LARS, spectrum-specific augmentation, and an optional multi-GPU DDP workflow.
4. **Binary classification.** A classifier is attached to the pretrained encoder. The documented production configuration uses 5-fold cross-validation, soft-voting ensemble inference, dynamic spectrum augmentation, Focal Loss, and a decision threshold of `0.50`.
5. **Nine-class classification.** A second encoder/classifier predicts one of nine NPS categories: Fentanyls, Cathinones, Synthetic Cannabinoids, Arylcyclohexylamines, Benzodiazepines, Nitazenes, Opiates, Phenethylamines, and Tryptamines. In the application, multi-class confidence below `0.98` is reported as out-of-distribution (OOD).
6. **Known-compound retrieval.** For samples predicted positive by the binary model, FlashEntropySearch queries a prebuilt positive library. The default match threshold is `0.80`, with MS1 and MS2 tolerances of `0.2 Da` and `0.02 Da`, respectively.
7. **Web inference.** The Streamlit interface accepts an uploaded spectrum, displays the model result, cleaned spectrum, probabilities, cleaned peaks, and an optional matching SMILES structure.

## Repository layout

```text
.
├── app/
│   ├── models/                  # Model architectures and weight loader
│   ├── streamlit/               # Streamlit UI
│   ├── cache/positive_idx/      # Generated FlashEntropySearch index (not committed)
│   ├── data_queue/              # Opt-in contribution and review queues
│   ├── pipeline.py              # Single-sample inference pipeline
│   ├── recognizer.py            # Entropy library matching
│   └── server.py                # Streamlit launcher
├── common/
│   ├── augmentation.py          # Spectrum augmentation strategies
│   ├── data_processor.py        # MSP parsing, cleaning, binning, normalization
│   └── lars.py                  # LARS optimizer
├── embedding_pretrain/
│   ├── data_preprocess.py       # MSP -> normalized Parquet
│   ├── main.py                  # SimCLR pretraining entry point
│   ├── evaluate_best_model.py   # Embedding-quality diagnostics
│   └── lib/                     # Models, losses, utilities
├── finetune/
│   ├── data_preprocess.py       # Labeled MSP -> binary-classification Parquet
│   ├── main.py                  # 5-fold binary fine-tuning
│   ├── convert_to_safetensors.py
│   └── lib/                     # Dataset, trainer, model, evaluation helpers
├── entropy/
│   └── build_index.py           # Build the positive-library search index
├── multi_classifier/            # Nine-class experiment notebook
├── pca/ and heatmap/            # Optional embedding/attribution analyses
├── scratch/                     # Diagnostic and evaluation scripts
├── Modelcard.md                 # Model-card template/notes
└── pyproject.toml               # Core Python dependency definition
```

## Requirements

- Python `>=3.13` as declared in `pyproject.toml`.
- CPU inference is supported. CUDA is recommended for pretraining and fine-tuning.
- Docker is the recommended way to deploy the application in all cases; a prebuilt image is published on Docker Hub (see [Quick start with Docker](#quick-start-with-docker)). A native installation is intended for development, training, and analysis, or for hosts where Docker is unavailable.
- Full training and analysis also use `pandas`, `pyarrow`, `scikit-learn`, `seaborn`, and `tqdm`; these training-only packages are not all listed in the core `pyproject.toml` dependency section.

## Installation

**Docker deployment is recommended in all cases** and requires none of the steps in this section — see [Quick start with Docker](#quick-start-with-docker). The native installation below is needed only for development, training, and analysis, or on hosts where Docker is unavailable.

### Linux or macOS

```bash
git clone <repository-url>
cd cannabinoids-master

python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .

# Training and analysis dependencies:
python -m pip install pandas pyarrow scikit-learn seaborn tqdm openpyxl
```

### Windows PowerShell

```powershell
git clone <repository-url>
cd cannabinoids-master

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .

# Training and analysis dependencies:
python -m pip install pandas pyarrow scikit-learn seaborn tqdm openpyxl
```

If PowerShell blocks virtual-environment activation for the current shell only, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

An optional `uv` workflow is also supported because dependency groups and a PyTorch index are defined in `pyproject.toml`:

```bash
uv sync --group deploy
uv sync --group deploy --group dev
```

## Required runtime artifacts

The application cannot perform inference from source code alone. This section applies to native deployments from a source checkout; the prebuilt Docker image already bundles the weights and index described here. Place the following artifacts in these exact locations before starting the UI:

```text
app/models/
├── official_ensemble_fold_1.safetensors
├── official_ensemble_fold_2.safetensors
├── official_ensemble_fold_3.safetensors
├── official_ensemble_fold_4.safetensors
├── official_ensemble_fold_5.safetensors
└── best_multi_class_model_YYYYMMDD_HHMMSS.pt   # or .safetensors

app/cache/positive_idx/                         # Files generated by entropy/build_index.py
```

Model-loading behavior:

- With all five `official_ensemble_fold_*.safetensors` files, the binary model is evaluated as a five-model soft-voting ensemble.
- With only one valid binary model file, the loader falls back to single-model inference and prints a warning.
- With no valid binary model file, the application raises `FileNotFoundError`.
- The multi-class loader searches for `best_multi_class_model_*.pt`, `best_multi_class_model_*.safetensors`, `*multi_class*.pt`, and `*multi_class*.safetensors` in `app/models/`.

If model weights are distributed separately, add their download URL and SHA-256 checksums to the release bundle. Do not commit large weights directly unless the repository's storage policy permits it.

## Build the library-search index

`entropy/build_index.py` checks, in order:

1. `finetune/data_source/阳性-含CanonicalSMILES-5类骨架(4).msp`
2. `entropy/positive.msp`

Place the positive reference library at one of those paths and run:

```bash
python entropy/build_index.py
```

Expected terminal output is similar to:

```text
Reading spectra from .../positive.msp...
Loaded <N> spectra. Building FlashEntropySearch index...
Writing index to .../app/cache/positive_idx...
Index build completed successfully!
```

For a custom library without modifying the script:

```bash
python -c "from entropy.build_index import build_positive_index; build_positive_index('path/to/positive_library.msp')"
```

The index is required only when a positive prediction triggers known-compound retrieval. It is not required for loading the classifiers themselves.

## Run the Streamlit application

From the repository root:

```bash
python app/server.py
```

The default server settings are:

- URL: `http://localhost:8001/cannabinoids`
- Bind address: `0.0.0.0`
- Port: `8001`
- Streamlit base URL path: `cannabinoids`

The equivalent direct command is:

```bash
python -m streamlit run app/streamlit/app.py \
  --server.port=8001 \
  --server.address=0.0.0.0 \
  --server.baseUrlPath=cannabinoids
```

Workflow in the browser:

1. Select the binary model or the nine-class model.
2. Set the library similarity threshold (default `0.80`).
3. Choose whether to consent to data sharing.
4. Upload an `.msp` or `.mgf` file.
5. Inspect the prediction, cleaned spectrum, probabilities, and retrieval result.

## Run inference from Python

The public entry point is `app.pipeline.run_pipeline()`:

```python
from pathlib import Path
import json

from app.pipeline import run_pipeline

spectrum_path = Path("example.msp")
result = run_pipeline(
    file_bytes=spectrum_path.read_bytes(),
    filename=spectrum_path.name,
    model_type="binary",       # "binary" or "multi"
    min_similarity=0.80,
)

print(json.dumps(result, indent=2))
```

For a batch of spectra, use `app.pipeline.run_pipeline_batch()`. It returns aggregate risk statistics and per-spectrum details.

## Input format and example

The parser expects MSP-style key-value records. At minimum, each record should contain `Name:` and a peak table. `PrecursorMZ:` improves identity-mode entropy matching.

```text
Name: Example Compound
PrecursorMZ: 287.2
Num Peaks: 7
55.1 12.0
77.2 28.0
91.1 100.0
105.2 43.0
119.1 31.0
147.2 19.0
287.2 8.0
```

A file may contain multiple records separated by a new `Name:` line. The web UI analyzes the first record; `run_pipeline_batch()` processes all parsed records.

**MGF caveat:** files with `.mgf` extensions are accepted by the uploader, but the current parser is MSP-oriented and expects a `Name:` line. Plain `BEGIN IONS`/`TITLE=`/`PEPMASS=` records may require conversion to MSP format before upload.

## Expected outputs and benchmark results

### Application/API output schema

A successful binary prediction returns a dictionary with this structure. Exact probabilities depend on the selected model and input spectrum.

```json
{
  "filename": "example.msp",
  "name": "Example Compound",
  "num_cleaned_peaks": 7,
  "precursor_mz": 287.2,
  "model_type_selected": "binary",
  "is_positive": true,
  "is_ood": false,
  "model_inference": {
    "model_type": "binary",
    "model_name": "Binary Model (SC versus NSC)",
    "risk_probability": 0.93,
    "risk_percentage": "93.00%",
    "risk_level": "High Risk",
    "is_positive": true,
    "is_ood": false,
    "status_text": "Positive (High Risk)"
  },
  "entropy_match": {
    "is_triggered": true,
    "is_matched": true,
    "is_ood": false,
    "similarity_score": 0.91,
    "matched_smiles": "CCCCC...",
    "matched_name": "Reference compound",
    "min_similarity_threshold": 0.8,
    "message": "Known molecule matched..."
  },
  "peaks": [[55.1, 12.0], [77.2, 28.0]]
}
```

Interpretation:

- `model_inference.risk_probability >= 0.50` is a positive/high-risk binary prediction.
- For the multi-class model, the top class and its confidence are reported in `pred_class` and `confidence`; confidence below `0.98` is returned as `OOD`.
- Library matching is triggered only for positive, non-OOD samples.
- A positive sample below the selected similarity threshold may indicate a novel positive derivative and returns no SMILES.

### Included binary benchmark record

The checked-in evaluation artifacts under `finetune/scratch/` report the following threshold-`0.50` results. These are evidence records from the supplied run; rerunning the full pipeline is the independent verification step.

| Split | Samples | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 | Specificity | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,433 | 1,436 | 33 | 10 | 954 | 0.9823 | 0.9666 | 0.9896 | 0.9780 | 0.9775 | 0.9983 |
| Validation | 495 | 310 | 4 | 7 | 174 | 0.9778 | 0.9775 | 0.9613 | 0.9694 | 0.9873 | 0.9962 |
| Independent test | 525 | 304 | 10 | 11 | 200 | 0.9600 | 0.9524 | 0.9479 | 0.9501 | 0.9682 | 0.9861 |

Other expected artifacts from the binary fine-tuning run include:

- Five `official_ensemble_fold_*.pt` checkpoints.
- Confusion matrices and ROC curves for train, validation, and test splits.
- SMILES-level positive performance tables.
- False-positive and false-negative compound lists.
- An Excel workbook containing evaluation tables and figures.

The included pretraining loss record (`embedding_pretrain/pretraining_loss.csv`) contains 500 epochs and ends at a loss of approximately `0.560243` at epoch 500. This file is a historical result, not a guarantee of identical convergence on different hardware.

## Reproduce preprocessing and training

All commands below are run from the repository root and assume the training-only dependencies from [Installation](#installation) are installed.

### 1. Preprocess pretraining spectra

Input MSP default:

```text
embedding_pretrain/data_source/combined_spectra_normalized.msp
```

Command:

```bash
python embedding_pretrain/data_preprocess.py \
  --input_msp embedding_pretrain/data_source/combined_spectra_normalized.msp \
  --output_parquet embedding_pretrain/data_source/preprocessed_spectra.parquet \
  --min_peaks 5 \
  --mz_min 40 \
  --mz_max 600
```

Output columns:

| Column | Type | Description |
|---|---|---|
| `name` | string | Spectrum/compound name |
| `smiles` | string | Structure when available |
| `precursor_mz` | float64 | Precursor m/z when available |
| `num_peaks` | int32 | Number of cleaned peaks |
| `spectrum_vector` | list[float32] | Normalized 561-dimensional vector |

### 2. Pretrain the SimCLR encoder

Single-process example, useful for smoke testing:

```bash
python embedding_pretrain/main.py \
  --data_path embedding_pretrain/data_source/preprocessed_spectra.parquet \
  --output_dir embedding_pretrain/results_reproduction \
  --max_samples 10000 \
  --max_epochs 10
```

Documented two-GPU configuration:

```bash
torchrun --nproc_per_node=2 embedding_pretrain/main.py \
  --data_path embedding_pretrain/data_source/preprocessed_spectra.parquet \
  --output_dir embedding_pretrain/results_reproduction \
  --global_batch_size 2048 \
  --base_lr 0.030 \
  --warmup_epochs 10 \
  --max_epochs 500
```

The run directory contains `best_model.pt`; the final encoder is also written as `embedding_pretrain/pretrained_encoder_final_v1.pt`. The Slurm wrapper is `embedding_pretrain/main.sbatch`.

### 3. Preprocess the labeled binary dataset

Expected default inputs:

```text
finetune/data_source/阳性-含CanonicalSMILES-5类骨架(4).msp
finetune/data_source/阴性(4).msp
```

Command:

```bash
python finetune/data_preprocess.py \
  --pos_msp "finetune/data_source/阳性-含CanonicalSMILES-5类骨架(4).msp" \
  --neg_msp "finetune/data_source/阴性(4).msp" \
  --output_parquet finetune/data_source/preprocessed_finetune.parquet
```

Positive records receive label `1`; negative records receive label `0`.

### 4. Fine-tune and evaluate the binary ensemble

```bash
python finetune/main.py \
  --encoder_path embedding_pretrain/results_reproduction/best_model.pt \
  --n_folds 5 \
  --batch_size 128 \
  --warmup_epochs 15 \
  --unfreeze_epochs 75 \
  --encoder_lr 3e-5 \
  --classifier_lr 3e-4 \
  --patience 15 \
  --threshold 0.50
```

The split uses seed `42`. Positive compounds are grouped by SMILES before splitting, which prevents the same structure from appearing across train, validation, and test partitions. The default held-out proportions are 15% validation and 15% test.

Outputs are written to `finetune/results_<timestamp>/`, including five fold checkpoints, metrics, plots, and prediction tables.

### 5. Convert ensemble checkpoints for the application

`finetune/convert_to_safetensors.py` currently contains a hard-coded `results_dir` pointing to a historical run. Edit that variable to point to the newly generated fine-tuning result directory, then run:

```bash
python finetune/convert_to_safetensors.py
```

The script writes `official_ensemble_fold_1.safetensors` through `official_ensemble_fold_5.safetensors` into both the result directory and `app/models/`.

### 6. Train the nine-class model and run optional analyses

- Open `multi_classifier/多分类.ipynb` to reproduce nine-class training and prediction.
- Use `pca/降维聚类.py` for PCA/t-SNE embedding visualization and model summaries.
- Use `heatmap/grad ana.py` for gradient-based spectral attribution.
- Use scripts under `scratch/` and `finetune/scratch/` for diagnostics and threshold analyses.

These notebooks/scripts contain some historical paths. Review and update input/output paths before rerunning.

## Docker

Docker deployment is recommended in all cases.

### Option 1 — Use the prebuilt image (recommended)

The deployable image is published at [hub.docker.com/r/molyleaf/cannabinoids](https://hub.docker.com/repository/docker/molyleaf/cannabinoids). Versioned tags are `1.0.0` through `1.0.6`; the currently recommended tag is `1.0.6`. The image is self-contained: it bundles the application, the `common/` package, the committed model weights in `app/models/`, and the prebuilt FlashEntropySearch index, and it runs as an unprivileged user.

```bash
docker pull molyleaf/cannabinoids:1.0.6
docker run --rm -p 8001:8001 molyleaf/cannabinoids:1.0.6
```

Open `http://localhost:8001/cannabinoids`. No volume mounts are required.

To run the image with locally supplied model weights or a rebuilt entropy index instead of the in-image copies, mount them read-only:

```bash
docker run --rm -p 8001:8001 \
  -v "$(pwd)/app/models:/srv/cannabinoids/app/models:ro" \
  -v "$(pwd)/app/cache/positive_idx:/srv/cannabinoids/app/cache/positive_idx:ro" \
  molyleaf/cannabinoids:1.0.6
```

Note that the image does not contain the restricted source datasets (`data_source/` is excluded from the repository), so training data must still be provided separately for reproduction.

### Option 2 — Build from source

Build from the repository root:

```bash
docker build -f app/Dockerfile -t nps-spectral-app .
```

The build context includes whatever is currently present in `app/` and `common/` (there is no `.dockerignore`), so an image built from a complete working tree — with model weights in `app/models/` and the index in `app/cache/positive_idx/` — is self-contained in the same way as the published image. Run it with the same commands as above, substituting `nps-spectral-app` for `molyleaf/cannabinoids:1.0.6`.

## Verification checklist for reviewers

The fastest verification path uses the prebuilt Docker image, which is recommended in all cases:

```bash
docker run --rm -p 8001:8001 molyleaf/cannabinoids:1.0.6
```

Open `http://localhost:8001/cannabinoids`, upload a valid MSP record (see [Input format and example](#input-format-and-example)), and confirm that a prediction with probabilities is returned.

For a source-level check of the repository itself, a minimal independent execution check is:

```bash
python -c "from app.pipeline import run_pipeline; print(run_pipeline.__name__)"
python scratch/test_i18n.py
python entropy/build_index.py
python scratch/test_pipeline.py
python app/server.py
```

Expected verification outcomes:

- The imports complete without errors.
- The i18n smoke test prints English and Chinese translations.
- The entropy index is written to `app/cache/positive_idx/`.
- `scratch/test_pipeline.py` prints a predicted class, entropy score, SMILES/name fields, and a probability-key list when `entropy/positive.msp`, model weights, and the index are present.
- The browser opens at `http://localhost:8001/cannabinoids` and returns results for a valid MSP upload.

For direct reproduction of the reported binary result, rerun preprocessing, fine-tuning, and evaluation from the original labeled datasets and compare the generated test confusion matrix with the table above.

## Reproducibility notes and limitations

- The current `.gitignore` excludes `data_source/`, `data_preprocessed/`, `*.npy`, `*.msp`, `*.xlsx`, and `*.parquet`, including the generated index directory `app/cache/positive_idx/`. The application model weights under `app/models/` are committed, so inference can be reproduced from a source checkout once the entropy-search index is rebuilt from a separately supplied positive library (see [Build the library-search index](#build-the-library-search-index)) or obtained from the prebuilt Docker image. Full training reproduction additionally requires the original labeled datasets.
- Exact GPU/CUDA versions, nondeterministic GPU kernels, and library versions can cause small numerical differences. Split seed `42` controls the documented binary train/validation/test partition.
- The current checkout has no configured Git remote. Before submission, publish the repository or release bundle and replace `<repository-url>` with the durable URL.
- The UI accepts `.mgf`, but the parser is primarily MSP-compatible as noted above.
- Model confidence and entropy similarity are decision-support outputs, not definitive forensic identifications.
- The nine-class OOD rule (`confidence < 0.98`) and binary threshold (`0.50`) are application defaults; validate them on the target instrument and dataset before operational use.

## Data privacy, citation, and license

- If the user does **not** consent to data sharing, the application does not persist uploaded spectra.
- If consent is given, contributions are appended to `app/data_queue/authorized_contributions.jsonl`; binary probabilities in `[0.3, 0.7]` are additionally placed in `app/data_queue/uncertain_samples.jsonl` for manual review.
- Do not publish raw spectra or contributor information without appropriate consent and institutional approval.
- Add the article citation, dataset DOI, model-weight DOI/URL, repository commit hash, and a software license before the public/reviewer release. No license file is included in the current source tree.
- The checklist reference provided with this submission is licensed under CC BY 4.0; that checklist license does not automatically license this software or its datasets.
