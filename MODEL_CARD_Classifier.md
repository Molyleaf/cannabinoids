---
library_name: pytorch
tags:
  - mass-spectrometry
  - gc-ei-ms
  - new-psychoactive-substances
  - synthetic-cannabinoids
  - binary-classification
---

# Model Card for Cannabinoid Binary Classifier (SC vs NSC)

## Summary

This model is a PyTorch binary classifier for GC-EI mass spectra. Given a binned mass spectrum, it estimates whether the spectrum belongs to the synthetic cannabinoid (SC, positive) class or the non-synthetic cannabinoid (NSC, negative) class.

Production inference uses a five-fold soft-voting ensemble. Each fold returns a sigmoid probability, the probabilities are averaged, and the application applies a default decision threshold of `0.50`.

This model is intended for screening and triage only. It is not a confirmatory analytical method.

## Model Details

### Model Description

- **Model ID:** `cannabinoids-binary-sc-vs-nsc`
- **Developed by:** `[More Information Needed]`
- **Funded by [optional]:** `[More Information Needed]`
- **Shared by [optional]:** `[More Information Needed]`
- **Model type:** Supervised 1D CNN encoder plus an MLP binary classification head; five-fold ensemble at inference
- **Language(s) (NLP):** Not applicable (mass-spectrometry data)
- **License:** `[More Information Needed]`; no license is declared in `pyproject.toml` or elsewhere in the repository
- **Finetuned from model [optional]:** Shared SimCLR mass-spectrum encoder. The formal finetune script defaults to `embedding_pretrain/results_20260725_095428/best_model.pt`.
- **Repository snapshot used for this card:** 2026-09-27

The model weights are not stored in this repository snapshot. The repository ignores `*.pt` and `*.safetensors` files.

### Model Sources [optional]

- **Repository:** `E:\cannabinoids-master`
- **Paper [optional]:** `[More Information Needed]`
- **Demo [optional]:** Streamlit application in `app/streamlit/app.py`

### Architecture

| Component | Specification |
|---|---|
| Input | 561-dimensional vector |
| m/z range | 40-600 Da |
| Resolution | 1 Da bins |
| Preprocessing | Spectrum cleaning, TIC normalization, square-root transform |
| Encoder | Three 1D convolution blocks: `1 -> 64 -> 128 -> 256` channels |
| Encoder normalization | `GroupNorm` and `LayerNorm` |
| Encoder pooling | `AdaptiveAvgPool1d(1)` |
| Embedding | 256 dimensions, L2 normalized |
| Classifier head | `256 -> 128 -> 64 -> 1` with LayerNorm, ReLU, and dropout |
| Dropout | `0.3` after the first hidden layer; `0.2` after the second |
| Output | One logit converted to an SC probability with sigmoid |
| Ensemble | Arithmetic mean of probabilities from five fold models |

The binary encoder deliberately uses `GroupNorm` and `LayerNorm` rather than BatchNorm. The source comments describe this as a design choice to avoid BatchNorm-related information leakage.

### Input and Output Contract

- Supported uploaded formats in the application: MSP and MGF.
- The application currently sends the first parsed spectrum to the model.
- Raw input peaks are cleaned with `ms_entropy.clean_spectrum`.
- Peaks outside `m/z = 40-600` are dropped by the vectorizer.
- The model returns an SC probability in `[0, 1]`.
- `probability >= 0.50` is treated as positive/high risk in the pipeline.
- The application separately flags probabilities in `[0.3, 0.7]` for manual review when data sharing is authorized.
- The binary model itself has no out-of-distribution output.

## Uses

### Direct Use

- Screening a GC-EI mass spectrum for the synthetic cannabinoid versus non-synthetic-cannabinoid distinction.
- Batch triage of MSP/MGF spectra when used through the project pipeline.
- Laboratory decision support under trained human supervision.

### Downstream Use [optional]

In the application, a positive binary prediction triggers a separate FlashEntropySearch lookup against a known positive spectral library. That lookup is not part of this classifier and must be reported separately. A model prediction does not establish chemical identity.

### Out-of-Scope Use

Do not use this model as the sole basis for:

- forensic identification;
- regulatory, legal, clinical, or enforcement decisions;
- quantitation, purity assessment, or concentration estimation;
- structural elucidation, isomer discrimination, or salt/formulation identification;
- nine-class NPS classification;
- spectra acquired with incompatible ionization, instrument settings, or mass ranges;
- samples whose chemistry is not represented in the training distribution.

## Bias, Risks, and Limitations

- **Internal validation only:** No independent external validation cohort is present in the repository.
- **Dataset provenance is incomplete:** The original spectra are not versioned in the repository, so representativeness, collection conditions, and instrument diversity cannot be fully audited.
- **Class imbalance:** The evaluation artifacts imply 1,356 positive and 2,097 negative samples. Training uses a positive-class weight of `sqrt(1.8)` to reduce imbalance.
- **Threshold sensitivity:** At a `0.50` threshold, the stored test confusion matrix is `TN=304`, `FP=10`, `FN=11`, `TP=200`. A different threshold changes the false-positive/false-negative tradeoff.
- **Probability calibration is not established:** No calibration curve or reliability analysis is stored in the repository. A sigmoid value should not be interpreted as a calibrated probability of chemical identity.
- **No binary OOD guard:** Out-of-domain spectra can still receive a high or low score. The downstream `[0.3, 0.7]` review interval is heuristic, not a validated OOD detector.
- **Analogue bias:** Performance may degrade for new scaffolds, derivatives, mixtures, low-quality spectra, or data from a different instrument or collision/ionization regime.
- **Augmentation gap:** Training uses intensity jitter and Rayleigh baseline noise. These transformations may not reproduce every real-world spectral artifact.
- **Consequence asymmetry:** False negatives in screening can be especially harmful, while false positives can cause unnecessary confirmatory work.

### Recommendations

Use this model only as one component of an expert-reviewed workflow. Recalibrate the threshold on a local, representative validation set before deployment. Evaluate by instrument, acquisition method, chemical family, and relevant subpopulations. Record model version, threshold, and spectrum quality with every result. Confirm every actionable finding with orthogonal analytical methods. Retrain or abstain for spectra outside the validated domain.

## How to Get Started with the Model

The easiest way to run the application that serves this model is the prebuilt Docker image [`molyleaf/cannabinoids`](https://hub.docker.com/repository/docker/molyleaf/cannabinoids), which bundles the weights and the entropy-search index; Docker deployment is recommended in all cases (see the repository README). To use the weights directly, the repository expects five deployment artifacts named `official_ensemble_fold_1.safetensors` through `official_ensemble_fold_5.safetensors` in the selected model directory. The training code can also load compatible `.pt` checkpoints.

```python
from app.models.loader import get_ensemble_models, predict_risk_ensemble
from common.data_processor import (
    parse_msp,
    clean_spectrum,
    peaks_to_vector,
    preprocess_spectra,
)

# Replace with the directory containing the five deployment artifacts.
models = get_ensemble_models(models_dir=r"path/to/model_directory")

compound = parse_msp("sample.msp", min_peaks=1)[0]
vec = peaks_to_vector(clean_spectrum(compound["peaks"]))
vec_norm = preprocess_spectra(vec)

probability = float(predict_risk_ensemble(vec_norm, models)[0])
prediction = "SC / positive" if probability >= 0.50 else "NSC / negative"

print({"risk_probability": round(probability, 4), "prediction": prediction})
```

If only one compatible checkpoint is available, `get_ensemble_models(model_path="path/to/model.pt")` loads a single model and returns a one-model list. This fallback does not provide five-fold soft voting.

## Training Details

### Training Data

The finetune preprocessing script defaults to:

- Positive MSP: `finetune/data_source/阳性-含CanonicalSMILES-5类骨架(4).msp`
- Negative MSP: `finetune/data_source/阴性(4).msp`
- Processed dataset: `finetune/data_source/preprocessed_finetune.parquet`

The source data and processed Parquet file are not present in this repository snapshot. Counts below were reconstructed from the checked-in evaluation confusion matrices and should be treated as artifact-derived rather than as a complete dataset card.

| Split | NSC / negative | SC / positive | Total |
|---|---:|---:|---:|
| Train | 1,469 | 964 | 2,433 |
| Validation | 314 | 181 | 495 |
| Test | 314 | 211 | 525 |
| Total | 2,097 | 1,356 | 3,453 |

The dataset preparation code groups positive samples by SMILES before splitting the train, validation, and test partitions. This reduces leakage from repeated positive structures during those initial splits. The subsequent five-fold development-pool partition in `finetune/main.py` is index-shuffled rather than group-aware.

### Preprocessing [optional]

1. Parse MSP records and retain records with at least five peaks.
2. Clean peaks with `ms_entropy.clean_spectrum`.
3. Map peaks into 561 one-Dalton bins over `m/z = 40-600`.
4. Normalize each vector by total ion current.
5. Apply a square-root transform.

### Training Procedure

The binary head is finetuned on top of a shared SimCLR encoder in two phases.

| Parameter | Value |
|---|---|
| Number of folds | 5 |
| Ensemble inference | Soft voting across fold probabilities |
| Batch size | 128 |
| Warmup phase | 15 epochs with encoder frozen |
| Joint finetune phase | Up to 75 epochs with encoder unfrozen |
| Warmup optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Joint optimizer | AdamW; encoder learning rate `3e-5`; head learning rate `3e-4`; weight decay `1e-4` |
| Scheduler | `ReduceLROnPlateau`, factor `0.5`, patience `4`, minimum learning rate `1e-7` |
| Early stopping | Patience `15` |
| Loss | Focal Loss with `gamma=2.0` and positive weight `sqrt(1.8)` |
| Gradient clipping | Maximum norm `1.0` |
| Train augmentation | Intensity jitter `0.85-1.15` plus Rayleigh baseline noise |
| Evaluation/inference augmentation | None |
| Decision threshold | `0.50` |

The shared encoder was pretrained with SimCLR on binned mass spectra. The pretraining script records these defaults: two-GPU DDP, global batch size 2,048, base learning rate `0.030` scaled to `0.240`, 10 warmup epochs, up to 500 epochs, LARS weight decay `1e-6`, NT-Xent temperature `0.07`, and convergence patience `30`. The checked-in loss log ends at epoch 500 with training loss approximately `0.5602`.

After training, the five fold models are saved as `official_ensemble_fold_{1..5}.pt`. `finetune/convert_to_safetensors.py` converts the combined encoder and classifier state dictionaries into `.safetensors` artifacts for the application loader.

#### Speeds, Sizes, Times [optional]

Exact wall-clock training time, hardware hours, checkpoint sizes, and cloud cost are not recorded in the repository.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The stored evaluation artifacts under `finetune/scratch` report the split sizes shown above. The independent test set contains 525 spectra: 314 negative/NSC and 211 positive/SC.

#### Factors

No formal subgroup breakdown is stored by instrument, acquisition site, precursor charge, spectrum quality, or chemical scaffold.

#### Metrics

The repository computes accuracy, precision, recall, F1, specificity, and ROC-AUC. AUC values below were computed from the checked-in ROC CSV files using trapezoidal integration.

### Results

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,433 | 0.9823 | 0.9666 | 0.9896 | 0.9780 | 0.9775 | 0.9986 |
| Validation | 495 | 0.9778 | 0.9775 | 0.9613 | 0.9694 | 0.9873 | 0.9971 |
| Test | 525 | 0.9600 | 0.9524 | 0.9479 | 0.9501 | 0.9682 | 0.9872 |

Test confusion matrix at threshold `0.50`:

| | Predicted NSC | Predicted SC |
|---|---:|---:|
| Actual NSC | 304 | 10 |
| Actual SC | 11 | 200 |

These are internal results from recorded project artifacts. They were not rerun while creating this model card because the weights and raw datasets are absent from the repository snapshot.

#### Summary

The binary classifier reports strong internal discrimination, with test Accuracy `0.9600`, F1 `0.9501`, and ROC-AUC `0.9872`. External validity, calibration, and robustness remain unknown.

## Model Examination [optional]

No interpretability, saliency, embedding-probe, or calibration study is stored for this model. The project includes a separate Grad-CAM/heatmap area, but it is not linked here as evidence about this specific checkpoint.

## Environmental Impact

Carbon emissions cannot be estimated reliably because the repository does not record the exact hardware, training duration, cloud provider, region, or energy source.

- **Hardware Type:** Training script expects CUDA; shared pretraining comments specify two G100 GPUs. Inference supports CPU or CUDA.
- **Hours used:** `[More Information Needed]`
- **Cloud Provider:** `[More Information Needed]`
- **Compute Region:** `[More Information Needed]`
- **Carbon Emitted:** `[More Information Needed]`

Carbon emissions can be estimated with the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

## Technical Specifications [optional]

### Model Architecture and Objective

The objective is binary classification with a single logit and binary cross-entropy-based focal loss during training. During inference, sigmoid probabilities from five fold models are averaged before thresholding.

### Compute Infrastructure

The training path supports CUDA and falls back to CPU. The application can run inference on CPU or CUDA.

#### Hardware

`[More Information Needed]` for exact fine-tuning hardware and duration.

#### Software

Relevant project dependencies include Python `>=3.13`, PyTorch, NumPy, `ms-entropy>=1.5.2`, safetensors, and Streamlit. Training and evaluation scripts also use scikit-learn, pandas, pyarrow, matplotlib, and related scientific Python libraries.

## Citation [optional]

No paper or formal model citation is recorded.

**BibTeX:**

`[More Information Needed]`

**APA:**

`[More Information Needed]`

## Glossary [optional]

- **SC:** Synthetic cannabinoid, the positive class used by the application.
- **NSC:** Non-synthetic cannabinoid, the negative class used by the application.
- **GC-EI-MS:** Gas chromatography-electron ionization mass spectrometry.
- **m/z:** Mass-to-charge ratio.
- **TIC:** Total ion current.
- **SMILES:** Text representation of molecular structure.
- **Soft voting:** Averaging class probabilities from multiple models.
- **Focal Loss:** A loss that down-weights easy examples to focus learning on harder examples.
- **OOD:** Out of distribution.

## More Information [optional]

This card was derived from repository code and recorded artifacts on 2026-09-27. The model weights, original training spectra, processed Parquet data, and exact run metadata are not included in the repository snapshot. The card intentionally marks those items as missing rather than inferring unsupported values.

## Model Card Authors [optional]

Project maintainers; individual author names are not specified in the repository.

## Model Card Contact

`[More Information Needed]`
