---
library_name: pytorch
tags:
  - mass-spectrometry
  - gc-ei-ms
  - new-psychoactive-substances
  - multi-class-classification
  - forensic-screening
---

# Model Card for NPS Multi-Class Classifier

## Summary

This model is a PyTorch nine-class classifier for GC-EI mass spectra. It assigns a binned spectrum to one of nine broad new psychoactive substance (NPS) categories:

1. Fentanyls
2. Cathinones
3. Synthetic Cannabinoids
4. Arylcyclohexylamines
5. Benzodiazepines
6. Nitazenes
7. Opiates
8. Phenethylamines
9. Tryptamines

The model produces nine logits, which are converted to class probabilities with softmax. The application treats a maximum confidence below `0.98` as out-of-distribution (OOD) and does not trigger library search in that case. That OOD rule is application logic; it is not learned or calibrated by the classifier itself.

The training notebook selected Fold 2 as the final checkpoint because it had the highest test-set accuracy. This selection process means the headline Fold 2 test result is optimistic compared with a prospectively chosen model.

## Model Details

### Model Description

- **Model ID:** `cannabinoids-multi-class-nps`
- **Developed by:** `[More Information Needed]`
- **Funded by [optional]:** `[More Information Needed]`
- **Shared by [optional]:** `[More Information Needed]`
- **Model type:** Supervised 1D CNN encoder plus an MLP nine-class classification head
- **Language(s) (NLP):** Not applicable (mass-spectrometry data)
- **License:** `[More Information Needed]`; no license is declared in `pyproject.toml` or elsewhere in the repository
- **Finetuned from model [optional]:** A shared SimCLR mass-spectrum encoder. The notebook run recorded `D:\DL\cann\建模\best_model_0725.pt`; matching pretraining code is under `embedding_pretrain/`.
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
| Training preprocessing | Minimum five peaks, minimum peak intensity 10, TIC normalization, square-root transform |
| Encoder | Three 1D convolution blocks: `1 -> 64 -> 128 -> 256` channels |
| Encoder normalization | `BatchNorm1d` |
| Encoder pooling | `AdaptiveAvgPool1d(1)` |
| Embedding | 256 dimensions, L2 normalized |
| Classifier head | `256 -> 256 -> 128 -> 9` |
| Head normalization | `BatchNorm1d` |
| Dropout | `0.3` after the first hidden layer; `0.2` after the second |
| Output | Nine logits followed by softmax |
| Training configuration | Encoder frozen; 100,617 trainable head parameters |

### Class Labels and Normalization

The application uses the following normalized class names:

```text
Fentanyls
Cathinones
Synthetic Cannabinoids
Arylcyclohexylamines
Benzodiazepines
Nitazenes
Opiates
Phenethylamines
Tryptamines
```

The training notebook displayed Chinese labels for several categories. `app/models/loader.py` maps aliases such as `芬太尼`, `卡西酮`, and `大麻素` to the normalized names above.

### Input and Output Contract

- Supported uploaded formats in the application: MSP and MGF.
- The application currently sends the first parsed spectrum to the model.
- The application cleans peaks with `ms_entropy.clean_spectrum` before vectorization.
- Peaks outside `m/z = 40-600` are dropped by the vectorizer.
- The model returns nine softmax probabilities and the argmax class.
- `max(probability) >= 0.98` is treated as a confident class prediction by the application.
- `max(probability) < 0.98` is treated as OOD and library search is skipped.
- The model itself always returns a class distribution; abstention is implemented by the surrounding pipeline.

### Deployment Artifact Discovery

`app/models/loader.py` searches for deployment artifacts in this order:

1. `best_multi_class_model_*.pt`
2. `best_multi_class_model_*.safetensors`
3. `*multi_class*.pt`
4. `*multi_class*.safetensors`

The training notebook saved `best_multi_class_model_<timestamp>.pt`, `best_multi_class_model_weights_<timestamp>.pt`, and `best_multi_class_model_latest.pt`.

## Uses

### Direct Use

- Broad-category screening of GC-EI mass spectra from the supported NPS database domain.
- Triage and distribution analysis when followed by expert review.
- Category-conditioned downstream retrieval in the application after the confidence threshold is passed.

### Downstream Use [optional]

The application uses the predicted category primarily as an inference result. For confident predictions, it may continue to a separate FlashEntropySearch known-library lookup. That lookup is not part of the nine-class model and does not validate the predicted category.

### Out-of-Scope Use

Do not use this model as the sole basis for:

- exact compound identification;
- isomer, salt, formulation, mixture, or regioisomer discrimination;
- purity, potency, or quantitative estimation;
- forensic, legal, clinical, regulatory, or enforcement decisions;
- detection of compounds outside the nine represented categories;
- use with non-GC-EI spectra or incompatible acquisition settings;
- interpretation of the softmax score as a calibrated probability of chemical identity.

## Bias, Risks, and Limitations

- **Internal validation only:** The repository contains no independent external validation cohort.
- **Class imbalance:** Category sizes range from 75 Opiates to 1,264 Synthetic Cannabinoids. Accuracy can therefore hide poor performance on small classes, so macro-F1 and per-class review are important.
- **Grouping implementation mismatch:** The notebook describes a split grouped by SMILES, but the code assigns `group_id = len(group_list)`, which is a unique group per compound/spectrum. It does not actually derive groups from SMILES. Shared structures or repeated analogues could therefore cross the train/test boundary if present in the source databases.
- **Optimistic model selection:** Fold 2 was selected by its test-set accuracy. The reported Fold 2 test metrics are therefore not an unbiased estimate of deployment performance.
- **Shared test set across folds:** All five fold models were evaluated on the same 758-spectrum holdout. This does not make the per-fold metrics invalid, but it increases the risk of overfitting model selection to that holdout.
- **No calibration study:** Softmax confidence may be overconfident or underconfident. The `0.98` OOD threshold is not shown to be calibrated or externally validated.
- **Training/inference preprocessing mismatch:** The notebook's multi-class training path did not call `ms_entropy.clean_spectrum`, while the application inference path does. The effect of this mismatch has not been quantified.
- **Category boundary ambiguity:** Broad NPS categories may overlap chemically or pharmacologically, and some compounds or mixtures may not fit one mutually exclusive class.
- **Database dependence:** Performance depends on the composition, naming conventions, and spectral quality of the nine source databases.
- **Domain shift:** Performance may degrade for new scaffolds, derivatives, mixtures, low-signal spectra, different instruments, or different collision/ionization conditions.
- **Consequence risk:** A confident wrong category can misdirect confirmatory analysis or downstream library search.

### Recommendations

Use the predicted category as a screening hint, not as an identification. Review per-class precision and recall, not only overall accuracy. Re-split data by canonical structure before making deployment claims, tune the OOD threshold on an independent calibration set, and validate on external instruments and laboratories. Consider class weighting, targeted data collection, and a dedicated OOD model where abstention is important. Always confirm actionable findings with orthogonal analytical methods.

## How to Get Started with the Model

Place a compatible checkpoint in the model directory or pass its path directly. The checkpoint should contain either `model_state_dict` with encoder/classifier weights, separate `encoder_state_dict` and `classifier_state_dict`, or a directly loadable state dictionary.

```python
from app.models.loader import predict_multi_class
from common.data_processor import (
    parse_msp,
    clean_spectrum,
    peaks_to_vector,
    preprocess_spectra,
)

compound = parse_msp("sample.msp", min_peaks=1)[0]
vec = peaks_to_vector(clean_spectrum(compound["peaks"]))
vec_norm = preprocess_spectra(vec)

result = predict_multi_class(
    vec_norm,
    model_path=r"path/to/best_multi_class_model_latest.pt",
)

predicted_category = result["pred_class"]
confidence = result["confidence"]
application_status = "OOD" if confidence < 0.98 else predicted_category

print({
    "category": predicted_category,
    "confidence": confidence,
    "application_status": application_status,
    "probabilities": result["probabilities"],
})
```

## Training Details

### Training Data

The notebook run loaded nine category databases from `D:\DL\cann\建模`:

| Class | Total compounds | Train | Test |
|---|---:|---:|---:|
| Fentanyls | 619 | 496 | 123 |
| Cathinones | 1,082 | 866 | 216 |
| Synthetic Cannabinoids | 1,264 | 1,012 | 252 |
| Arylcyclohexylamines | 94 | 76 | 18 |
| Benzodiazepines | 240 | 192 | 48 |
| Nitazenes | 129 | 104 | 25 |
| Opiates | 75 | 60 | 15 |
| Phenethylamines | 152 | 122 | 30 |
| Tryptamines | 158 | 127 | 31 |
| **Total** | **3,813** | **3,055** | **758** |

The recorded output reports 3,813 compounds and 3,813 spectra. The exact source database files and spectra are not included in the repository snapshot.

The first split reserves 20% of each class for the test set. The remaining 3,055 training spectra are divided into five folds with approximately 2,439-2,447 training spectra and 608-616 validation spectra per fold.

### Preprocessing [optional]

The notebook preprocessing path:

1. Parses MSP records.
2. Retains at least five peaks.
3. Retains only peaks with intensity at least 10 in that parser.
4. Maps peaks to 561 one-Dalton bins over `m/z = 40-600`.
5. Normalizes by total ion current.
6. Applies a square-root transform.

No dynamic spectrum augmentation is used in the recorded multi-class notebook run.

### Training Procedure

| Parameter | Value |
|---|---|
| Cross-validation | Stratified five-fold over the 80% training portion |
| Test evaluation | One shared 758-spectrum holdout |
| Batch size | 64 |
| Maximum epochs | 1,000 |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Loss | Cross-entropy |
| Scheduler | `ReduceLROnPlateau`, factor `0.5`, patience `5`, minimum learning rate `1e-6` |
| Early stopping | Patience `30` |
| Encoder state | Frozen during head training |
| Trainable parameters | 100,617 |
| Augmentation | None in the notebook run |
| Random seed | `42` for the recorded split routines |

The base encoder was initialized from a pretrained mass-spectrum encoder. The notebook loaded Fold 2 weights into the final model after selecting the fold with the highest test accuracy.

#### Speeds, Sizes, Times [optional]

Exact wall-clock training time, hardware hours, checkpoint sizes, and cloud cost are not recorded in the repository snapshot. The recorded run used CUDA.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The test set contains 758 spectra across the nine classes listed above. It is an internal holdout from the same source databases and is reused for all five fold models.

#### Factors

The recorded results include class-level distribution but do not provide validated subgroup results by instrument, acquisition site, spectrum quality, chemical scaffold, or external laboratory.

#### Metrics

- **Accuracy:** overall fraction classified correctly.
- **Macro-F1:** unweighted mean of per-class F1, giving every class equal importance despite class imbalance.
- **AUC:** one-vs-rest area under the ROC curve for multiclass discrimination.

### Results

Recorded per-fold test results:

| Fold | Test accuracy | Macro-F1 | AUC |
|---|---:|---:|---:|
| 1 | 97.36% | 0.9577 | 0.9984 |
| 2 | 98.42% | 0.9731 | 0.9988 |
| 3 | 97.63% | 0.9597 | 0.9987 |
| 4 | 98.02% | 0.9606 | 0.9988 |
| 5 | 97.36% | 0.9598 | 0.9993 |
| Mean +/- standard deviation | 97.76% +/- 0.41% | 0.9622 +/- 0.0055 | 0.9988 +/- 0.0003 |

Best recorded checkpoint:

| Metric | Fold 2 result |
|---|---:|
| Test accuracy | 98.42% |
| Test macro-F1 | 0.9731 |
| Test AUC | 0.9988 |

Fold 2 was selected because it had the highest test accuracy among the five folds. The result is therefore a selected maximum, not an unbiased estimate from a model chosen before seeing the holdout. The five-fold mean is more suitable for descriptive comparison but still comes from one internal test set.

The repository stores aggregate notebook output only. It does not contain the raw test predictions for recomputing confidence intervals or per-class precision/recall.

#### Summary

The recorded internal results are high: mean five-fold accuracy is 97.76%, mean macro-F1 is 0.9622, and mean AUC is 0.9988. The main scientific limitations are the class imbalance, the test-set-based fold selection, the ineffective SMILES grouping implementation, and the lack of external validation and calibration.

## Model Examination [optional]

No interpretability, representation-probing, per-class error analysis, or calibration study is stored in the repository snapshot.

## Environmental Impact

Carbon emissions cannot be estimated reliably because the repository does not record the exact hardware, training duration, cloud provider, region, or energy source.

- **Hardware Type:** CUDA GPU used in the recorded notebook; exact model not recorded.
- **Hours used:** `[More Information Needed]`
- **Cloud Provider:** `[More Information Needed]`
- **Compute Region:** `[More Information Needed]`
- **Carbon Emitted:** `[More Information Needed]`

Carbon emissions can be estimated with the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

## Technical Specifications [optional]

### Model Architecture and Objective

The objective is nine-class cross-entropy classification. The model uses a BatchNorm-based 1D CNN encoder that maps a 561-dimensional spectrum into a 256-dimensional L2-normalized embedding, followed by a three-layer MLP head that outputs nine logits.

### Compute Infrastructure

Training used CUDA if available. Inference supports CUDA or CPU. The application loads `.pt` checkpoints or safetensors state dictionaries.

#### Hardware

`[More Information Needed]` for exact training and evaluation hardware.

#### Software

Relevant project dependencies include Python `>=3.13`, PyTorch, NumPy, `ms-entropy>=1.5.2`, safetensors, and Streamlit. The training notebook also uses scikit-learn, pandas, matplotlib, seaborn, and Excel export libraries.

## Citation [optional]

No paper or formal model citation is recorded.

**BibTeX:**

`[More Information Needed]`

**APA:**

`[More Information Needed]`

## Glossary [optional]

- **NPS:** New psychoactive substance.
- **GC-EI-MS:** Gas chromatography-electron ionization mass spectrometry.
- **OOD:** Out of distribution; the application label for maximum softmax confidence below `0.98`.
- **Softmax:** A function that converts logits into a normalized probability distribution.
- **Macro-F1:** The unweighted average of class-level F1 scores.
- **AUC:** Area under the receiver operating characteristic curve.
- **SMILES:** Text representation of molecular structure.
- **TIC:** Total ion current.

## More Information [optional]

This card was derived from repository code and recorded notebook output on 2026-09-27. The model weights, source databases, processed tensors, and exact run metadata are not included in the repository snapshot. The card intentionally distinguishes recorded results from deployment claims and marks unavailable evidence as missing.

## Model Card Authors [optional]

Project maintainers; individual author names are not specified in the repository.

## Model Card Contact

`[More Information Needed]`
