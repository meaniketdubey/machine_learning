# KNN Sandbox

A K-Nearest Neighbors binary classification sandbox built on a synthetic Kaggle dataset. This project walks through the full supervised ML pipeline — from raw data loading and EDA through imputation, encoding, scaling, model training, hyperparameter tuning, and feature selection — with a focus on clarity and reproducibility.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Key Results](#key-results)
- [Setup & Usage](#setup--usage)
- [Requirements](#requirements)

---

## Overview

This notebook-based project demonstrates a complete KNN binary classification pipeline on a synthetic tabular dataset. It is designed as an educational sandbox for exploring:

- Exploratory data analysis (EDA) and missing value handling
- Categorical encoding and feature scaling (without data leakage)
- KNN model training and evaluation with multiple metrics
- Hyperparameter tuning optimised for F1-score and Recall
- Feature selection via Pearson correlation and Mutual Information

---

## Dataset

**Source:** [`iamaniketdubey/sandbox-data`](https://www.kaggle.com/datasets/iamaniketdubey/sandbox-data) (downloaded via `kagglehub`)

**File:** `synthetic_data.csv`

| Property | Detail |
|---|---|
| Rows | ~5 000 (synthetic) |
| Total columns | 24 |
| Numerical features | `num_feature_1` … `num_feature_18` (18 cols) |
| Categorical features | `cat_region`, `cat_segment`, `cat_flag` |
| Noise features | `noise_feature_1`, `noise_feature_2` |
| Target | Binary (0 / 1) |

**Missing values present in:** `num_feature_3` (499), `num_feature_7` (790), `noise_feature_1` (223), `cat_region` (283)

---

## Project Structure

```
knn-sandbox/
├── knn-sandbox.ipynb          # Main notebook (38 cells)
├── synthetic_data.csv         # Raw input dataset
├── synthetic_data_imputed.csv # Generated at runtime — post-imputation
├── synthetic_data_encoded.csv # Generated at runtime — post-encoding
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Workflow

The notebook is organised into the following stages:

| Stage | Description |
|---|---|
| 1. Environment Setup & Data Loading | Import libraries, download dataset via `kagglehub`, load CSV |
| 2. Exploratory Data Analysis | Column dtypes, missing value counts and percentages |
| 3. Missing Value Imputation | Median for numeric columns; `"Missing"` literal for categorical |
| 4. Save Imputed Data | Write `synthetic_data_imputed.csv` |
| 5. Categorical Encoding | `pd.get_dummies()` with `drop_first=True` → `synthetic_data_encoded.csv` |
| 6. Train / Test Split | 80 / 20 split, `stratify=y`, `random_state=42` |
| 7. Feature Scaling | `StandardScaler` fit on train only — no data leakage |
| 8. Baseline KNN (k=5) | Train, predict, full metrics report |
| 9. Hyperparameter Tuning — F1 | Grid search k=1…30, 5-fold CV on F1 → best k=11 |
| 10. Hyperparameter Tuning — Recall | Grid search k=1…30, 5-fold CV on Recall → best k=3 |
| 11. Feature Selection — Pearson | Drop features with \|r\| < 0.05 → 27 → 17 features |
| 12. Feature Selection — Mutual Info | `mutual_info_classif`, keep MI > 0.001 → 13 features |

---

## Key Results

| Configuration | Accuracy | Notes |
|---|---|---|
| Baseline KNN (k=5, 27 features) | ~92.55% | Precision ~92.53% |
| Tuned KNN k=11 (F1-optimised) | ~92.65% | Marginal accuracy gain |
| Tuned KNN k=3 (Recall-optimised) | — | Higher recall, lower precision trade-off |
| 17-feature correlation subset | ~91.75% | Drops 10 low-correlation features |
| 13-feature MI subset | dynamic | Drops features with MI ≤ 0.001 |

`noise_feature_1` and `noise_feature_2` are correctly identified as uninformative and removed during both feature selection stages.

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone <repo-url>
cd knn-sandbox
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Kaggle credentials

The notebook downloads the dataset automatically via `kagglehub`. You must have a valid Kaggle API token configured:

```bash
# Place kaggle.json in the default location
~/.kaggle/kaggle.json   # Linux / macOS
%USERPROFILE%\.kaggle\kaggle.json  # Windows
```

Alternatively, set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

### 4. Run the notebook

```bash
jupyter notebook knn-sandbox.ipynb
```

Or open it directly in [Kaggle Notebooks](https://www.kaggle.com/), where no additional credential setup is required.

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full pinned dependency list.

Core dependencies:

| Package | Purpose |
|---|---|
| `numpy` | Array operations, linear algebra |
| `pandas` | Data processing, CSV I/O |
| `scikit-learn` | KNN model, preprocessing, metrics, feature selection |
| `kagglehub` | Kaggle dataset download client |

---

## License

This project is released for educational and experimental purposes.