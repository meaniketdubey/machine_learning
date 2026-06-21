# Logistic Regression — Jupyter Notebook

A hands-on implementation of **Logistic Regression** using Python and scikit-learn, presented as an interactive Jupyter Notebook. This project walks through the full machine-learning pipeline — from data exploration and preprocessing to model training, evaluation, and visualisation.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [License](#license)

---

## Overview

Logistic Regression is a fundamental **supervised learning** algorithm used for **binary (and multi-class) classification** tasks. Despite its name, it is a classification algorithm — not a regression one. It models the probability that a given input belongs to a particular class by applying the **sigmoid function** to a linear combination of input features, squashing the output to a value between 0 and 1.

This notebook demonstrates:

- How logistic regression works mathematically and intuitively
- How to prepare and explore a real-world dataset
- How to train a logistic regression model with scikit-learn
- How to evaluate and interpret model performance

---

## Key Concepts

| Concept | Description |
|---|---|
| **Sigmoid Function** | Maps any real-valued number to a probability in the range (0, 1): σ(z) = 1 / (1 + e⁻ᶻ) |
| **Decision Boundary** | The threshold (default 0.5) that separates predicted classes |
| **Cost Function** | Log-loss (binary cross-entropy) measures how well the model fits the training data |
| **Gradient Descent** | Optimisation algorithm used to minimise the cost function by iteratively updating model weights |
| **Binary Classification** | Predicting one of two possible outcomes (e.g. 0 or 1, Yes or No, Malignant or Benign) |
| **Regularisation** | L1 (Lasso) and L2 (Ridge) penalties to prevent overfitting |

---

## Dataset

> **Default suggestion:** This notebook uses a well-known classification dataset such as the **Breast Cancer Wisconsin Dataset**, **Titanic Survival Dataset**, or **Iris Dataset** (binary subset). Replace this section with the actual dataset details once confirmed.

| Property | Details |
|---|---|
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/) / [Kaggle](https://www.kaggle.com/) / scikit-learn built-ins |
| **Features** | Numerical and/or categorical input variables |
| **Target** | Binary class label (0 / 1) |
| **Size** | Varies by dataset |

If you would like to use the built-in scikit-learn dataset, it can be loaded directly in the notebook:

```python
from sklearn.datasets import load_breast_cancer   # or load_iris, etc.
data = load_breast_cancer()
```

---

## Project Structure

```
K-Nearest/
│
├── logestic-regression.ipynb   # Main Jupyter Notebook
├── README.md                   # Project documentation (this file)
│
├── data/                       # (Optional) Raw or processed dataset files
│   └── dataset.csv
│
└── images/                     # (Optional) Exported plots and figures
    └── confusion_matrix.png
```

---

## Prerequisites

Ensure you have the following installed before running the notebook:

- **Python** 3.8 or higher
- **Jupyter Notebook** or **JupyterLab**

### Python Libraries

| Library | Purpose |
|---|---|
| `numpy` | Numerical computations and array operations |
| `pandas` | Data loading, manipulation, and analysis |
| `matplotlib` | Base plotting and visualisation |
| `seaborn` | Statistical data visualisation |
| `scikit-learn` | Model training, preprocessing, and evaluation |

---

## Installation

1. **Clone or download this repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name/K-Nearest
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

   Or, if a `requirements.txt` is provided:

   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

1. Start Jupyter Notebook from the project directory:

   ```bash
   jupyter notebook
   ```

2. In the browser tab that opens, navigate to and click on:

   ```
   logestic-regression.ipynb
   ```

3. Run all cells sequentially using **Cell → Run All**, or step through them individually with **Shift + Enter**.

---

## Model Evaluation

The notebook evaluates the trained logistic regression model using the following metrics:

- **Accuracy** — Overall percentage of correct predictions
- **Confusion Matrix** — Breakdown of True Positives, True Negatives, False Positives, and False Negatives
- **Precision** — Of all predicted positives, how many were actually positive?
- **Recall (Sensitivity)** — Of all actual positives, how many did the model correctly identify?
- **F1 Score** — Harmonic mean of Precision and Recall; useful for imbalanced datasets
- **ROC-AUC Score** — Area under the Receiver Operating Characteristic curve; measures discrimination ability

```python
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
```

---

## Results

> Results will vary depending on the dataset used and the train/test split applied.

| Metric | Score |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1 Score | — |
| ROC-AUC | — |

Visual outputs included in the notebook:

- 📊 Feature correlation heatmap
- 📈 ROC Curve
- 🟦 Confusion Matrix heatmap
- 🔵 Decision boundary plot (for 2-feature subsets)

---

## License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute this code for educational and personal projects.

---

> **Note:** The notebook filename `logestic-regression.ipynb` preserves the original spelling as created in this project.