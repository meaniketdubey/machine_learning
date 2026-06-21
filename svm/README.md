# Multiclass Credit Risk Classification with SVM

A synthetic credit risk classification project built end-to-end: data generation, EDA, feature engineering, multicollinearity diagnosis, and SVM model training/tuning — with every preprocessing decision made deliberately and validated against the data, not applied by default.

## Problem Statement

Classify loan applicants into three risk tiers — **Low_Risk**, **Medium_Risk**, **High_Risk** — based on 22 financial/demographic features, using a Support Vector Machine (SVM) classifier.

## Dataset

- **10,000 rows**, 22 numeric features + 1 categorical (`employment_type`) + target (`risk_category`)
- Generated synthetically using `sklearn.datasets.make_classification` (14 informative features, 4 redundant, 3 classes, 2 clusters per class)
- Deliberately injected real-world messiness for practice:
  - ~4% missing values in 3 columns (`income`, `savings_balance`, `employment_years`)
  - Outliers in `loan_amount` (30 rows inflated 15–25x)
  - Class imbalance: ~50% / 30% / 20% (Low / Medium / High)
  - Label noise (`flip_y=0.03`)

## EDA Findings

| Check | Finding |
|---|---|
| Missing values | ~4% in 3 columns, low overlap between columns → consistent with MCAR (Missing Completely At Random) |
| Class balance | 50/30/20 split — accuracy alone would be a misleading metric (predicting majority class gives ~49.5% "free" accuracy) |
| `loan_amount` | Heavy right-tail outliers confirmed via `.describe()` and boxplot (75th percentile ≈ 1.5, max ≈ 92.8) |
| `age` distribution | Non-normal, "folded" shape — caused by an `abs()` transform during data generation |
| Pairwise correlation | No single pair exceeded 0.6 — looked deceptively clean |
| **VIF (multicollinearity)** | Several features showed **infinite VIF** — exact linear dependency invisible to pairwise correlation, since `make_classification`'s redundant features are combinations of *multiple* informative features at once, not 1-to-1 copies |

**Key lesson:** pairwise correlation can miss multicollinearity that spans 3+ variables. VIF caught what the correlation heatmap didn't.

## Feature Engineering

1. **Multicollinearity removal** — iterative VIF elimination (threshold = 10), dropping the highest-VIF feature one at a time and recomputing, until all remaining features had VIF < 4.
   - Dropped: `income`, `debt_ratio`, `collateral_value`, `credit_history_years`, `credit_utilization`
   - Result: 22 → 17 numeric features, all well-conditioned (max VIF ≈ 3.8)
2. **Missing value imputation** — median imputation, justified by checking skew first (both columns had skew near 0, confirming the choice rather than assuming it).
3. **Outlier handling (`loan_amount`)** — capped at the 99th percentile (winsorizing), chosen over log-transform because the outliers were known to be artificially injected extreme values rather than organic skew, and because the column contained negative values (log transform isn't directly applicable without shifting).
4. **Categorical encoding (`employment_type`)** — one-hot encoding (not label encoding), because SVM is distance-based and label encoding would impose a false ordinal relationship/magnitude between unrelated categories (e.g., implying "Retired" is mathematically "further" from "Salaried" than "Unemployed" is).
5. **Target encoding (`risk_category`)** — ordinal label encoding (0/1/2), appropriate here since Low < Medium < High is a genuine real-world order.
6. **Train/test split** — 80/20, **stratified** on the target to preserve the 50/30/20 class balance in both splits.
7. **Feature scaling** — `StandardScaler`, fit **only on training data**, then applied to both train and test via `.transform()` — avoiding data leakage (fitting on the full dataset would let test-set statistics influence the scaler before evaluation).

## Model Training & Results

### Kernel comparison (default hyperparameters)

| Kernel | Accuracy |
|---|---|
| Linear | 0.7385 |
| **RBF** | **0.8940** |
| Poly | 0.8465 |

RBF substantially outperformed linear (+15.5 points), consistent with the data's generation structure (`n_clusters_per_class=2` means each class occupies multiple separate regions in feature space — not linearly separable by a single hyperplane).

### Hyperparameter tuning

- `GridSearchCV` over `C` and `gamma`, **5-fold cross-validation**
- Scored on **macro F1** (not accuracy), to avoid optimizing for the majority class
- Added `class_weight='balanced'` to directly penalize misclassifying the minority `High_Risk` class
- Best params: `C=1, gamma=0.1` (Best CV macro F1: 0.889)

### Baseline vs. Tuned Model

| Metric | Baseline (default RBF) | Tuned (balanced + grid search) |
|---|---|---|
| Accuracy | 0.894 | **0.9025** |
| Macro F1 | 0.88 | **0.89** |
| High_Risk recall | 0.80 | **0.85** |
| High_Risk → Low_Risk errors (costliest mistake) | 59 | **41** (-31%) |

The tuned model improved on every metric simultaneously — both better overall performance *and* better minority-class recall, rather than the accuracy/fairness tradeoff that's often expected. This was due to genuinely better hyperparameters (not just `'scale'` default gamma), combined with class balancing.

**Business framing:** in a real credit risk system, classifying a genuinely high-risk applicant as low-risk is the costliest type of error (approving a bad loan). The tuned model reduced this specific error from 59 to 41 cases — an 31% reduction — directly addressing the highest-stakes mistake rather than optimizing a generic metric.

## Key Takeaways

- **Correlation ≠ multicollinearity.** VIF caught dependency that pairwise correlation completely missed.
- **Encoding strategy must match the model.** Distance-based models like SVM require care with categorical encoding (one-hot for nominal, label encoding only for genuinely ordinal data).
- **Scaling order matters.** Fit scalers only on training data to avoid leakage.
- **Accuracy can be misleading under class imbalance.** Macro F1 and per-class recall told a more honest story, especially for the minority `High_Risk` class.
- **Kernel choice should match data geometry**, not be chosen by default — `rbf`'s advantage here was directly explainable by how the synthetic classes were generated (multi-cluster, non-linear structure).

## Tech Stack

`pandas`, `numpy`, `scikit-learn`, `statsmodels` (VIF), `matplotlib`, `seaborn`
