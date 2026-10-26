# College Debt Burden Prediction

**SIADS 696 — Milestone II | University of Michigan, Master of Applied Data Science**

**Team 15:** Dominick Devarti · Gabriella Gass · Vamsi Krishna Ramesh

## Project Overview

Can historical characteristics of U.S. colleges predict student debt burden? And can unsupervised learning identify institution types that provide additional predictive information?

Using U.S. Department of Education **College Scorecard** data, this project combines supervised regression, principal component analysis (PCA), clustering, and leakage-aware model integration to examine these questions at the **institution level**. The analysis includes **3,296 institutions**, **35 predictors**, and a held-out test set of **660 institutions**.

**Main result:** A LightGBM regression model achieved a held-out **RMSE of 0.2809**, **MAE of 0.1922**, and **R² of 0.5825** on log debt burden. Clustering revealed four interpretable institution types, but cluster-derived features did **not** provide a reliable improvement over the supervised model alone.

## Key Findings

### Supervised prediction

The project compared mean and sector-based baselines, Ridge, Lasso, Elastic Net, Random Forest, and LightGBM. Models were developed using training-only cross-validation and evaluated using RMSE (primary), MAE, and R².

**LightGBM** was selected based on cross-validation. Its final performance on the untouched 20% test set was:

| Held-out metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| RMSE | 0.2809 | 0.2530–0.3069 |
| MAE | 0.1922 | 0.1763–0.2064 |
| R² | 0.5825 | 0.5163–0.6418 |

The bootstrap intervals resample held-out institutions with the trained model fixed; they do not capture model-training or future-cohort uncertainty.

**Robustness checks** (training-set cross-validation):

- **Feature set:** removing the strongest predictor, federal-loan participation (`PCTFLOAN`), raises RMSE by only 0.004 (R² 0.583 → 0.571), so the results do not depend on the feature most closely related to the debt outcome. Removing sector or the sparse admissions variables has a similarly small effect.
- **Target definition:** the proposal's alternative target, annual payment ÷ earnings, turned out to be the main target shifted by a constant (the Scorecard computes payments with one fixed amortization factor), so it gives identical results. Under two definitions that genuinely differ, all-borrower debt ÷ 10-year earnings and completer debt ÷ 6-year earnings, LightGBM remains the best model (R² 0.57–0.59).

### Institutional clusters

The unsupervised workflow compared **k-means**, **Gaussian mixture models**, and **Ward agglomerative clustering** after feature preprocessing and PCA. A stable **four-cluster k-means** solution was selected, describing:

1. Certificate-focused schools
2. Four-year teaching colleges
3. Doctoral universities
4. Community colleges

These groups partially overlap with traditional public, private nonprofit, and private for-profit categories, but are not identical to them. Debt burden was excluded from cluster formation and examined only afterward.

### Supervised–unsupervised integration

The integration notebook tested cluster membership, distances to cluster centroids, and PCA-based representations as additional supervised features. Preprocessing and clustering were refitted **within cross-validation training folds** to avoid leakage.

Although some integrated variants showed very small improvements in average validation RMSE, **none met the project's predefined improvement criterion**. The final model therefore remained the **supervised-only LightGBM pipeline**. The cluster features appear largely redundant with patterns LightGBM can already learn from the original institutional predictors.

### Diagnostics and limitations

Prediction error was not uniform across institutions. The held-out analysis found larger errors for **private for-profit**, **certificate-focused**, and **smaller-enrollment** institutions. Permutation importance highlighted federal-loan participation, degree-level indicators, and tuition-related features, among others. These are **predictive associations, not causal effects**; demographic feature importance should not be interpreted as a causal claim about student groups.

The outcome and predictions are **institution-level aggregates**, not individual borrower outcomes. Missingness, privacy suppression, historical cohort alignment, and changes in future economic or policy conditions limit generalizability.

## Data and Target

**Source:** [U.S. Department of Education College Scorecard](https://collegescorecard.ed.gov/data/), using a bulk-data snapshot available during the project (download page updated June 10, 2026).

The unit of analysis is the **six-digit Office of Postsecondary Education identifier (`OPEID6`)**. Multiple campus records sharing an OPEID6 are consolidated using feature-appropriate aggregation rules, including sums, enrollment-weighted averages, and representative-branch values.

The target is the natural logarithm of the institution-level debt-to-earnings ratio:

```text
debt_burden = log(GRAD_DEBT_MDN / MD_EARN_WNE_P10)
```

- `GRAD_DEBT_MDN`: median debt among students who completed their programs.
- `MD_EARN_WNE_P10`: median earnings of working, non-enrolled students ten years after entry.

Predictors use historical institution characteristics, including sector, enrollment, student composition, aid participation, tuition, cost, admissions, and program offerings. The target components, target itself, and identifier are **excluded from model features**. Fitted transformations are learned on training data only.

The full **leakage exclusion list** is saved in [`data/leakage_exclusion_list.csv`](data/leakage_exclusion_list.csv). Built from the Scorecard data dictionary, it excludes every debt, earnings, repayment/default, and completion variable (2,739 variables) and records the loan-participation rates that were reviewed and allowed, such as `PCTFLOAN`, with the reason.

### Modeling sample

| Dataset | Institutions | Share |
|---|---:|---:|
| Training | 2,636 | 80% |
| Held-out test | 660 | 20% |
| **Total** | **3,296** | **100%** |

The held-out test set was reserved for the final frozen-model evaluation, not model selection.

## Repository Structure

```text
College-Debt-Burden-Prediction/
├── data/
│   ├── raw/                         # College Scorecard bulk snapshot (CSVs stored as .csv.gz)
│   ├── leakage_exclusion_list.csv   # Variables excluded as outcome leakage
│   └── processed/
│       ├── modeling_dataset.csv
│       ├── train_dataset.csv
│       └── test_dataset.csv
├── figures/
│   ├── 01_data_preparation/
│   ├── 02_supervised/
│   ├── 03_unsupervised/
│   ├── 04_integration/              # Generated by integration notebook
│   └── 05_final_evaluation/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_supervised_learning.ipynb
│   ├── 03_unsupervised_learning.ipynb
│   ├── 04_integration.ipynb
│   └── 05_final_evaluation.ipynb
├── results/
│   ├── 02_supervised/
│   ├── 03_unsupervised/
│   ├── 04_integration/
│   ├── 05_final_evaluation/
│   └── report_tables.md         # Report-ready tables built from the results
├── src/
│   ├── __init__.py
│   ├── integration.py
│   ├── report_tables.py         # python -m src.report_tables
│   └── unsupervised.py
├── .gitignore
├── requirements.txt
└── README.md
```

Generated artifacts include model-comparison tables, clustering diagnostics, integration decisions, held-out predictions, subgroup metrics, bootstrap intervals, and figures. The raw CSVs are stored gzip-compressed (`.csv.gz`) to stay within GitHub's file-size limit; pandas reads them directly.

## Notebook Workflow

| Notebook | Purpose |
|---|---|
| **01 — Data Preparation** | Read and document College Scorecard files, construct the target, aggregate by OPEID6, build features, and freeze the train/test split. |
| **02 — Supervised Learning** | Compare baselines and regression models, tune hyperparameters, select LightGBM, and analyze training-fold diagnostics. |
| **03 — Unsupervised Learning** | Preprocess training-only features, perform PCA, compare clustering algorithms, assess stability, and profile institution types. |
| **04 — Integration** | Evaluate leakage-safe cluster-derived features within supervised cross-validation and decide whether to adopt them. |
| **05 — Final Evaluation** | Evaluate the frozen selected model once on the held-out test set, examine subgroup errors and failure cases, and estimate bootstrap uncertainty. |

## Setup and Reproduction

1. Clone this repository and open it in a Python/Jupyter environment. The analysis was run with **Python 3.13**.
2. Install the pinned dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. No separate data download is needed. The College Scorecard snapshot used by the project is included in the repository under `data/raw/College_Scorecard_Raw_Data_06032026/`, with each CSV stored as `.csv.gz`. The notebooks read the compressed files directly.
4. Open Jupyter from the `notebooks/` directory (the notebooks use `Path("..")` to locate the project root).
5. Execute notebooks in order: **01 → 02 and 03 → 04 → 05**. Notebooks 02 and 03 can run independently after Notebook 01. Notebook 02 takes about five minutes because of nested cross-validation.

If you only need to inspect the prepared data and saved results, start with `data/processed/`, `results/`, and `figures/`.

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, SciPy, Matplotlib, seaborn, SHAP, PCA, UMAP, k-means, Gaussian mixtures, Ward clustering, Jupyter, and Git/GitHub.

## Interpretation

This project is a predictive and exploratory analysis, **not a causal study**. Its findings should not be used to infer the debt burden of a particular student or to attribute institutional outcomes to any individual demographic characteristic.

---

*Developed for SIADS 696 — Milestone II, University of Michigan Master of Applied Data Science program.*