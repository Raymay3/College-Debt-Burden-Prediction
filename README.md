# College Debt Burden Prediction

## SIADS 696 - Milestone II

**Team 15**

- Dominick Devarti
- Gabriella Gass
- Vamsi Krishna Ramesh

## Project Overview

This project investigates student debt burden at the U.S. higher-education institution level using data from the **U.S. Department of Education College Scorecard**.

The project addresses two primary questions:

1. **Can an institution's student debt burden be predicted using institutional characteristics observed before student outcomes occur?**
2. **Do U.S. higher-education institutions form meaningful groups beyond the traditional public, private nonprofit, and for-profit categories?**

To answer these questions, the project combines supervised and unsupervised machine learning. The supervised analysis predicts institution-level debt burden from historical institutional characteristics, while the unsupervised analysis examines whether institutions form meaningful data-driven groups. The final stage integrates the two approaches to test whether information discovered through clustering improves debt-burden prediction.

---

## Data Source

The project uses the **College Scorecard bulk data** published by the U.S. Department of Education.

The fixed data snapshot used for this project contains:

- Institution-level College Scorecard files
- Historical annual institution-level files
- Field-of-study data
- OPEID/UNITID crosswalk information
- College Scorecard metadata and documentation

The College Scorecard download page reported that the data were last updated **June 10, 2026** when the project data were acquired.

Raw data are stored under:

```text
data/raw/
```

The repository stores the large raw CSV files in compressed `.csv.gz` format. Pandas can read these files directly without manually extracting them.

Raw source data should not be modified in place.

---

## Unit of Analysis

The project uses the **6-digit Office of Postsecondary Education identifier (`OPEID6`)** as the institution-level identifier.

College Scorecard data may contain multiple `UNITID` records associated with the same OPEID6 because institutions can have multiple branches. These records are therefore consolidated to produce one modeling observation per OPEID6.

Institution-level characteristics are aggregated using rules appropriate to each feature, including enrollment-weighted averages, sums, and representative-branch values.

---

## Target Variable

The primary supervised-learning outcome is an institution-level measure of **student debt burden**:

```text
debt_burden = log(GRAD_DEBT_MDN / MD_EARN_WNE_P10)
```

where:

- `GRAD_DEBT_MDN` = median cumulative student debt among completers
- `MD_EARN_WNE_P10` = median earnings of students working and not enrolled 10 years after entry

The logarithm of the debt-to-earnings ratio is used as the modeling target.

Institutions without the information necessary to construct a valid target are excluded from the final modeling sample.

---

## Leakage Prevention

A major design goal of this project is to determine whether debt burden can be predicted using information that would have been available **before the outcome occurred**.

Variables that directly reveal the target are therefore excluded from the predictor matrix.

At minimum, the two variables used to construct the primary outcome are explicitly excluded:

```text
GRAD_DEBT_MDN
MD_EARN_WNE_P10
```

`OPEID6` is retained only as an identifier and is not used as a model predictor.

Learned preprocessing operations such as imputation, scaling, encoding, PCA, and other fitted transformations are performed using training data only to prevent information leakage.

---

## Modeling Dataset

The completed data-preparation workflow produces:

- **3,296 institutions**
- **35 approved predictors**
- **1 debt-burden target**
- One observation per `OPEID6`

The predictors include institution-level characteristics and compact field-of-study features covering areas such as:

- Institution type and degree characteristics
- Enrollment
- Student demographic composition
- Part-time enrollment
- Pell Grant participation
- Federal loan participation
- Admissions characteristics
- Tuition
- Net price
- Cost of attendance
- Program breadth
- Credential-level offerings

### Train/Test Split

The modeling sample is divided into:

| Dataset | Institutions | Share |
|---|---:|---:|
| Training | 2,636 | 80% |
| Held-out Test | 660 | 20% |
| **Total** | **3,296** | **100%** |

The training set is used for model development, preprocessing, cross-validation, hyperparameter tuning, and supervised/unsupervised integration.

The held-out test set is reserved for final model evaluation and should not be used for model selection.

Processed datasets are stored under:

```text
data/processed/
```

including:

```text
modeling_dataset.csv
train_dataset.csv
test_dataset.csv
```

---

## Supervised Learning

The supervised-learning component evaluates whether pre-outcome institutional characteristics can predict later debt burden.

Planned model families include:

- Mean baseline
- Sector-based baseline
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest
- LightGBM

Model development is performed using cross-validation on the training data.

Primary evaluation metrics include:

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of Determination

Model interpretation and diagnostic analyses may include:

- Predicted vs. actual values
- Residual analysis
- Feature importance
- SHAP analysis
- Learning curves
- Error analysis across institution types

---

## Unsupervised Learning

The unsupervised-learning component investigates whether institutions form meaningful data-driven groups that differ from conventional institution categories.

The planned workflow includes:

1. Preprocessing the institution feature matrix
2. Principal Component Analysis (PCA)
3. K-Means clustering
4. Gaussian Mixture Models
5. Ward/agglomerative hierarchical clustering
6. Quantitative comparison of candidate cluster solutions
7. Cluster stability analysis
8. Comparison between discovered clusters and existing institution categories
9. Profiling and interpretation of the resulting institution groups

Candidate cluster structures are evaluated using measures such as:

- Inertia/elbow behavior
- Silhouette score
- Davies-Bouldin index
- Adjusted Rand Index (ARI)
- Cluster stability
- Interpretability

Debt burden is **not used to construct the clusters**. It is examined afterward to determine whether the discovered institution groups differ meaningfully in debt burden.

---

## Supervised + Unsupervised Integration

After the supervised and unsupervised analyses are developed independently, the project integrates the two approaches.

Useful information discovered through unsupervised learning, such as:

- Cluster membership
- Distances to cluster centers

may be incorporated into the supervised-learning pipeline.

The project then compares:

```text
Best Supervised Model
        vs.
Best Supervised Model + Unsupervised Features
```

using the same cross-validation framework.

PCA and clustering used to generate model features must be fitted within the corresponding training folds to avoid leakage.

This analysis tests whether the institutional structure discovered through unsupervised learning contains predictive information beyond the original institutional characteristics.

---

## Repository Structure

```text
College-Debt-Burden-Prediction/
│
├── data/
│   ├── raw/
│   │   └── College_Scorecard_Raw_Data_06032026/
│   └── processed/
│       ├── modeling_dataset.csv
│       ├── train_dataset.csv
│       └── test_dataset.csv
│
├── figures/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_supervised_learning.ipynb
│   ├── 03_unsupervised_learning.ipynb
│   ├── 04_integration.ipynb
│   └── 05_final_evaluation.ipynb
│
├── results/
│
├── src/
│
├── .gitignore
├── requirements.txt
└── README.md
```

Some directories and later notebooks may be populated as the project progresses.

---

## Notebook Workflow

### `01_data_preparation.ipynb`

Creates the common modeling dataset used throughout the project.

Major tasks include:

- Inventorying and documenting the College Scorecard source files
- Inspecting relevant data schemas
- Verifying target-variable definitions
- Constructing the debt-burden target
- Establishing OPEID6 as the unit of analysis
- Selecting historical pre-outcome predictors
- Aggregating branch-level institution records
- Constructing field-of-study features
- Identifying target leakage
- Examining missingness
- Performing initial exploratory data analysis
- Establishing the 80/20 train/test split
- Saving and validating the processed datasets

**Output:** common modeling, training, and held-out test datasets.

### `02_supervised_learning.ipynb`

Develops and compares supervised regression models.

Major tasks include:

- Baseline models
- Cross-validation
- Training-data preprocessing
- Regularized linear models
- Random Forest
- LightGBM
- Hyperparameter tuning
- Model comparison
- Feature interpretation
- Learning curves

**Output:** selected supervised-learning pipeline and cross-validation results.

### `03_unsupervised_learning.ipynb`

Examines the structure of institutions without using debt burden to form the groups.

Major tasks include:

- Unsupervised preprocessing
- PCA
- K-Means
- Gaussian Mixture Models
- Ward/hierarchical clustering
- Cluster evaluation
- Stability analysis
- Cluster profiling
- Unsupervised visualizations

**Output:** selected institution clustering/representation and potential cluster-derived features.

### `04_integration.ipynb`

Connects the supervised and unsupervised analyses.

The notebook tests whether institution structure discovered through unsupervised learning improves debt-burden prediction.

**Output:** comparison of the best supervised model with and without unsupervised-derived features.

### `05_final_evaluation.ipynb`

Performs final evaluation after the modeling decisions have been made using training-data cross-validation.

Major tasks include:

- Final evaluation on the held-out test set
- Final model comparisons
- Sensitivity and robustness analyses
- Examination of model failures
- Production of final tables and figures

**Output:** final project results used for interpretation and reporting.

---

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

The project uses Python and Jupyter notebooks.

Core dependencies include:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- seaborn
- LightGBM
- SHAP
- Jupyter

Additional dependencies may be added as the modeling notebooks are developed.

---

## Reproducing the Analysis

The notebooks are intended to be run in numerical order:

```text
01_data_preparation.ipynb
        ↓
02_supervised_learning.ipynb
03_unsupervised_learning.ipynb
        ↓
04_integration.ipynb
        ↓
05_final_evaluation.ipynb
```

Notebooks 2 and 3 can be developed in parallel after Notebook 1 has produced the common modeling dataset.

For reproducibility:

1. Clone the repository.
2. Install the packages listed in `requirements.txt`.
3. Confirm that the College Scorecard raw data are available under `data/raw/`.
4. Run `01_data_preparation.ipynb` to reproduce the processed datasets.
5. Run the supervised and unsupervised notebooks.
6. Run the integration notebook after the supervised and unsupervised approaches have been selected.
7. Run the final-evaluation notebook last.

The held-out test data should remain untouched during model development.

---

## Data and Project Limitations

The analysis is conducted at the **institution level**, not the individual student level.

Accordingly, the results should not be interpreted as predictions about individual borrowers.

Additional limitations include:

- College Scorecard outcomes represent institution-level aggregates.
- Some observations are privacy-suppressed.
- Missingness may be systematic rather than random.
- Debt and earnings measures may represent pooled cohorts.
- Historical relationships may not generalize to future policy environments or future student cohorts.
- Institutional characteristics may describe associations with debt burden without establishing causal effects.

---

## Technologies

The project primarily uses:

- **Python**
- **pandas**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **seaborn**
- **LightGBM**
- **SHAP**
- **Jupyter Notebook**
- **Git / GitHub**

---

## Course

This project was developed for **SIADS 696 - Milestone II** in the University of Michigan Master of Applied Data Science program.
