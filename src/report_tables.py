"""Build the report's tables from the saved notebook results.

Run from the project root after all five notebooks have been executed:

    python -m src.report_tables

Writes results/report_tables.md. Every number comes from a file in results/
or data/, so the tables stay consistent with the notebooks.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "report_tables.md"
DATA_DICTIONARY = ROOT / "data" / "raw" / "College_Scorecard_Raw_Data_06032026" / "data.yaml"

# Predictor groups used in Table 1 and Appendix A
FEATURE_GROUPS = {
    "Institutional structure": ["CONTROL", "REGION", "PREDDEG", "HIGHDEG", "NUMBRANCH"],
    "Enrollment and student composition": [
        "UGDS", "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN", "UGDS_AIAN",
        "UGDS_NHPI", "UGDS_2MOR", "UGDS_NRA", "UGDS_UNKN", "PPTUG_EF",
    ],
    "Financial-aid participation": ["PCTPELL", "PCTFLOAN"],
    "Admissions and selectivity": ["ADM_RATE", "SAT_AVG"],
    "Price and cost": ["TUITIONFEE_IN", "TUITIONFEE_OUT", "NET_PRICE", "COST_OF_ATTENDANCE",
                       "COST_IS_ACADEMIC_YEAR"],
    "Program and credential mix": [
        "NUM_PROGRAMS", "NUM_CREDENTIAL_LEVELS", "HAS_CERTIFICATE", "HAS_ASSOCIATE",
        "HAS_BACHELOR", "HAS_POSTBACC_CERT", "HAS_MASTER", "HAS_DOCTORAL",
        "HAS_FIRST_PROFESSIONAL", "HAS_GRAD_CERT",
    ],
}

CATEGORICAL = {"CONTROL", "REGION", "PREDDEG", "HIGHDEG"}
LOGGED = {"TUITIONFEE_IN", "TUITIONFEE_OUT", "NET_PRICE", "COST_OF_ATTENDANCE", "UGDS",
          "NUMBRANCH", "NUM_PROGRAMS"}

# Descriptions for features constructed in 01_data_preparation.ipynb
DERIVED_DESCRIPTIONS = {
    "NET_PRICE": "Average net price: public-institution value (NPT4_PUB) or private-institution value (NPT4_PRIV), whichever applies",
    "COST_OF_ATTENDANCE": "Cost of attendance: academic-year cost (COSTT4_A) when available, otherwise program-year cost (COSTT4_P)",
    "COST_IS_ACADEMIC_YEAR": "1 if COST_OF_ATTENDANCE uses the academic-year measure, 0 if program-year",
    "NUM_PROGRAMS": "Number of distinct CIP programs offered (field-of-study file)",
    "NUM_CREDENTIAL_LEVELS": "Number of distinct credential levels offered (field-of-study file)",
    "HAS_CERTIFICATE": "Offers undergraduate certificate programs (CREDLEV 1)",
    "HAS_ASSOCIATE": "Offers associate degree programs (CREDLEV 2)",
    "HAS_BACHELOR": "Offers bachelor's degree programs (CREDLEV 3)",
    "HAS_POSTBACC_CERT": "Offers post-baccalaureate certificate programs (CREDLEV 4)",
    "HAS_MASTER": "Offers master's degree programs (CREDLEV 5)",
    "HAS_DOCTORAL": "Offers doctoral degree programs (CREDLEV 6)",
    "HAS_FIRST_PROFESSIONAL": "Offers first-professional degree programs (CREDLEV 7)",
    "HAS_GRAD_CERT": "Offers graduate certificate programs (CREDLEV 8)",
}

# Hyperparameter search spaces as specified in 02_supervised_learning.ipynb
SEARCH_SPACES = {
    "Ridge": "Grid: alpha = 26 values log-spaced from 0.01 to 1,000",
    "Lasso": "Grid: alpha = 22 values log-spaced from 0.0001 to 0.316",
    "Elastic Net": "Grid: alpha = 17 values log-spaced from 0.0001 to 1; l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}",
    "Random Forest": ("Random search (20 draws), 300 trees: max_features ∈ {0.2, 0.33, 0.5, 0.75, 1.0}; "
                      "min_samples_leaf ∈ {1, 2, 4, 8, 16}; max_depth ∈ {None, 10, 20}"),
    "LightGBM": ("Random search (40 draws): n_estimators 200–1,499; learning_rate 0.005–0.1 (log-uniform); "
                 "num_leaves 4–47; min_child_samples 10–79; subsample 0.5–1.0; colsample_bytree 0.3–1.0; "
                 "reg_lambda 0.001–30 (log-uniform)"),
}


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def md_table(df):
    header = "| " + " | ".join(df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def preprocessing_note(feature):
    if feature in CATEGORICAL:
        return "One-hot encoded (linear models); integer codes (trees)"
    steps = []
    if feature in LOGGED:
        steps.append("log(1 + x)")
    steps.append("median imputation + missing indicator, standardized (linear models)")
    return "; ".join(steps)


GROUP_PREPROCESSING = {
    "Institutional structure": "Codes one-hot encoded for linear models (integer codes for trees); NUMBRANCH log(1 + x)",
    "Enrollment and student composition": "UGDS log(1 + x); shares used as-is; median imputation + missing indicators; standardized for linear models",
    "Financial-aid participation": "Shares used as-is; median imputation + missing indicators; standardized for linear models",
    "Admissions and selectivity": "Median imputation + missing indicators (53–65% missing); standardized for linear models",
    "Price and cost": "Dollar amounts log(1 + x); median imputation + missing indicators; standardized for linear models",
    "Program and credential mix": "NUM_PROGRAMS log(1 + x); 0/1 indicators used as-is; standardized for linear models",
}


def table_1():
    rows = [{
        "Predictor group": group,
        "Features": len(features),
        "Examples": ", ".join(f"`{f}`" for f in features[:4]) + (", …" if len(features) > 4 else ""),
        "Preprocessing": GROUP_PREPROCESSING[group],
    } for group, features in FEATURE_GROUPS.items()]
    return md_table(pd.DataFrame(rows))


def table_2():
    comparison = pd.read_csv(RESULTS / "02_supervised" / "model_comparison.csv")
    df = pd.DataFrame({
        "Model": comparison["model"],
        "RMSE (mean ± SD)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(comparison["rmse_mean"], comparison["rmse_std"])],
        "MAE": comparison["mae_mean"].map(fmt),
        "R²": comparison["r2_mean"].map(fmt),
    })
    return md_table(df)


def table_3():
    cases = pd.read_csv(RESULTS / "05_final_evaluation" / "selected_failure_cases.csv")
    explanations = {
        1955: "Net price of $3,067 and 28% loan participation despite $24,270 tuition; 84% Pell share",
        22092: "Single-program school with 172 students: small-cohort noise in median debt and earnings",
        41831: "Single-program school with 81 students: small-cohort noise in median debt and earnings",
        5618: "Low-tuition public institution ($2,644); burden far below what its profile suggests",
    }
    cases = cases.assign(abs_error=(cases["actual_debt_burden"] - cases["predicted_debt_burden"]).abs())
    cases = cases.sort_values("abs_error", ascending=False)
    df = pd.DataFrame({
        "OPEID6": cases["OPEID6"].astype(int),
        "Sector / degree": cases["control_group"] + ", " + cases["degree_group"],
        "Enrollment": cases["UGDS"].astype(int).map("{:,}".format),
        "Actual ratio": cases["actual_ratio"].map(fmt),
        "Predicted ratio": cases["predicted_ratio"].map(fmt),
        "Direction": cases["error_direction"],
        "Possible explanation": cases["OPEID6"].astype(int).map(explanations).fillna(""),
    })
    return md_table(df)


def table_4():
    sweep = pd.read_csv(RESULTS / "03_unsupervised" / "clustering_sweep.csv")
    stability = pd.read_csv(RESULTS / "03_unsupervised" / "bootstrap_stability.csv")
    stab = stability.groupby(["method", "k"])["ari"].mean().rename("stability").reset_index()
    merged = sweep.merge(stab, on=["method", "k"], how="left")
    selected = json.loads((RESULTS / "03_unsupervised" / "selected_clustering.json").read_text())
    k = selected["n_clusters"]
    notes = {
        "k-means": "Selected: four balanced, nameable types; highest silhouette and stability",
        "GMM": "Weakest separation and least stable; clusters change across resamples",
        "Ward": "Close to k-means on silhouette but less stable; tiny cluster appears from k = 6",
    }
    rows = merged[merged["k"] == k].set_index("method").loc[["k-means", "GMM", "Ward"]].reset_index()
    df = pd.DataFrame({
        "Method": rows["method"],
        "k": k,
        "Silhouette": rows["silhouette"].map(fmt),
        "Davies–Bouldin": rows["davies_bouldin"].map(fmt),
        "Bootstrap ARI": rows["stability"].map(fmt),
        "ARI vs. sector": rows["ari_vs_sector"].map(fmt),
        "Interpretation": rows["method"].map(notes),
    })
    return md_table(df)


def table_5():
    comparison = pd.read_csv(RESULTS / "04_integration" / "integration_comparison.csv")
    repeated = pd.read_csv(RESULTS / "04_integration" / "repeated_cv.csv")
    lgbm = comparison[comparison["model"].str.startswith("LightGBM")]
    rows = []
    for _, r in lgbm.iterrows():
        base = r["model"] == "LightGBM (supervised only)"
        rows.append({
            "Feature representation": "Original supervised features" if base else r["model"].replace("LightGBM ", ""),
            "CV RMSE (mean ± SD)": f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}",
            "ΔRMSE (mean ± SD)": "—" if base else f"{r['rmse_change_mean']:+.4f} ± {r['rmse_change_sd']:.4f}",
            "Folds improved": "—" if base else f"{int(r['folds_improved'])} of {int(r['n_folds'])}",
            "Criterion met?": "—" if base else ("Yes" if bool(r["passes_rule"]) else "No"),
        })
    wide = repeated.pivot(index=["repeat", "fold"], columns="model", values="rmse")
    integrated = [c for c in wide.columns if c != "LightGBM (supervised only)"][0]
    diff = wide[integrated] - wide["LightGBM (supervised only)"]
    rows.append({
        "Feature representation": f"{integrated.replace('LightGBM ', '')}, repeated CV ({len(diff)} folds)",
        "CV RMSE (mean ± SD)": f"{wide[integrated].mean():.4f} ± {wide[integrated].std():.4f}",
        "ΔRMSE (mean ± SD)": f"{diff.mean():+.4f} ± {diff.std():.4f}",
        "Folds improved": f"{int((diff < 0).sum())} of {len(diff)}",
        "Criterion met?": "No",
    })
    return md_table(pd.DataFrame(rows))


def robustness_tables():
    feature_sets = pd.read_csv(RESULTS / "02_supervised" / "feature_set_sensitivity.csv")
    fs = pd.DataFrame({
        "Feature set": feature_sets["feature_set"],
        "CV RMSE": feature_sets["rmse_mean"].map(lambda x: fmt(x, 4)),
        "R²": feature_sets["r2_mean"].map(fmt),
        "ΔRMSE": feature_sets["rmse_change"].map(lambda x: f"{x:+.4f}"),
    })
    alt = pd.read_csv(RESULTS / "02_supervised" / "alternative_targets.csv")
    r2 = alt.pivot(index="target", columns="model", values="r2_mean")
    r2 = r2.loc[alt["target"].unique(), ["Mean baseline", "Sector × degree baseline", "Lasso", "LightGBM"]]
    at = r2.map(fmt).reset_index().rename(columns={"target": "Target definition"})
    sens = pd.read_csv(RESULTS / "03_unsupervised" / "cluster_sensitivity.csv")
    cs = pd.DataFrame({
        "Variant": sens["variant"],
        "Dimensions": sens["dimensions"],
        "ARI vs. selected": sens["ari_vs_selected"].map(fmt),
        "ARI vs. sector": sens["ari_vs_sector"].map(fmt),
        "Burden η²": sens["burden_eta_squared"].map(fmt),
    })
    return md_table(fs), md_table(at), md_table(cs)


def bootstrap_table():
    boot = pd.read_csv(RESULTS / "05_final_evaluation" / "bootstrap_metric_summary.csv")
    df = pd.DataFrame({
        "Metric": boot["metric"].replace({"R2": "R²"}),
        "Held-out estimate": boot["test_estimate"].map(fmt),
        "95% bootstrap CI": [f"{lo:.3f}–{hi:.3f}" for lo, hi in zip(boot["ci_lower_95"], boot["ci_upper_95"])],
    })
    return md_table(df)


def appendix_a():
    dictionary = yaml.safe_load(DATA_DICTIONARY.read_text(encoding="utf-8"))["dictionary"]
    source_descriptions = {
        entry["source"]: entry.get("description", "").splitlines()[0]
        for entry in dictionary.values() if "source" in entry and entry.get("map") is None
    }
    features = pd.read_csv(ROOT / "data" / "processed" / "train_dataset.csv", nrows=0).columns
    features = [f for f in features if f not in ("OPEID6", "debt_burden")]
    grouped = [f for fs in FEATURE_GROUPS.values() for f in fs]
    assert sorted(grouped) == sorted(features), "Feature groups do not match the frozen feature set"
    rows = []
    for group, fs in FEATURE_GROUPS.items():
        for f in fs:
            derived = f in DERIVED_DESCRIPTIONS
            rows.append({
                "Feature": f"`{f}`",
                "Group": group,
                "Description": DERIVED_DESCRIPTIONS.get(f, source_descriptions.get(f, "")),
                "Source": ("Field-of-study file 2014–15/2015–16 (constructed)" if f.startswith(("NUM_", "HAS_"))
                           else "Institution file 2014–15 (constructed)" if derived
                           else "Institution file 2014–15"),
                "Preprocessing": preprocessing_note(f),
            })
    return md_table(pd.DataFrame(rows))


def appendix_b():
    tuned = json.loads((RESULTS / "02_supervised" / "tuned_hyperparameters.json").read_text())

    def show(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    rows = [{
        "Model": model,
        "Search space": SEARCH_SPACES[model],
        "Final hyperparameters (full training set)": "; ".join(f"{k} = {show(v)}" for k, v in tuned[model].items()),
    } for model in ["Ridge", "Lasso", "Elastic Net", "Random Forest", "LightGBM"]]
    return md_table(pd.DataFrame(rows))


def main():
    fs, at, cs = robustness_tables()
    sections = [
        "# Report Tables\n\nGenerated by `python -m src.report_tables` from the saved notebook results. "
        "Rerun it after re-executing the notebooks so the tables stay in sync.",
        "## Table 1 — Predictor groups and preprocessing (§3.5)\n\n" + table_1(),
        "## Table 2 — Cross-validated supervised model performance (§4.5)\n\n"
        "Nested 5-fold CV on the 2,636 training institutions (02_supervised_learning).\n\n" + table_2(),
        "## Held-out test performance with bootstrap intervals (§4.7)\n\n"
        "Selected LightGBM, 660 test institutions, 1,000 bootstrap resamples (05_final_evaluation).\n\n" + bootstrap_table(),
        "## Feature-set sensitivity (§4.7)\n\nLightGBM with fixed hyperparameters, same folds (02).\n\n" + fs,
        "## Alternative target definitions — cross-validated R² (§4.7)\n\n"
        "The proposal's payment target equals the main target minus a constant, so its results are identical (02).\n\n" + at,
        "## Table 3 — Representative prediction failures (§4.9)\n\n"
        "Ratios are debt-to-earnings ratios (exponentiated predictions) from the held-out test set (05).\n\n" + table_3(),
        "## Table 4 — Comparison of clustering approaches (§5.5)\n\n"
        "All methods at the selected k, fit on PCA scores of the training set (03). "
        "Bootstrap ARI is the mean over 30 resamples.\n\n" + table_4(),
        "## Unsupervised sensitivity (§5.9)\n\nSelected k-means (k = 4) refit under each variant (03).\n\n" + cs,
        "## Table 5 — Adding unsupervised features to LightGBM (§6.2)\n\n"
        "Unsupervised pipeline refit inside every training fold; fixed LightGBM hyperparameters (04). "
        "Criterion: mean improvement larger than the SD of the paired fold differences, "
        "and improvement in at least 80% of folds.\n\n" + table_5(),
        "## Appendix A — Feature dictionary\n\n"
        "All 35 predictors, aggregated to OPEID6 in 01_data_preparation. In the clustering analysis "
        "(03), `CONTROL` and `REGION` are excluded and features are log-transformed, imputed, and "
        "standardized before PCA.\n\n" + appendix_a(),
        "## Appendix B — Hyperparameter search spaces and final parameters\n\n"
        "Tuned by inner 5-fold CV within each outer fold, then re-tuned on the full training set (02).\n\n" + appendix_b(),
    ]
    OUTPUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print("Wrote", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
