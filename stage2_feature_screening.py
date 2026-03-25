"""Stage 2: exploratory feature screening."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from matplotlib_venn import venn2
    HAS_VENN = True
except ImportError:
    HAS_VENN = False

RANDOM_STATE = 42

RAW_FEATURES = [
    "tmax6", "tmax8", "tmax10", "rcbf30", "rcbf34", "rcbf38",
    "Culprit_Vessel", "Stenosis_Pct", "Multi_Stenosis", "Collateral",
    "age", "sex", "BMI", "SBP", "DBP", "NIHSS_In",
    "Hist_Stroke", "Hist_HTN", "Hist_DM", "Hist_CAD", "Hist_Smoke",
    "Tx_DAPT", "Intensive_Lipid_Eze", "Risk_Clo_Resist", "CYP2C19",
    "WBC", "Neutrophil", "Lymphocyte", "Monocyte", "HGB", "PLT",
    "RDW_CV", "MPV", "ESR", "Na", "eGFR", "Urea", "UA", "ALT", "AST",
    "ALB", "TBIL", "GGT", "A/G", "TG", "TC", "HDL", "LDL", "apoA1", "apoB",
    "Hcy", "hsCRP", "Glucose", "HbA1c", "Urine_Pro", "BNP",
    "PAgT_AA", "PAgT_ADP", "D_Dimer", "PT_INR", "APTT", "Fbg",
]

CATEGORICAL_COLUMNS = [
    "Culprit_Vessel", "Multi_Stenosis", "Collateral", "sex",
    "Hist_Stroke", "Hist_HTN", "Hist_DM", "Hist_CAD", "Hist_Smoke",
    "Tx_DAPT", "Intensive_Lipid_Eze", "Risk_Clo_Resist", "CYP2C19",
]

FINAL_FEATURES = [
    "age", "SBP", "NIHSS_In", "eGFR", "Glucose", "LDL",
    "Stenosis_Pct", "tmax6", "rcbf34",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--target", default="Label_Recur")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.data.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    df = read_table(args.data)
    df.columns = [str(c).strip() for c in df.columns]

    available_features = [feature for feature in RAW_FEATURES if feature in df.columns]
    if not available_features:
        raise ValueError("No predefined features were found in the input file.")
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' was not found in the input file.")

    X_raw = df[available_features].copy()
    y = df[args.target].values

    categorical = [column for column in CATEGORICAL_COLUMNS if column in X_raw.columns]
    numerical = [column for column in X_raw.columns if column not in categorical]

    X_imputed = X_raw.copy()
    if categorical:
        X_imputed[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(X_raw[categorical])
    if numerical:
        X_imputed[numerical] = IterativeImputer(max_iter=10, random_state=RANDOM_STATE).fit_transform(
            X_raw[numerical]
        )

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imputed), columns=X_imputed.columns)

    cv_lasso = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lasso = LogisticRegressionCV(
        cv=cv_lasso,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    lasso.fit(X_scaled, y)
    coefficients = pd.Series(np.abs(lasso.coef_[0]), index=X_scaled.columns)
    lasso_selected = coefficients[coefficients > 0].sort_values(ascending=False).index.tolist()

    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rfe = RFE(estimator=rf, n_features_to_select=15, step=1)
    rfe.fit(X_imputed, y)
    rfe_selected = X_imputed.columns[rfe.support_].tolist()

    rf.fit(X_imputed, y)
    rf_importances = pd.Series(rf.feature_importances_, index=X_imputed.columns)

    set_lasso = set(lasso_selected)
    set_rfe = set(rfe_selected)
    tier_a = list(set_lasso & set_rfe)
    tier_b = list(set_rfe - set_lasso)
    tier_c = list(set_lasso - set_rfe)

    final_features_in_data = [feature for feature in FINAL_FEATURES if feature in rf_importances.index]
    if not final_features_in_data:
        raise ValueError("None of the final features were found in the input file.")
    final_importance = rf_importances[final_features_in_data].sort_values()

    color_tier_a = "#E63946"
    color_tier_b = "#F4A261"
    color_tier_c = "#457B9D"

    if HAS_VENN:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        venn = venn2(
            subsets=(len(set_lasso - set_rfe), len(set_rfe - set_lasso), len(set_lasso & set_rfe)),
            set_labels=("LASSO", "RF-RFE"),
            ax=ax1,
        )
        for patch_id, color in zip(["10", "01", "11"], [color_tier_c, color_tier_b, color_tier_a]):
            patch = venn.get_patch_by_id(patch_id)
            if patch is not None:
                patch.set_color(color)
                patch.set_alpha(0.65 if patch_id != "11" else 0.55)
        ax1.set_title("Feature Selection Overlap", fontsize=14)
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    colors = [
        color_tier_a
        if feature in tier_a
        else color_tier_b
        if feature in tier_b
        else color_tier_c
        if feature in tier_c
        else "#999999"
        for feature in final_importance.index
    ]
    final_importance.plot(kind="barh", color=colors, ax=ax2)
    ax2.set_title("Importance of Final Predictors", fontsize=14)
    ax2.set_xlabel("Random Forest Importance")

    plt.tight_layout()
    plt.savefig(
        output_dir / "Supplementary_Figure_S1.tif",
        format="tiff",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


if __name__ == "__main__":
    main()
