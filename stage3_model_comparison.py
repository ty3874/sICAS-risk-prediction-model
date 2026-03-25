"""Stage 3: nested cross-validation model comparison."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

IMPUTER_SEED = 42
INNER_CV_SEED = 42
OUTER_CV_SEED = 1357
SEARCH_SEED = 42

FINAL_FEATURES = [
    "age", "SBP", "NIHSS_In", "eGFR", "Glucose", "LDL",
    "Stenosis_Pct", "tmax6", "rcbf34",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def get_model_spaces(class_ratio: float) -> dict[str, tuple[Pipeline, dict]]:
    pipe_lr = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=IMPUTER_SEED)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEARCH_SEED)),
    ])
    param_lr = {"clf__C": uniform(0.001, 10), "clf__penalty": ["l2"]}

    pipe_svm = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=IMPUTER_SEED)),
        ("scaler", StandardScaler()),
        ("clf", SVC(class_weight="balanced", probability=True, kernel="rbf", random_state=SEARCH_SEED)),
    ])
    param_svm = {"clf__C": uniform(0.1, 15), "clf__gamma": ["scale", 0.001, 0.01, 0.05, 0.1]}

    pipe_rf = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=IMPUTER_SEED)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEARCH_SEED)),
    ])
    param_rf = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__min_samples_leaf": [2, 4, 6],
    }

    pipe_xgb = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=IMPUTER_SEED)),
        (
            "clf",
            XGBClassifier(
                scale_pos_weight=class_ratio,
                eval_metric="logloss",
                random_state=SEARCH_SEED,
            ),
        ),
    ])
    param_xgb = {
        "clf__n_estimators": [50, 100, 150, 200],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    }

    return {
        "Logistic Regression": (pipe_lr, param_lr),
        "SVM": (pipe_svm, param_svm),
        "Random Forest": (pipe_rf, param_rf),
        "XGBoost": (pipe_xgb, param_xgb),
    }


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
    df.columns = [str(column).strip() for column in df.columns]

    missing_features = [feature for feature in FINAL_FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' was not found in the input file.")

    X = df[FINAL_FEATURES].copy()
    y = df[args.target].values.ravel()

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_CV_SEED)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INNER_CV_SEED)

    mean_fpr = np.linspace(0, 1, 100)
    colors = {
        "Voting Ensemble (LR + SVM)": "#E63946",
        "SVM": "#2A9D8F",
        "Logistic Regression": "#457B9D",
        "Random Forest": "#F4A261",
        "XGBoost": "#9467BD",
    }
    roc_storage = {name: {"tprs": [], "aucs": []} for name in colors}

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        positive_count = np.sum(y_train == 1)
        negative_count = np.sum(y_train == 0)
        class_ratio = float(negative_count) / positive_count if positive_count > 0 else 1.0

        model_spaces = get_model_spaces(class_ratio)
        best_models: dict[str, Pipeline] = {}

        for model_name, (estimator, param_space) in model_spaces.items():
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_space,
                n_iter=30,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=SEARCH_SEED,
            )
            search.fit(X_train, y_train)
            best_models[model_name] = search.best_estimator_

        for model_name in ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]:
            model = best_models[model_name]
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
            interpolated_tpr[0] = 0.0
            roc_storage[model_name]["tprs"].append(interpolated_tpr)
            roc_storage[model_name]["aucs"].append(auc(fpr, tpr))

        voting_model = VotingClassifier(
            estimators=[
                ("SVM", best_models["SVM"]),
                ("LR", best_models["Logistic Regression"]),
            ],
            voting="soft",
        )
        voting_model.fit(X_train, y_train)
        y_prob_voting = voting_model.predict_proba(X_test)[:, 1]
        fpr_v, tpr_v, _ = roc_curve(y_test, y_prob_voting)
        interpolated_tpr_v = np.interp(mean_fpr, fpr_v, tpr_v)
        interpolated_tpr_v[0] = 0.0
        roc_storage["Voting Ensemble (LR + SVM)"]["tprs"].append(interpolated_tpr_v)
        roc_storage["Voting Ensemble (LR + SVM)"]["aucs"].append(auc(fpr_v, tpr_v))

    plt.figure(figsize=(9, 7))
    for model_name, roc_info in roc_storage.items():
        mean_tpr = np.mean(roc_info["tprs"], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(roc_info["aucs"])
        std_auc = np.std(roc_info["aucs"])
        plt.plot(
            mean_fpr,
            mean_tpr,
            color=colors[model_name],
            lw=3.5 if "Voting" in model_name else 1.6,
            alpha=0.85,
            label=f"{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="grey", alpha=0.5)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    plt.title("Figure 1. ROC Curve Comparison", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure_1_Model_Comparison.tif",
        format="tiff",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


if __name__ == "__main__":
    main()
