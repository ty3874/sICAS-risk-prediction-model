import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMPUTER_SEED = 42
MODEL_SEED = 42
CV_SEED = 1357

FINAL_FEATURES = [
    "age", "SBP", "NIHSS_In", "eGFR", "Glucose", "LDL",
    "Stenosis_Pct", "tmax6", "rcbf34",
]

DISPLAY_FEATURES = [
    "Age", "SBP", "NIHSS", "eGFR", "FPG", "LDL-C",
    "Stenosis", "Tmax >6 s", "rCBF <34%",
]

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)

def calculate_net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> list[float]:
    net_benefits = []
    n_samples = len(y_true)
    for threshold in thresholds:
        y_pred = y_prob >= threshold
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        net_benefits.append((tp / n_samples) - (fp / n_samples) * (threshold / (1 - threshold)))
    return net_benefits

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--target", default="Label_Recur")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.data.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    df = read_table(args.data)
    df.columns = [str(c).strip() for c in df.columns]

    X = df[FINAL_FEATURES].copy()
    y = df[args.target].values.ravel()

    pipe_lr = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=IMPUTER_SEED)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000, random_state=MODEL_SEED)),
    ])
    pipe_svm = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=IMPUTER_SEED)),
        ("scaler", StandardScaler()),
        ("clf", SVC(class_weight="balanced", C=1.0, kernel="rbf", probability=True, random_state=MODEL_SEED)),
    ])

    voting_base = VotingClassifier(estimators=[("SVM", pipe_svm), ("LR", pipe_lr)], voting="soft")
    calibrated_voting = CalibratedClassifierCV(estimator=voting_base, method="sigmoid", cv=3)

    global_imputer = IterativeImputer(max_iter=10, random_state=IMPUTER_SEED)
    X_imputed = pd.DataFrame(global_imputer.fit_transform(X), columns=X.columns)
    joblib.dump(global_imputer, output_dir / "global_imputer.joblib")

    rf_surrogate = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=1, random_state=MODEL_SEED)
    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)

    y_prob_calibrated = cross_val_predict(calibrated_voting, X, y, cv=cv_eval, method="predict_proba", n_jobs=-1)[:, 1]
    
    pd.DataFrame({"True_Label": y, "OOF_Prob_ML": y_prob_calibrated}).to_csv(
        output_dir / "OOF_Probabilities_For_Stage4.csv", index=False,
    )
    
    brier_cal = brier_score_loss(y, y_prob_calibrated)
    calibrated_voting.fit(X, y)
    rf_surrogate.fit(X_imputed, y)
    
    joblib.dump(calibrated_voting, output_dir / "calibrated_ensemble.joblib")
    joblib.dump(rf_surrogate, output_dir / "rf_surrogate.joblib")

    explainer = shap.TreeExplainer(rf_surrogate)
    shap_values_raw = explainer.shap_values(X_imputed)
    shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else (shap_values_raw[:, :, 1] if len(shap_values_raw.shape) == 3 else shap_values_raw)

    plt.figure(figsize=(10, 8))
    X_display = X_imputed.copy()
    X_display.columns = DISPLAY_FEATURES
    shap.summary_plot(shap_values, X_display, plot_type="dot", show=False, max_display=10)
    plt.title("Figure 5. SHAP Feature Importance", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure_5_SHAP_Summary.tif",
        format="tiff", dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"}
    )

    fig = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    fraction_of_positives, mean_predicted_value = calibration_curve(y, y_prob_calibrated, n_bins=5, strategy="quantile")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", color="#E63946", linewidth=2.5, label=f"Final Ensemble\n(Brier = {brier_cal:.3f})")
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.set(ylabel="Fraction of Positives", xlabel="Mean Predicted Probability", title="A. Calibration Curve (Platt Scaling)")
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    thresholds = np.linspace(0.01, 0.50, 100)
    ax2.plot(thresholds, calculate_net_benefit(y, y_prob_calibrated, thresholds), color="#E63946", linewidth=3, label="Final Ensemble")
    ax2.plot(thresholds, calculate_net_benefit(y, np.ones_like(y), thresholds), color="gray", linestyle="--", linewidth=2, label="Treat All")
    ax2.plot(thresholds, np.zeros_like(thresholds), color="black", linewidth=2, label="Treat None")
    ax2.set(xlim=[0.01, 0.40], ylim=[-0.02, 0.12], xlabel="Threshold Probability", ylabel="Net Benefit", title="B. Decision Curve Analysis (DCA)")
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure_4_Comprehensive.tif",
        format="tiff", dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"}
    )

if __name__ == "__main__":
    main()
