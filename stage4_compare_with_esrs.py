import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, confusion_matrix

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)

def calculate_esrs(row: pd.Series) -> int:
    score = 0
    score += 0 if row["age"] < 65 else 1 if row["age"] <= 75 else 2
    score += sum(row.get(column, 0) == 1 for column in ["Hist_HTN", "Hist_DM", "Hist_Stroke", "Hist_Smoke", "Hist_CAD"])
    return score

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--target", default="Label_Recur")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.data.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    df = read_table(args.data)
    df.columns = [str(c).strip() for c in df.columns]

    y = df[args.target].values.ravel()
    y_prob_ml = pd.read_csv(args.oof)["OOF_Prob_ML"].values
    esrs_scores = df.apply(calculate_esrs, axis=1).values

    fpr_ml, tpr_ml, thresholds_ml = roc_curve(y, y_prob_ml)
    auc_ml = auc(fpr_ml, tpr_ml)
    fpr_esrs, tpr_esrs, _ = roc_curve(y, esrs_scores)
    auc_esrs = auc(fpr_esrs, tpr_esrs)

    youden_index = tpr_ml - fpr_ml
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds_ml[best_idx]

    y_pred_ml = (y_prob_ml >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_ml).ravel()
    sens_ml = tp / (tp + fn)
    spec_ml = tn / (tn + fp)
    acc_ml = (tp + tn) / len(y)

    n_bootstraps = 2000
    rng = np.random.RandomState(42)
    boot_metrics = {"auc_ml": [], "auc_esrs": [], "sens": [], "spec": [], "acc": []}

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        y_b = y[indices]
        prob_b = y_prob_ml[indices]
        esrs_b = esrs_scores[indices]

        boot_metrics["auc_ml"].append(auc(*roc_curve(y_b, prob_b)[:2]))
        boot_metrics["auc_esrs"].append(auc(*roc_curve(y_b, esrs_b)[:2]))

        pred_b = (prob_b >= best_threshold).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_b, pred_b).ravel()
        boot_metrics["sens"].append(tp_b / (tp_b + fn_b))
        boot_metrics["spec"].append(tn_b / (tn_b + fp_b))
        boot_metrics["acc"].append((tp_b + tn_b) / len(y_b))

    def get_ci(metric_list):
        return np.percentile(metric_list, [2.5, 97.5])

    ci_auc_ml, ci_auc_esrs = get_ci(boot_metrics["auc_ml"]), get_ci(boot_metrics["auc_esrs"])
    ci_sens, ci_spec, ci_acc = get_ci(boot_metrics["sens"]), get_ci(boot_metrics["spec"]), get_ci(boot_metrics["acc"])

    print(f"Optimal Threshold (Youden Index): {best_threshold:.3f}")
    print(f"Voting Ensemble AUC: {auc_ml:.3f} (95% CI: {ci_auc_ml[0]:.3f}-{ci_auc_ml[1]:.3f})")
    print(f"ESRS AUC           : {auc_esrs:.3f} (95% CI: {ci_auc_esrs[0]:.3f}-{ci_auc_esrs[1]:.3f})")
    print(f"Sensitivity        : {sens_ml:.3f} (95% CI: {ci_sens[0]:.3f}-{ci_sens[1]:.3f})")
    print(f"Specificity        : {spec_ml:.3f} (95% CI: {ci_spec[0]:.3f}-{ci_spec[1]:.3f})")
    print(f"Accuracy           : {acc_ml:.3f} (95% CI: {ci_acc[0]:.3f}-{ci_acc[1]:.3f})")

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_ml, tpr_ml, color="#E63946", lw=3.2, label=f"Voting Ensemble (LR + SVM) (AUC = {auc_ml:.3f})")
    plt.plot(fpr_esrs, tpr_esrs, color="#457B9D", lw=2.4, linestyle="--", label=f"ESRS (AUC = {auc_esrs:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.6, linestyle=":", alpha=0.8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("Figure 3. Predictive Performance Compared with ESRS", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11.5, frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure_3_ML_vs_ESRS.tif",
        format="tiff", dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"}
    )

if __name__ == "__main__":
    main()
