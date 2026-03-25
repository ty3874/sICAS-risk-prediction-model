"""Stage 5: comparison between the ensemble model and ESRS."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def calculate_esrs(row: pd.Series) -> int:
    score = 0
    score += 0 if row["age"] < 65 else 1 if row["age"] <= 75 else 2
    score += sum(
        row.get(column, 0) == 1
        for column in ["Hist_HTN", "Hist_DM", "Hist_Stroke", "Hist_Smoke", "Hist_CAD"]
    )
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
    df.columns = [str(column).strip() for column in df.columns]
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' was not found in the input file.")

    y = df[args.target].values.ravel()
    y_prob_ml = pd.read_csv(args.oof)["OOF_Prob_ML"].values

    esrs_scores = df.apply(calculate_esrs, axis=1).values

    fpr_ml, tpr_ml, _ = roc_curve(y, y_prob_ml)
    auc_ml = auc(fpr_ml, tpr_ml)
    fpr_esrs, tpr_esrs, _ = roc_curve(y, esrs_scores)
    auc_esrs = auc(fpr_esrs, tpr_esrs)

    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr_ml,
        tpr_ml,
        color="#E63946",
        lw=3.2,
        label=f"Voting Ensemble (LR + SVM) (AUC = {auc_ml:.3f})",
    )
    plt.plot(
        fpr_esrs,
        tpr_esrs,
        color="#457B9D",
        lw=2.4,
        linestyle="--",
        label=f"ESRS (AUC = {auc_esrs:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="grey", lw=1.6, linestyle=":", alpha=0.8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    plt.title("Figure 3. Predictive Performance Compared with ESRS", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11.5, frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure_3_ML_vs_ESRS.tif",
        format="tiff",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


if __name__ == "__main__":
    main()
