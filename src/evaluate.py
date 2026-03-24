"""
Evaluation and Visualization Module for Study Crash Predictor.

Generates confusion matrices, feature importance plots,
model comparison charts, and saves them as images.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.train_models import FEATURE_COLS, get_models, load_data, preprocess


def plot_confusion_matrices(results, le, save_dir):
    """Plot and save confusion matrices for all models."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax,
        )
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(results, save_dir):
    """Bar chart comparing accuracy and F1 across models."""
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    f1_scores = [results[n]["f1_score"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#4A90D9")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 Score (weighted)", color="#E8734A")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: Accuracy vs F1 Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(model, feature_cols, model_name, save_dir):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not support feature_importances_. Skipping.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_cols)))

    bars = ax.barh(
        range(len(feature_cols)),
        importances[indices][::-1],
        color=colors,
    )
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices][::-1])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, importances[indices][::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_cross_validation(results, save_dir):
    """Plot cross-validation scores as box plots."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = []
    cv_means = []
    cv_stds = []
    for name, res in results.items():
        names.append(name)
        cv_means.append(res["cv_mean"])
        cv_stds.append(res["cv_std"])

    x = range(len(names))
    ax.bar(x, cv_means, yerr=cv_stds, capsize=5, color="#6C5CE7", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "cross_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    """Run the full evaluation pipeline and generate all plots."""
    project_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(project_dir, "data", "study_data.csv")
    models_dir = os.path.join(project_dir, "models")
    plots_dir = os.path.join(project_dir, "models", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data and retrain all models to get results
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, scaler, le = preprocess(df)

    models = get_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, average="weighted")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

    # Generate all plots
    plot_confusion_matrices(results, le, plots_dir)
    plot_model_comparison(results, plots_dir)
    plot_cross_validation(results, plots_dir)

    # Feature importance for best tree-based model
    for name in ["Gradient Boosting", "Random Forest"]:
        if name in results:
            plot_feature_importance(results[name]["model"], FEATURE_COLS, name, plots_dir)
            break

    print(f"\nAll evaluation plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
