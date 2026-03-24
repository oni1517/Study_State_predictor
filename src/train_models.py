"""
Model Training Module for Study Crash Predictor.

Trains and compares multiple ML classifiers on the study dataset.
Saves the best model, the label encoder, and the scaler to disk.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

FEATURE_COLS = [
    "hours_slept",
    "study_duration_minutes",
    "phone_pickups_last_hour",
    "social_media_opens_last_hour",
    "break_duration_minutes",
    "subject_difficulty",
    "current_time_of_day",
    "caffeine_intake",
    "previous_day_productivity",
    "mood_level",
]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame):
    """Encode target, scale features, split train/test.

    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    le = LabelEncoder()
    y = le.fit_transform(df["target"])

    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y,
    )

    return X_train, X_test, y_train, y_test, scaler, le


def get_models():
    """Return a dictionary of model names and classifier instances."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42,
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    """Train all models, evaluate, and return results dict."""
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:\n{report}")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "report": report,
        }

    return results


def save_best_model(results, scaler, le, feature_cols, save_dir):
    """Identify the best model and save artifacts."""
    os.makedirs(save_dir, exist_ok=True)

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = results[best_name]["model"]

    print(f"\n{'='*50}")
    print(f"Best Model: {best_name}")
    print(f"F1 Score: {results[best_name]['f1_score']:.4f}")
    print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"{'='*50}")

    # Save model, scaler, encoder
    joblib.dump(best_model, os.path.join(save_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))

    # Save feature columns
    with open(os.path.join(save_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    # Save comparison summary
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "accuracy": round(res["accuracy"], 4),
            "f1_score": round(res["f1_score"], 4),
            "cv_mean": round(res["cv_mean"], 4),
            "cv_std": round(res["cv_std"], 4),
        }
    with open(os.path.join(save_dir, "model_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll artifacts saved to {save_dir}")
    return best_name, results


def main():
    """Run the full training pipeline."""
    project_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(project_dir, "data", "study_data.csv")
    models_dir = os.path.join(project_dir, "models")

    df = load_data(csv_path)
    print(f"Dataset loaded: {df.shape}")

    X_train, X_test, y_train, y_test, scaler, le = preprocess(df)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    results = train_and_evaluate(X_train, X_test, y_train, y_test, le)
    save_best_model(results, scaler, le, FEATURE_COLS, models_dir)

    # Save test data for evaluation module
    np.savez(
        os.path.join(models_dir, "test_data.npz"),
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
