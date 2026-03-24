"""
Prediction Utility Module for Study Crash Predictor.

Loads the saved model and provides prediction functions
for both batch and single-sample inference.
"""

import os
import json
import joblib
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


def load_artifacts(models_dir: str = MODELS_DIR):
    """Load the saved model, scaler, label encoder, and feature columns.

    Returns:
        model, scaler, label_encoder, feature_cols
    """
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))

    with open(os.path.join(models_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    return model, scaler, le, feature_cols


def predict_single(features_dict: dict, models_dir: str = MODELS_DIR):
    """Predict the study state for a single sample.

    Args:
        features_dict: Dictionary with keys matching FEATURE_COLS.
        models_dir: Path to the directory with saved artifacts.

    Returns:
        predicted_label: The predicted class name (str).
        probabilities: Dict mapping class name to probability.
    """
    model, scaler, le, feature_cols = load_artifacts(models_dir)

    # Build feature vector in the correct order
    X = np.array([[features_dict[col] for col in feature_cols]])
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    label = le.inverse_transform([prediction])[0]

    # Probabilities (if model supports it)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)}
    else:
        probabilities = {label: 1.0}

    return label, probabilities


def get_recommendation(prediction: str) -> str:
    """Return smart recommendations based on the prediction.

    Args:
        prediction: One of "Deep Focus", "Distracted", "Study Crash Incoming".

    Returns:
        A recommendation string with actionable advice.
    """
    recommendations = {
        "Deep Focus": (
            "**You're in deep focus! Great job!**\n\n"
            "- Keep going — you're in the zone.\n"
            "- Minimize notifications for the next 30 minutes.\n"
            "- Take a short 5-minute break in ~25 minutes (Pomodoro style).\n"
            "- Stay hydrated and maintain your current pace."
        ),
        "Distracted": (
            "**You're getting distracted. Let's fix that.**\n\n"
            "- Put your phone in another room or enable Do Not Disturb.\n"
            "- Close all unrelated browser tabs right now.\n"
            "- Stand up and do 2 minutes of stretching.\n"
            "- Set a 25-minute focused timer and commit to it.\n"
            "- Switch to a less difficult topic for 15 minutes to rebuild momentum.\n"
            "- Try background white noise or lo-fi music."
        ),
        "Study Crash Incoming": (
            "**Warning: Study crash incoming. Please take care of yourself.**\n\n"
            "- Stop studying NOW — your brain needs a reset.\n"
            "- Take a 20-30 minute break minimum.\n"
            "- Go for a short walk or do light exercise.\n"
            "- Drink water and have a healthy snack.\n"
            "- If it's late, consider ending the session and getting sleep.\n"
            "- Block social media apps using a focus tool.\n"
            "- Plan a lighter study schedule for tomorrow.\n"
            "- Remember: rest is productive. A crash helps no one."
        ),
    }
    return recommendations.get(prediction, "No recommendation available.")


if __name__ == "__main__":
    # Quick test
    sample = {
        "hours_slept": 7.5,
        "study_duration_minutes": 45,
        "phone_pickups_last_hour": 2,
        "social_media_opens_last_hour": 1,
        "break_duration_minutes": 10,
        "subject_difficulty": 5,
        "current_time_of_day": 10,
        "caffeine_intake": 1,
        "previous_day_productivity": 7.0,
        "mood_level": 8.0,
    }
    label, probs = predict_single(sample)
    print(f"Prediction: {label}")
    print(f"Probabilities: {probs}")
    print(f"\nRecommendation:\n{get_recommendation(label)}")
