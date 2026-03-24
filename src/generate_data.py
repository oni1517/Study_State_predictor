"""
Synthetic Dataset Generator for Study Crash Predictor.

Generates realistic student study session data with behavioral features
that correlate with focus levels, distraction, and study crashes.
"""

import os
import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic dataset of student study sessions.

    Each row represents a snapshot of a student's study state with
    behavioral signals that predict whether they will stay focused,
    become distracted, or crash.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with features and a 3-class target label.
    """
    rng = np.random.default_rng(seed)

    # --- Feature Generation ---

    # Hours slept (4-10 hours, normal-ish distribution centered at 7)
    hours_slept = np.clip(rng.normal(7, 1.5, n_samples), 4, 10).round(1)

    # Study duration so far in minutes (10-240)
    study_duration_minutes = np.clip(rng.exponential(60, n_samples) + 10, 10, 240).astype(int)

    # Phone pickups in the last hour (0-30, right-skewed)
    phone_pickups_last_hour = np.clip(rng.exponential(4, n_samples), 0, 30).astype(int)

    # Social media opens in the last hour (0-20, correlated with phone pickups)
    social_media_opens_last_hour = np.clip(
        (phone_pickups_last_hour * rng.uniform(0.3, 0.9, n_samples) + rng.exponential(1, n_samples)),
        0, 20,
    ).astype(int)

    # Break duration in minutes (0-60)
    break_duration_minutes = np.clip(rng.exponential(8, n_samples), 0, 60).astype(int)

    # Subject difficulty (1-10 scale)
    subject_difficulty = rng.integers(1, 11, n_samples)

    # Current time of day (hour 0-23, bimodal: morning and evening study)
    time_peak_1 = rng.normal(10, 2, n_samples // 2)
    time_peak_2 = rng.normal(20, 2, n_samples - n_samples // 2)
    current_time_of_day = np.clip(
        np.concatenate([time_peak_1, time_peak_2]) % 24, 6, 23,
    ).astype(int)
    rng.shuffle(current_time_of_day)

    # Caffeine intake (cups 0-6)
    caffeine_intake = np.clip(rng.poisson(1.5, n_samples), 0, 6)

    # Previous day productivity (1-10 scale)
    previous_day_productivity = np.clip(rng.normal(6, 2, n_samples), 1, 10).round(1)

    # Mood level (1-10 scale, influenced by sleep and time of day)
    mood_base = (
        hours_slept / 10 * 5
        + (1 - np.abs(current_time_of_day - 14) / 14) * 3
        + rng.normal(0, 1.5, n_samples)
    )
    mood_level = np.clip(mood_base, 1, 10).round(1)

    # --- Target Label Logic ---
    # Compute a "crash score" from features; higher = more likely to crash.
    # Then map to 3 classes via quantile thresholds.

    crash_score = (
        -0.35 * hours_slept
        + 0.02 * study_duration_minutes
        + 0.18 * phone_pickups_last_hour
        + 0.12 * social_media_opens_last_hour
        - 0.06 * break_duration_minutes
        + 0.15 * subject_difficulty
        - 0.05 * caffeine_intake
        - 0.10 * previous_day_productivity
        - 0.25 * mood_level
        + rng.normal(0, 0.6, n_samples)  # noise
    )

    # Also boost crash score for very late night (after 22) or very long study sessions
    late_night = (current_time_of_day >= 22).astype(float)
    long_session = (study_duration_minutes >= 150).astype(float)
    crash_score += 1.0 * late_night + 0.8 * long_session

    # Distracted boost: high phone/social media, moderate time
    distracted_score = (
        0.25 * phone_pickups_last_hour
        + 0.20 * social_media_opens_last_hour
        - 0.15 * mood_level
        + rng.normal(0, 0.5, n_samples)
    )

    combined_score = crash_score + 0.3 * distracted_score

    q33 = np.percentile(combined_score, 33)
    q66 = np.percentile(combined_score, 66)

    target = np.where(
        combined_score <= q33, "Deep Focus",
        np.where(combined_score <= q66, "Distracted", "Study Crash Incoming"),
    )

    df = pd.DataFrame({
        "hours_slept": hours_slept,
        "study_duration_minutes": study_duration_minutes,
        "phone_pickups_last_hour": phone_pickups_last_hour,
        "social_media_opens_last_hour": social_media_opens_last_hour,
        "break_duration_minutes": break_duration_minutes,
        "subject_difficulty": subject_difficulty,
        "current_time_of_day": current_time_of_day,
        "caffeine_intake": caffeine_intake,
        "previous_day_productivity": previous_day_productivity,
        "mood_level": mood_level,
        "target": target,
    })

    return df


def main():
    """Generate dataset and save to CSV."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate_dataset(n_samples=5000)
    path = os.path.join(data_dir, "study_data.csv")
    df.to_csv(path, index=False)

    print(f"Dataset saved to {path}")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")


if __name__ == "__main__":
    main()
