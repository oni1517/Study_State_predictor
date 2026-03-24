"""
Setup script — generates data and trains models if they don't exist.
Run once before deploying or on first app load.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join("data", "study_data.csv")
MODEL_PATH = os.path.join("models", "best_model.joblib")


def setup():
    if not os.path.exists(DATA_PATH):
        print("Generating dataset...")
        from src.generate_data import generate_dataset
        os.makedirs("data", exist_ok=True)
        df = generate_dataset(n_samples=5000)
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved to {DATA_PATH}")
    else:
        print(f"Dataset already exists at {DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        print("Training models...")
        from src.train_models import main as train_main
        train_main()
        print("Models trained and saved.")
    else:
        print(f"Models already exist at {MODEL_PATH}")


if __name__ == "__main__":
    setup()
