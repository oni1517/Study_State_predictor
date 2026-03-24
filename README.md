# Study Crash Predictor

A machine learning project that predicts whether a student is likely to remain in **Deep Focus**, become **Distracted**, or experience a **Study Crash** based on real-time study habits and daily behavior.

---

## Project Overview

Students often lose productivity without warning. This project uses behavioral signals — sleep, phone usage, study duration, mood, and more — to predict a student's current study state and provide actionable recommendations.

Three classes:
| Class | Meaning |
|---|---|
| **Deep Focus** | Student is fully engaged and productive |
| **Distracted** | Attention is slipping, corrective action needed |
| **Study Crash Incoming** | Burnout is imminent, rest is required |

---

## Project Structure

```
Study_Crash_Predictor/
├── data/                  # Generated dataset (CSV)
├── notebooks/             # Jupyter notebook for EDA
│   └── exploration.ipynb
├── models/                # Saved models and plots
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── feature_cols.json
│   ├── model_comparison.json
│   └── plots/
│       ├── confusion_matrices.png
│       ├── model_comparison.png
│       ├── feature_importance.png
│       └── cross_validation.png
├── app/                   # Streamlit web application
│   └── streamlit_app.py
├── src/                   # Source modules
│   ├── __init__.py
│   ├── generate_data.py   # Synthetic dataset generator
│   ├── train_models.py    # Model training & comparison
│   ├── evaluate.py        # Evaluation & visualization
│   └── predict.py         # Prediction utility
├── requirements.txt
└── README.md
```

---

## Dataset

A synthetic dataset of 5,000 student study sessions with 10 features:

| Feature | Description | Range |
|---|---|---|
| `hours_slept` | Hours slept last night | 4.0 - 10.0 |
| `study_duration_minutes` | Current session length | 10 - 240 |
| `phone_pickups_last_hour` | Phone pickups | 0 - 30 |
| `social_media_opens_last_hour` | Social media app opens | 0 - 20 |
| `break_duration_minutes` | Last break duration | 0 - 60 |
| `subject_difficulty` | Subject difficulty rating | 1 - 10 |
| `current_time_of_day` | Hour of day | 6 - 23 |
| `caffeine_intake` | Cups of coffee/tea today | 0 - 6 |
| `previous_day_productivity` | Yesterday's productivity | 1.0 - 10.0 |
| `mood_level` | Current mood rating | 1.0 - 10.0 |

The target label is generated from a weighted combination of features with added noise to simulate realistic patterns.

---

## Models Trained

| Model | Description |
|---|---|
| **Random Forest** | Ensemble of 200 decision trees |
| **Gradient Boosting** | 200 boosting rounds, learning rate 0.1 |
| **Logistic Regression** | Multinomial logistic with L2 regularization |
| **SVM** | RBF kernel with probability estimates |

Models are evaluated using:
- **Accuracy**
- **Weighted F1 Score**
- **5-Fold Cross-Validation**
- **Confusion Matrix**
- **Classification Report** (precision, recall, F1 per class)

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset

```bash
python -m src.generate_data
```

This creates `data/study_data.csv` with 5,000 samples.

### 3. Train Models

```bash
python -m src.train_models
```

This trains 4 models, compares them, and saves the best one to `models/`.

### 4. Generate Evaluation Plots

```bash
python -m src.evaluate
```

This creates confusion matrices, model comparison charts, feature importance, and cross-validation plots in `models/plots/`.

### 5. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Streamlit App Features

- **Input Controls**: Sidebar sliders for all 10 study features
- **Real-time Prediction**: One-click prediction with confidence scores
- **Probability Visualization**: Horizontal bar chart of class probabilities
- **Smart Recommendations**: Context-aware advice based on the prediction
- **Model Insights**: Feature importance and model comparison charts
- **Modern UI**: Academic-themed design with gradient cards and clean typography

---

## Evaluation Results

Results are saved to `models/model_comparison.json` after training.

Example (will vary by random seed):
```
Random Forest:      Accuracy ~0.82,  F1 ~0.82
Gradient Boosting:  Accuracy ~0.83,  F1 ~0.83  (typically best)
Logistic Regression: Accuracy ~0.75,  F1 ~0.75
SVM:                Accuracy ~0.80,  F1 ~0.80
```

---

## Quick Start (All-in-One)

```bash
# Clone and enter project
cd Study_Crash_Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install
pip install -r requirements.txt

# Generate data, train models, and evaluate
python setup.py
python -m src.evaluate

# Launch app
streamlit run app/streamlit_app.py
```

---

## Deploy to the Cloud

Streamlit needs a live Python server, so it **cannot** deploy on Vercel.
Use one of these free platforms instead:

### Option 1: Streamlit Community Cloud (Recommended)

1. Push your project to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo
4. Set **main file path** to `app/streamlit_app.py`
5. Click **Deploy** — done in ~2 minutes

Your repo must include:
- `requirements.txt` (already present)
- `data/study_data.csv` (committed to repo)
- `models/*.joblib` (committed to repo)

### Option 2: Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
3. Railway auto-detects the `Procfile` and `runtime.txt`
4. App deploys automatically

### Option 3: Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Upload all project files
4. The app builds and runs automatically

### Option 4: Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo
4. Set **Build Command**: `pip install -r requirements.txt && python setup.py`
5. Set **Start Command**: `streamlit run app/streamlit_app.py --server.port=$PORT --server.headless=true`
6. Deploy

---

## Tech Stack

- **Python 3.9+**
- **scikit-learn** - ML models and preprocessing
- **pandas / numpy** - Data manipulation
- **matplotlib / seaborn** - Visualizations
- **Streamlit** - Web application
- **joblib** - Model serialization

---

## License

MIT
