# Healthy_Prediction

**Live demo:** https://healthyprediction.streamlit.app/ (Chinese UI; the first load can take 30–60 seconds while the free server wakes up)

Predict how hard today's workout will feel — as average exercise heart rate — from your recent sleep and training history, using your own Apple Health data.

The project has two parts:

1. **Analysis pipeline**: parses an Apple Health export, builds a daily table of sleep, heart rate, and workouts, and measures how the previous night's sleep relates to next-day exercise heart rate.
2. **Streamlit app**: predicts average heart rate for a planned workout and gives recovery-aware suggestions. Use the built-in demo model, or upload your own export to train a personal one.

## How it works

```
export.zip (Apple Health)
   │  stream-parse export.xml (no full extraction)
   ▼
daily table: sleep hours · exercise HR (mean / max) · workouts (type, minutes, distance, energy)
   │  features: last night's sleep, 3-day sleep average, previous workout HR, planned workout
   ▼
RandomForestRegressor ──► predicted average exercise HR (BPM) + suggestions
```

- **Streaming parser**: `export.xml` is often over 1 GB. It is read straight from the zip with `iterparse`, so it is never extracted to disk and memory stays low (about 100× less than batch parsing).
- **Model**: `RandomForestRegressor` (100 trees, max depth 6) trained on workout days. The app reports R² and RMSE (BPM) and warns when R² is below 0.3.

| Feature | Description |
|---|---|
| `sleep_lag1` | Sleep hours the night before |
| `sleep_rolling_3d` | Average sleep over the previous 3 days |
| `hr_mean_lag1`, `hr_max_lag1` | Mean / max heart rate of the previous workout |
| `is_strength`, `is_walking`, `is_running`, `has_hiit` | Planned workout type |
| `total_workout_minutes`, `total_workout_distance`, `total_workout_energy` | Planned duration, distance, and energy |

## Streamlit app

| Tab | What it does |
|---|---|
| Quick demo | Uses `exercise_hr_rf_model.pkl`, trained on the author's Apple Watch data. Results vary from person to person. |
| Personal prediction | Upload `export.zip`; the app trains a model on your data in about 10–60 seconds and shows its R² and error. |

Enter last night's sleep, your 3-day sleep average, your previous workout's heart rate, and today's plan (type, duration, distance, intensity). The app returns the predicted average heart rate plus suggestions, such as a sleep-debt warning or pacing advice by run type.

`app.py` is the Chinese UI with both tabs; `app-en.py` is an English UI that uses the demo model only.

### Privacy

Uploaded exports are read in memory for the current session only. They are not written to disk, and the raw data is discarded after training. Raw Apple Health data (`apple_health_export/`, `export.xml`) is excluded by `.gitignore` and has never been committed to this repo.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# App only (demo model, or upload your own export in the second tab)
streamlit run app.py

# Full pipeline on your own data: put export.zip in the project root, then
./run_pipeline.sh
```

How to get `export.zip`: on iPhone, open **Health** → tap your profile picture → **Export All Health Data**.

`run_pipeline.sh` runs three steps:

1. `Healthy_Prediction.py`: parses the export and writes CSVs next to `export.xml` (normally `apple_health_export/`), including the previous-night sleep vs. next-day heart-rate correlation (Pearson / Spearman) and a scatter plot.
2. `ml_healthy_detection_exercise_suggestion.py`: trains the model and saves `exercise_hr_rf_model.pkl`.
3. Starts the Streamlit app.

## Project structure

| File | Purpose |
|---|---|
| `Healthy_Prediction.py` | Apple Health parsing, daily aggregation, sleep vs. heart-rate correlation, plots |
| `ml_healthy_detection_exercise_suggestion.py` | Feature engineering and model training |
| `app.py` | Streamlit app (Chinese): demo tab and personal-model tab |
| `app-en.py` | Streamlit app (English): demo model |
| `exercise_hr_rf_model.pkl` | Demo model trained on the author's data |
| `run_pipeline.sh` | Runs parse → train → app |

## Disclaimer

This is a personal side project for exploring wearable data. It is not medical advice. Predictions are only as good as the data behind them, and a personal model with few workout days will have low accuracy.
