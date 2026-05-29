import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['workout_types'] = df.get('workout_types', '').fillna('')
    numeric_defaults = [
        'total_workout_minutes',
        'total_workout_distance',
        'total_workout_energy',
        'num_workouts',
        'has_running',
        'has_walking',
        'has_hiit',
    ]
    for col in numeric_defaults:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    return df


def prepare_features(df):
    df = df.copy()
    df['sleep_lag1'] = df['sleep_hours'].shift(1)
    df['sleep_rolling_3d'] = df['sleep_hours'].rolling(window=3, min_periods=1).mean().shift(1)
    df['hr_mean_lag1'] = df['mean_exercise_hr'].shift(1)
    df['hr_max_lag1'] = df['max_hr'].shift(1)

    df['total_workout_minutes'] = df['total_workout_minutes'].fillna(0)
    df['total_workout_distance'] = df['total_workout_distance'].fillna(0)
    df['total_workout_energy'] = df['total_workout_energy'].fillna(0)
    df['num_workouts'] = df['num_workouts'].fillna(0).astype(int)
    df['has_running'] = df['has_running'].fillna(0).astype(int)
    df['has_walking'] = df['has_walking'].fillna(0).astype(int)
    df['has_hiit'] = df['has_hiit'].fillna(0).astype(int)

    df['is_strength'] = df['workout_types'].astype(str).str.contains('Strength', na=False).astype(int)
    df['is_walking'] = df['workout_types'].astype(str).str.contains('Walking', na=False).astype(int)
    df['is_running'] = df['workout_types'].astype(str).str.contains('Running', na=False).astype(int)
    return df


def train_and_evaluate(df, model_path='exercise_hr_rf_model.pkl'):
    workout_days = df[df['workout_types'].notna()].copy()
    workout_days = workout_days.dropna(subset=['sleep_lag1', 'sleep_rolling_3d', 'hr_mean_lag1', 'hr_max_lag1'])

    features = [
        'sleep_lag1',
        'sleep_rolling_3d',
        'hr_mean_lag1',
        'hr_max_lag1',
        'is_strength',
        'is_walking',
        'is_running',
        'num_workouts',
        'total_workout_minutes',
        'total_workout_distance',
        'total_workout_energy',
        'has_running',
        'has_walking',
        'has_hiit',
    ]

    X = workout_days[features]
    y = workout_days['mean_exercise_hr']

    if X.empty:
        raise ValueError('No training rows available after filtering. Check the source CSV and feature generation step.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"模型解釋力 (R2 Score): {r2_score(y_test, y_pred):.3f}")
    print(f"平均預測誤差 (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} BPM")

    joblib.dump(model, model_path)
    print(f"AI 模型已成功儲存為 '{model_path}'")


def main():
    csv_path = os.path.join('apple_health_export', 'sleep_hr_correlation_prevnight_to_nextday_with_workouts.csv')
    df = load_data(csv_path)
    df = prepare_features(df)
    train_and_evaluate(df)


if __name__ == '__main__':
    main()
