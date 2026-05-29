#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Step 1: 處理 Apple Health 資料 ==="
python Healthy_Prediction.py

echo ""
echo "=== Step 2: 訓練 ML 模型 ==="
python ml_healthy_detection_exercise_suggestion.py

echo ""
echo "=== Step 3: 啟動 Streamlit App ==="
streamlit run app.py
