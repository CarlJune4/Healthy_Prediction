import os
import io
import tempfile
import zipfile
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from Healthy_Prediction import (
    load_records_from_xml,
    load_workouts_from_xml,
    compute_sleep_scores,
    compute_exercise_metrics,
    correlate_sleep_vs_exercise,
    map_workout_activity,
)
from ml_healthy_detection_exercise_suggestion import prepare_features

st.set_page_config(page_title="AI 運動負荷與智慧推薦系統", page_icon="⌚", layout="centered")
st.title("⌚ AI 運動生理負荷預測與智慧推薦系統")

# ── 解析 Apple Health export.zip ────────────────────────────────────────────

def parse_zip_to_df(uploaded_file) -> pd.DataFrame:
    """上傳的 export.zip → 解析成訓練用 DataFrame（與 Healthy_Prediction.main 邏輯相同）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_bytes = uploaded_file.read()
        zip_path = os.path.join(tmpdir, "export.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        # 解壓找 export.xml
        xml_path = None
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                if member.endswith("export.xml"):
                    z.extract(member, tmpdir)
                    xml_path = os.path.join(tmpdir, member)
                    break

        if xml_path is None or not os.path.exists(xml_path):
            raise FileNotFoundError("在 zip 中找不到 export.xml，請確認是從 Apple Health 匯出的原始檔案。")

        df = load_records_from_xml(xml_path)
        if df.empty:
            raise ValueError("解析後沒有任何紀錄，請確認 export.xml 內容。")

        hr = df[df["type"] == "HKQuantityTypeIdentifierHeartRate"].copy()
        sleep = df[df["type"] == "HKCategoryTypeIdentifierSleepAnalysis"].copy()

        sleep_scores = compute_sleep_scores(sleep)
        exercise_metrics = compute_exercise_metrics(hr)
        corr = correlate_sleep_vs_exercise(sleep_scores, exercise_metrics, sleep_shift_days=1)

        if corr["n"] == 0:
            raise ValueError("資料中找不到足夠的睡眠與運動心率重疊日期，無法訓練模型（至少需要 10 天以上的運動紀錄）。")

        wdf = load_workouts_from_xml(xml_path)
        merged = corr["merged"].copy()

        if not wdf.empty:
            wdf["workout_date"] = wdf["startDate"].dt.date
            wdf["activity_name"] = wdf["activity"].apply(map_workout_activity)
            workout_daily = wdf.groupby("workout_date").agg(
                workout_types=("activity_name", lambda vals: ";".join(sorted(set([v for v in vals if v])))),
                num_workouts=("activity_name", "count"),
                total_workout_minutes=("duration_minutes", "sum"),
                total_workout_distance=("totalDistance", "sum"),
                total_workout_energy=("totalEnergyBurned", "sum"),
            )
            workout_daily["has_running"] = workout_daily["workout_types"].str.contains("Running", na=False).astype(int)
            workout_daily["has_walking"] = workout_daily["workout_types"].str.contains("Walking", na=False).astype(int)
            workout_daily["has_hiit"] = workout_daily["workout_types"].str.contains("HIIT", na=False).astype(int)
            merged = merged.join(workout_daily, how="left")

        for col in ["workout_types"]:
            if col not in merged.columns:
                merged[col] = ""
            merged[col] = merged[col].fillna("")
        for col in ["num_workouts", "has_running", "has_walking", "has_hiit"]:
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0).astype(int)
        for col in ["total_workout_minutes", "total_workout_distance", "total_workout_energy"]:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)

        merged.index = pd.to_datetime(merged.index)
        merged = merged.rename_axis("date").reset_index()
        merged["date"] = pd.to_datetime(merged["date"])
        return merged


def train_personal_model(df: pd.DataFrame):
    """從 DataFrame 訓練個人化 Random Forest，回傳 (model, r2, rmse, n_samples)"""
    df = prepare_features(df)
    features = [
        "sleep_lag1", "sleep_rolling_3d", "hr_mean_lag1", "hr_max_lag1",
        "is_strength", "is_walking", "is_running",
        "num_workouts", "total_workout_minutes", "total_workout_distance",
        "total_workout_energy", "has_running", "has_walking", "has_hiit",
    ]
    workout_days = df[df["workout_types"].notna() & (df["workout_types"] != "")].copy()
    workout_days = workout_days.dropna(subset=["sleep_lag1", "sleep_rolling_3d", "hr_mean_lag1", "hr_max_lag1"])

    if len(workout_days) < 10:
        raise ValueError(f"有效運動日只有 {len(workout_days)} 天，至少需要 10 天才能訓練模型。")

    X = workout_days[features]
    y = workout_days["mean_exercise_hr"]

    if len(X) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, r2, rmse, len(workout_days)


# ── 預測 UI ────────────────────────────────────────────────────────────────

def prediction_ui(model, model_label: str):
    st.markdown(f"**使用模型：{model_label}**")
    st.header("📥 輸入今日身體狀態與計畫")

    sleep_yesterday = st.slider("昨晚睡眠時數 (小時)", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
    sleep_3d_avg = st.slider("過去 3 天平均睡眠 (小時)", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
    last_hr_mean = st.number_input("上一次運動的平均心率 (BPM)", min_value=60, max_value=200, value=105)
    last_hr_max = st.number_input("上一次運動的最大心率 (BPM)", min_value=80, max_value=220, value=135)
    workout_type = st.selectbox("預計今日運動項目", ["重量訓練 (Strength Training)", "戶外/室內健走 (Walking)", "跑步 (Running)", "其他常規運動"])
    planned_duration = st.slider("預計運動時間 (分鐘)", min_value=10, max_value=240, value=60, step=5)

    if workout_type in ["戶外/室內健走 (Walking)", "跑步 (Running)"]:
        planned_distance = st.number_input("預計運動距離 (公里)", min_value=0.0, max_value=100.0, value=5.0, step=0.5, format="%.1f")
    else:
        planned_distance = 0.0

    planned_intensity = st.selectbox("預計訓練強度", ["低強度", "中等強度", "高強度"])

    run_mode = None
    if "Running" in workout_type:
        run_mode = st.selectbox("跑步訓練類型", ["一般跑步", "輕鬆跑", "長距離", "間歇"])

    is_strength = 1 if "Strength" in workout_type else 0
    is_walking = 1 if "Walking" in workout_type else 0
    is_running = 1 if "Running" in workout_type else 0
    planned_energy = 80 if planned_intensity == "低強度" else 180 if planned_intensity == "中等強度" else 320
    has_hiit = 1 if planned_intensity == "高強度" and "Running" in workout_type else 0

    input_data = np.array([[
        sleep_yesterday, sleep_3d_avg, last_hr_mean, last_hr_max,
        is_strength, is_walking, is_running,
        1, planned_duration, planned_distance, planned_energy,
        is_running, is_walking, has_hiit,
    ]])

    if st.button("🚀 開始智慧身體負荷評估"):
        predicted_hr = model.predict(input_data)[0]
        st.subheader("📊 AI 預測與評估結果")
        st.metric(label="預估今日運動平均心率", value=f"{predicted_hr:.1f} BPM")
        st.markdown(f"**訓練計畫：{planned_duration} 分鐘 / {planned_distance:.1f} 公里 / {planned_intensity}**")
        if run_mode:
            st.markdown(f"**跑步訓練類型：{run_mode}**")

        st.subheader("💡 智慧運動與健康建議")
        if sleep_yesterday < 6.0 or sleep_3d_avg < 6.5:
            st.warning("⚠️ **睡眠負債警告**：您近期的睡眠時間明顯不足。")
            if predicted_hr > 115:
                st.error("🚨 **高心肺風險預警**：AI 預估今日運動心率偏高。在睡眠不足的情況下進行高強度運動會大幅增加心臟負擔與受傷風險。建議將今日運動改為低強度的健走或拉伸，或者減少 50% 的運動量。")
            else:
                st.info("💡 雖然 AI 預估該項目的心率負荷在正常範圍，但鑑於體能處於低谷，建議以動態恢復為主，切勿盲目衝擊重量。")
        else:
            st.success("✅ **充電狀態良好**：您近期的睡眠充足，身體復原狀況良好。")
            if is_running or predicted_hr > 115:
                if run_mode == "輕鬆跑":
                    st.success("🔥 **適合輕鬆跑**：今日適合進行恢復性的輕鬆跑，維持穩定配速即可。")
                elif run_mode == "長距離":
                    st.success("🔥 **適合長距離訓練**：建議控制配速，保持耐力穩定。")
                elif run_mode == "間歇":
                    st.success("🔥 **適合間歇訓練**：以高低強度交替為主。")
                else:
                    st.success("🔥 **適合高強度訓練**：今日身體各項指標非常適合進行心肺或高強度有氧訓練，可以嘗試突破個人紀錄（PR）！")
            else:
                if run_mode == "輕鬆跑":
                    st.info("💪 今日適合輕鬆跑。建議以穩定節奏、呼吸輕鬆為主。")
                elif run_mode == "長距離":
                    st.info("💪 今日適合長距離耐力跑。建議以耐力配速為主，避免一開始跑得太快。")
                elif run_mode == "間歇":
                    st.info("💡 今日間歇跑請確認熱身充分。")
                else:
                    st.info("💪 今日身體狀態優異，完成常規的重量訓練或輕度運動將非常輕鬆。")


# ── 主頁面 ─────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🚀 快速體驗（示範模型）", "🧬 個人化預測（上傳 Apple Health）"])

# Tab 1：使用預訓練的示範模型
with tab1:
    st.markdown("使用示範模型（基於作者本人的 Apple Watch 數據），可快速體驗功能，但準確度因人而異。")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "exercise_hr_rf_model.pkl")

    @st.cache_resource
    def load_default_model():
        return joblib.load(MODEL_PATH)

    try:
        default_model = load_default_model()
        prediction_ui(default_model, "示範模型（作者數據）")
    except Exception as exc:
        st.error(f"無法載入示範模型：{exc}")

# Tab 2：上傳個人 Apple Health 資料，訓練個人化模型
with tab2:
    st.markdown("""
    **如何取得 Apple Health 匯出檔案：**
    1. 打開 iPhone 的「健康」App
    2. 右上角點選頭像 → 向下捲動 → **匯出所有健康數據**
    3. 等待打包完成後，將 `export.zip` 傳到電腦並上傳至此

    > 您的資料只在瀏覽器與伺服器之間處理，模型訓練完成後即丟棄原始資料，不會儲存。
    """)

    uploaded = st.file_uploader("上傳 export.zip", type=["zip"])

    if uploaded is not None:
        if "personal_model" not in st.session_state or st.session_state.get("uploaded_name") != uploaded.name:
            with st.spinner("正在解析您的健康數據並訓練個人化模型，請稍候（約 10～60 秒）..."):
                try:
                    df = parse_zip_to_df(uploaded)
                    model, r2, rmse, n_days = train_personal_model(df)
                    st.session_state["personal_model"] = model
                    st.session_state["uploaded_name"] = uploaded.name
                    st.session_state["model_stats"] = (r2, rmse, n_days)
                except Exception as e:
                    st.error(f"處理失敗：{e}")
                    st.stop()

        if "personal_model" in st.session_state:
            r2, rmse, n_days = st.session_state["model_stats"]
            col1, col2, col3 = st.columns(3)
            col1.metric("訓練樣本（有效運動天）", f"{n_days} 天")
            col2.metric("模型解釋力 R²", f"{r2:.3f}")
            col3.metric("平均預測誤差", f"{rmse:.1f} BPM")

            if r2 < 0.3:
                st.warning("⚠️ 模型解釋力較低（R² < 0.3），可能是因為運動紀錄天數不足或數據變異較大，預測結果僅供參考。")

            st.divider()
            prediction_ui(st.session_state["personal_model"], "您的個人化模型")
