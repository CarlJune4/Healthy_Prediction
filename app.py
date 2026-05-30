import os  # still needed for MODEL_PATH
import io
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import timedelta
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# sklearn 和 Healthy_Prediction 只在需要時才載入（延遲 import）
# 這讓頁面初始渲染不被這些重型套件拖慢（冷啟動可省 2~3 秒）

st.set_page_config(page_title="AI 運動負荷與智慧推薦系統", page_icon="⌚", layout="centered")
st.title("⌚ AI 運動生理負荷預測與智慧推薦系統")

st.info("⏳ **首次開啟或長時間未使用時，伺服器需要約 30～60 秒喚醒，頁面完成載入後即可正常使用。**")

# ── workout 類型對應表（內聯，避免載入整個 Healthy_Prediction 模組）───────────

_WORKOUT_MAP = {
    "HKWorkoutActivityTypeRunning": "Running",
    "HKWorkoutActivityTypeWalking": "Walking",
    "HKWorkoutActivityTypeCycling": "Cycling",
    "HKWorkoutActivityTypeSwimming": "Swimming",
    "HKWorkoutActivityTypeTraditionalStrengthTraining": "Strength Training",
    "HKWorkoutActivityTypeHighIntensityIntervalTraining": "HIIT",
    "37": "Running", "38": "Running", "54": "Walking",
    "14": "Cycling", "48": "Swimming", "52": "Strength Training", "58": "Yoga",
}

def _map_workout(code: str) -> str:
    if not code:
        return ""
    if code in _WORKOUT_MAP:
        return _WORKOUT_MAP[code]
    for k, v in _WORKOUT_MAP.items():
        if k.lower() in code.lower():
            return v
    return code


# ── 串流解析 Apple Health export.xml ────────────────────────────────────────
# iterparse 逐筆掃描，只維護 daily bucket，每讀完一筆立刻 elem.clear()
# 記憶體用量從 ~500MB 降至 ~5MB（與 export.xml 大小無關）

_EXERCISE_HR_THRESHOLD = 90


def _stream_parse_xml(xml_source):
    """xml_source 可以是檔案路徑（str）或 file-like object，支援直接從 zip 串流讀取。"""
    sleep_buckets = defaultdict(float)
    hr_buckets = defaultdict(lambda: {"ex_sum": 0.0, "ex_cnt": 0, "max": 0.0})
    workout_buckets = defaultdict(lambda: {"types": set(), "minutes": 0.0, "distance": 0.0, "energy": 0.0})

    for _event, elem in ET.iterparse(xml_source, events=("end",)):
        tag = elem.tag

        if tag.endswith("Record"):
            rtype = elem.get("type", "")

            if rtype == "HKQuantityTypeIdentifierHeartRate":
                try:
                    val = float(elem.get("value", "nan"))
                    date = pd.to_datetime(elem.get("startDate"), utc=True).date()
                    b = hr_buckets[date]
                    if val > b["max"]:
                        b["max"] = val
                    if val >= _EXERCISE_HR_THRESHOLD:
                        b["ex_sum"] += val
                        b["ex_cnt"] += 1
                except Exception:
                    pass

            elif rtype == "HKCategoryTypeIdentifierSleepAnalysis":
                try:
                    start = pd.to_datetime(elem.get("startDate"), utc=True)
                    end = pd.to_datetime(elem.get("endDate"), utc=True)
                    sleep_buckets[end.date()] += (end - start).total_seconds() / 3600
                except Exception:
                    pass

        elif tag.endswith("Workout"):
            try:
                start = pd.to_datetime(elem.get("startDate"), utc=True)
                end = pd.to_datetime(elem.get("endDate"), utc=True)
                date = start.date()
                activity = _map_workout(elem.get("workoutActivityType") or elem.get("activityType") or "")
                b = workout_buckets[date]
                if activity:
                    b["types"].add(activity)
                b["minutes"] += (end - start).total_seconds() / 60
                b["distance"] += float(elem.get("totalDistance") or 0)
                b["energy"] += float(elem.get("totalEnergyBurned") or 0)
            except Exception:
                pass

        elem.clear()

    return sleep_buckets, hr_buckets, workout_buckets


def _buckets_to_df(sleep_buckets, hr_buckets, workout_buckets) -> pd.DataFrame:
    sleep_shifted = {date + timedelta(days=1): hours for date, hours in sleep_buckets.items()}

    hr_rows = [
        {"date": date, "mean_exercise_hr": b["ex_sum"] / b["ex_cnt"], "max_hr": b["max"]}
        for date, b in hr_buckets.items() if b["ex_cnt"] > 0
    ]
    if not hr_rows:
        raise ValueError("找不到足夠的運動心率紀錄（HR ≥ 90 BPM）。請確認 Apple Watch 有記錄運動數據。")

    hr_df = pd.DataFrame(hr_rows).set_index("date")
    hr_df.index = pd.to_datetime(hr_df.index)

    sleep_series = pd.Series(sleep_shifted, name="sleep_hours")
    sleep_series.index = pd.to_datetime(list(sleep_series.index))

    merged = hr_df.join(sleep_series, how="inner").dropna(subset=["sleep_hours"])
    if len(merged) == 0:
        raise ValueError("睡眠與運動紀錄沒有重疊的日期，請確認 Apple Watch 有同時記錄睡眠與運動。")

    w_rows = [{
        "date": pd.Timestamp(date),
        "workout_types": ";".join(sorted(b["types"])),
        "num_workouts": len(b["types"]),
        "total_workout_minutes": b["minutes"],
        "total_workout_distance": b["distance"],
        "total_workout_energy": b["energy"],
        "has_running": int("Running" in b["types"]),
        "has_walking": int("Walking" in b["types"]),
        "has_hiit": int("HIIT" in b["types"]),
    } for date, b in workout_buckets.items()]

    if w_rows:
        workout_df = pd.DataFrame(w_rows).set_index("date")
        merged = merged.join(workout_df, how="left")

    for col in ["workout_types"]:
        merged[col] = merged.get(col, pd.Series("", index=merged.index)).fillna("")
    for col in ["num_workouts", "has_running", "has_walking", "has_hiit"]:
        merged[col] = merged.get(col, pd.Series(0, index=merged.index)).fillna(0).astype(int)
    for col in ["total_workout_minutes", "total_workout_distance", "total_workout_energy"]:
        merged[col] = merged.get(col, pd.Series(0.0, index=merged.index)).fillna(0.0)

    return merged.rename_axis("date").reset_index()


def parse_zip_to_df(uploaded_file) -> pd.DataFrame:
    # io.BytesIO 讓 zipfile 直接從記憶體讀取，不需要寫到磁碟
    # z.open(member) 回傳 file-like object，ET.iterparse 直接串流讀取
    # → 完全跳過「解壓 1.8GB XML 到磁碟」這一步，節省大量時間與磁碟 IO
    zip_bytes = io.BytesIO(uploaded_file.read())

    xml_member = None
    with zipfile.ZipFile(zip_bytes, "r") as z:
        for member in z.namelist():
            if member.endswith("export.xml"):
                xml_member = member
                break

    if not xml_member:
        raise FileNotFoundError("在 zip 中找不到 export.xml，請確認是從 Apple Health 匯出的原始檔案。")

    zip_bytes.seek(0)
    with zipfile.ZipFile(zip_bytes, "r") as z:
        with z.open(xml_member) as xml_stream:
            sleep_buckets, hr_buckets, workout_buckets = _stream_parse_xml(xml_stream)

    return _buckets_to_df(sleep_buckets, hr_buckets, workout_buckets)


def train_personal_model(df: pd.DataFrame):
    # 延遲 import sklearn：只在使用者真的上傳資料時才載入
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from ml_healthy_detection_exercise_suggestion import prepare_features

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if len(X) >= 5 else (X, X, y, y)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), len(workout_days)


# ── 預測 UI ────────────────────────────────────────────────────────────────

def prediction_ui(model, model_label: str, key_prefix: str = "default"):
    # key_prefix 確保 Tab 1 / Tab 2 的 widget 有不同 element ID，避免 StreamlitDuplicateElementId
    st.markdown(f"**使用模型：{model_label}**")
    st.header("📥 輸入今日身體狀態與計畫")

    sleep_yesterday = st.slider("昨晚睡眠時數 (小時)", min_value=0.0, max_value=24.0, value=8.0, step=0.5, key=f"{key_prefix}_sleep_yesterday")
    sleep_3d_avg = st.slider("過去 3 天平均睡眠 (小時)", min_value=0.0, max_value=24.0, value=8.0, step=0.5, key=f"{key_prefix}_sleep_3d_avg")
    last_hr_mean = st.number_input("上一次運動的平均心率 (BPM)", min_value=60, max_value=200, value=105, key=f"{key_prefix}_hr_mean")
    last_hr_max = st.number_input("上一次運動的最大心率 (BPM)", min_value=80, max_value=220, value=135, key=f"{key_prefix}_hr_max")
    workout_type = st.selectbox("預計今日運動項目", ["重量訓練 (Strength Training)", "戶外/室內健走 (Walking)", "跑步 (Running)", "其他常規運動"], key=f"{key_prefix}_workout_type")
    planned_duration = st.slider("預計運動時間 (分鐘)", min_value=10, max_value=240, value=60, step=5, key=f"{key_prefix}_duration")

    planned_distance = 0.0
    if workout_type in ["戶外/室內健走 (Walking)", "跑步 (Running)"]:
        planned_distance = st.number_input("預計運動距離 (公里)", min_value=0.0, max_value=100.0, value=5.0, step=0.5, format="%.1f", key=f"{key_prefix}_distance")

    planned_intensity = st.selectbox("預計訓練強度", ["低強度", "中等強度", "高強度"], key=f"{key_prefix}_intensity")

    run_mode = None
    if "Running" in workout_type:
        run_mode = st.selectbox("跑步訓練類型", ["一般跑步", "輕鬆跑", "長距離", "間歇"], key=f"{key_prefix}_run_mode")

    is_strength = 1 if "Strength" in workout_type else 0
    is_walking = 1 if "Walking" in workout_type else 0
    is_running = 1 if "Running" in workout_type else 0
    planned_energy = {"低強度": 80, "中等強度": 180, "高強度": 320}[planned_intensity]
    has_hiit = 1 if planned_intensity == "高強度" and "Running" in workout_type else 0

    input_data = np.array([[
        sleep_yesterday, sleep_3d_avg, last_hr_mean, last_hr_max,
        is_strength, is_walking, is_running,
        1, planned_duration, planned_distance, planned_energy,
        is_running, is_walking, has_hiit,
    ]])

    if st.button("🚀 開始智慧身體負荷評估", key=f"{key_prefix}_submit"):
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

with tab1:
    st.markdown("使用示範模型（基於作者本人的 Apple Watch 數據），可快速體驗功能，但準確度因人而異。")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "exercise_hr_rf_model.pkl")

    @st.cache_resource
    def load_default_model():
        return joblib.load(MODEL_PATH)

    try:
        default_model = load_default_model()
        prediction_ui(default_model, "示範模型（作者數據）", key_prefix="tab1")
    except Exception as exc:
        st.error(f"無法載入示範模型：{exc}")

with tab2:
    st.markdown("""
    **如何取得 Apple Health 匯出檔案：**
    1. 打開 iPhone 的「健康」App
    2. 右上角點選頭像 → 向下捲動 → **匯出所有健康數據**
    3. 等待打包完成後，將 `export.zip` 傳到電腦並上傳至此

    > 您的資料只在本次連線中處理，模型訓練完成後原始資料即丟棄，不會儲存。
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
                st.warning("⚠️ 模型解釋力較低（R² < 0.3），預測結果僅供參考。")
            st.divider()
            prediction_ui(st.session_state["personal_model"], "您的個人化模型", key_prefix="tab2")
