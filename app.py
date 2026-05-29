import os
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import timedelta
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from Healthy_Prediction import map_workout_activity
from ml_healthy_detection_exercise_suggestion import prepare_features

st.set_page_config(page_title="AI 運動負荷與智慧推薦系統", page_icon="⌚", layout="centered")
st.title("⌚ AI 運動生理負荷預測與智慧推薦系統")

# ── 串流解析 Apple Health export.xml ────────────────────────────────────────
#
# 傳統做法：把所有 HR + Sleep records 讀進 DataFrame（1.8GB XML → ~500MB RAM）
# 串流做法：iterparse 逐筆掃描，只維護 daily bucket（字典），每讀完一筆立刻 elem.clear()
#           掃完後記憶體只剩 ~365 個日期的彙總數字，降低約 100 倍記憶體用量
#
# daily bucket 結構：
#   sleep_buckets[date]   = total_sleep_hours (float)
#   hr_buckets[date]      = {'ex_sum': 0, 'ex_cnt': 0, 'max': 0}
#                             ex = exercise samples (HR ≥ 90 BPM)
#   workout_buckets[date] = {'types': set(), 'minutes': 0, 'distance': 0, 'energy': 0}

_EXERCISE_HR_THRESHOLD = 90  # BPM


def _stream_parse_xml(xml_path: str):
    """單次 iterparse 掃描，回傳三個 daily bucket 字典。"""
    sleep_buckets = defaultdict(float)
    hr_buckets = defaultdict(lambda: {"ex_sum": 0.0, "ex_cnt": 0, "max": 0.0})
    workout_buckets = defaultdict(lambda: {"types": set(), "minutes": 0.0, "distance": 0.0, "energy": 0.0})

    for _event, elem in ET.iterparse(xml_path, events=("end",)):
        tag = elem.tag

        if tag.endswith("Record"):
            rtype = elem.get("type", "")

            if rtype == "HKQuantityTypeIdentifierHeartRate":
                try:
                    val = float(elem.get("value", "nan"))
                    date = pd.to_datetime(elem.get("startDate"), utc=True).date()
                    bucket = hr_buckets[date]
                    if val > bucket["max"]:
                        bucket["max"] = val
                    if val >= _EXERCISE_HR_THRESHOLD:
                        bucket["ex_sum"] += val
                        bucket["ex_cnt"] += 1
                except Exception:
                    pass

            elif rtype == "HKCategoryTypeIdentifierSleepAnalysis":
                try:
                    start = pd.to_datetime(elem.get("startDate"), utc=True)
                    end = pd.to_datetime(elem.get("endDate"), utc=True)
                    duration_hr = (end - start).total_seconds() / 3600
                    # endDate.date() 代表這段睡眠所屬的那天（與 compute_sleep_scores 一致）
                    sleep_buckets[end.date()] += duration_hr
                except Exception:
                    pass

        elif tag.endswith("Workout"):
            try:
                start = pd.to_datetime(elem.get("startDate"), utc=True)
                end = pd.to_datetime(elem.get("endDate"), utc=True)
                date = start.date()
                activity = map_workout_activity(elem.get("workoutActivityType") or elem.get("activityType") or "")
                minutes = (end - start).total_seconds() / 60
                distance = float(elem.get("totalDistance") or 0)
                energy = float(elem.get("totalEnergyBurned") or 0)
                b = workout_buckets[date]
                if activity:
                    b["types"].add(activity)
                b["minutes"] += minutes
                b["distance"] += distance
                b["energy"] += energy
            except Exception:
                pass

        elem.clear()  # 立刻釋放 XML node，不讓記憶體累積

    return sleep_buckets, hr_buckets, workout_buckets


def _buckets_to_df(sleep_buckets, hr_buckets, workout_buckets) -> pd.DataFrame:
    """將三個 daily bucket 合併成模型訓練用的 DataFrame。"""
    # sleep: shift +1 天，讓前一晚睡眠對應隔天運動（與 correlate_sleep_vs_exercise 一致）
    sleep_shifted = {date + timedelta(days=1): hours for date, hours in sleep_buckets.items()}

    # HR metrics
    hr_rows = []
    for date, b in hr_buckets.items():
        mean_ex = b["ex_sum"] / b["ex_cnt"] if b["ex_cnt"] > 0 else float("nan")
        hr_rows.append({"date": date, "mean_exercise_hr": mean_ex, "max_hr": b["max"]})
    hr_df = pd.DataFrame(hr_rows).set_index("date") if hr_rows else pd.DataFrame(columns=["mean_exercise_hr", "max_hr"])

    # 只保留有運動心率紀錄的日期
    hr_df = hr_df.dropna(subset=["mean_exercise_hr"])
    if hr_df.empty:
        raise ValueError("找不到足夠的運動心率紀錄（HR ≥ 90 BPM）。請確認 Apple Watch 有記錄運動數據。")

    # sleep Series（對齊 HR 日期）
    sleep_series = pd.Series(sleep_shifted, name="sleep_hours")
    sleep_series.index = pd.to_datetime(list(sleep_series.index))
    hr_df.index = pd.to_datetime(hr_df.index)

    merged = hr_df.join(sleep_series, how="inner")
    merged = merged.dropna(subset=["sleep_hours"])

    if len(merged) == 0:
        raise ValueError("睡眠與運動紀錄沒有重疊的日期，請確認 Apple Watch 有同時記錄睡眠與運動。")

    # workout 欄位
    w_rows = []
    for date, b in workout_buckets.items():
        w_rows.append({
            "date": pd.Timestamp(date),
            "workout_types": ";".join(sorted(b["types"])),
            "num_workouts": len(b["types"]),
            "total_workout_minutes": b["minutes"],
            "total_workout_distance": b["distance"],
            "total_workout_energy": b["energy"],
            "has_running": int("Running" in b["types"]),
            "has_walking": int("Walking" in b["types"]),
            "has_hiit": int("HIIT" in b["types"]),
        })
    workout_df = pd.DataFrame(w_rows).set_index("date") if w_rows else pd.DataFrame()

    if not workout_df.empty:
        merged = merged.join(workout_df, how="left")

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

    merged = merged.rename_axis("date").reset_index()
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def parse_zip_to_df(uploaded_file) -> pd.DataFrame:
    """上傳的 export.zip → 串流解析 → 訓練用 DataFrame（低記憶體）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_bytes = uploaded_file.read()
        zip_path = os.path.join(tmpdir, "export.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        xml_path = None
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                if member.endswith("export.xml"):
                    z.extract(member, tmpdir)
                    xml_path = os.path.join(tmpdir, member)
                    break

        if xml_path is None or not os.path.exists(xml_path):
            raise FileNotFoundError("在 zip 中找不到 export.xml，請確認是從 Apple Health 匯出的原始檔案。")

        sleep_buckets, hr_buckets, workout_buckets = _stream_parse_xml(xml_path)
        return _buckets_to_df(sleep_buckets, hr_buckets, workout_buckets)

    # tmpdir 離開 with block 時自動刪除，export.xml 也一併清除

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
