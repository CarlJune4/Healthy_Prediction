import os
import streamlit as st
import numpy as np
import joblib

# 設定網頁標題與圖示
st.set_page_config(page_title="AI 運動負荷與智慧推薦系統", page_icon="⌚")

st.title("⌚ AI 運動生理負荷預測與智慧推薦系統")
st.markdown("本系統基於您的 Apple Watch 歷史數據，結合隨機森林（Random Forest）演算法，預估您今日運動的身體負荷並提供個人化建議。")

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'exercise_hr_rf_model.pkl')

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"無法載入模型：{exc}")
    st.stop()

if model is not None:
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        st.header("📥 輸入今日身體狀態與計畫")

        # 使用者互動輸入
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

        # 轉換成模型特徵
        is_strength = 1 if "Strength" in workout_type else 0
        is_walking = 1 if "Walking" in workout_type else 0
        is_running = 1 if "Running" in workout_type else 0

        # 將使用者計畫欄位映射為模型輸入特徵
        planned_energy = 80 if planned_intensity == "低強度" else 180 if planned_intensity == "中等強度" else 320
        has_hiit = 1 if planned_intensity == "高強度" and "Running" in workout_type else 0
        num_workouts = 1

        # 建立輸入陣列，請注意模型期望 14 個特徵
        input_data = np.array([[
            sleep_yesterday,
            sleep_3d_avg,
            last_hr_mean,
            last_hr_max,
            is_strength,
            is_walking,
            is_running,
            num_workouts,
            planned_duration,
            planned_distance,
            planned_energy,
            is_running,
            is_walking,
            has_hiit,
        ]])

        if st.button("🚀 開始智慧身體負荷評估"):
            # 進行預測
            predicted_hr = model.predict(input_data)[0]

            st.subheader("📊 AI 預測與評估結果")

            # 顯示預測心率
            st.metric(label="預估今日運動平均心率", value=f"{predicted_hr:.1f} BPM")

            st.markdown(f"**訓練計畫：{planned_duration} 分鐘 / {planned_distance:.1f} 公里 / {planned_intensity}**")
            if run_mode:
                st.markdown(f"**跑步訓練類型：{run_mode}**")

            # 核心智慧推薦邏輯 (Expert System Rules)
            st.subheader("💡 智慧運動與健康建議")

            # 根據睡眠與預測心率給予不同建議
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
                        st.success("🔥 **適合輕鬆跑**：今日適合進行恢復性的輕鬆跑，維持穩定配速即可。平均心率 119 BPM 表示可以以節奏跑為主。")
                    elif run_mode == "長距離":
                        st.success("🔥 **適合長距離訓練**：建議控制配速，保持耐力穩定。平均心率略低通常是合理現象。")
                    elif run_mode == "間歇":
                        st.success("🔥 **適合間歇訓練**：平均心率不代表衝刺階段的瞬時心率，請以高低強度交替為主。")
                    else:
                        st.success("🔥 **適合高強度訓練**：今日您的身體各項指標非常適合進行心肺或高強度有氧訓練，可以嘗試突破個人紀錄（PR）！")
                else:
                    if run_mode == "輕鬆跑":
                        st.info("💪 今日適合輕鬆跑。建議以穩定節奏、呼吸輕鬆為主。")
                    elif run_mode == "長距離":
                        st.info("💪 今日適合長距離耐力跑。建議以耐力配速為主，避免一開始跑得太快。")
                    elif run_mode == "間歇":
                        st.info("💡 今日間歇跑請確認熱身充分，平均心率較低並不代表強度不夠。")
                    else:
                        st.info("💪 今日身體狀態優異，完成常規的重量訓練或輕度運動將非常輕鬆。")

# 執行方式：在終端機輸入 `streamlit run app.py`