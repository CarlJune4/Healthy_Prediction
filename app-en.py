import os
import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="AI Exercise Load & Smart Recommendation", page_icon="⌚")

st.title("⌚ AI Exercise Load Prediction and Smart Recommendation System")
st.markdown("This system uses your historical Apple Watch data and a Random Forest model to estimate your exercise load and provide personalized guidance.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'exercise_hr_rf_model.pkl')

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

if model is not None:
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        st.header("📥 Enter Today's Body Status and Plan")

        sleep_yesterday = st.slider("Last night sleep hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
        sleep_3d_avg = st.slider("3-day average sleep hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)

        last_hr_mean = st.number_input("Previous workout average heart rate (BPM)", min_value=60, max_value=200, value=105)
        last_hr_max = st.number_input("Previous workout max heart rate (BPM)", min_value=80, max_value=220, value=135)

        workout_type = st.selectbox(
            "Planned exercise type",
            ["Strength Training", "Walking", "Running", "Other Regular Exercise"]
        )

        planned_duration = st.slider("Planned exercise duration (minutes)", min_value=10, max_value=240, value=60, step=5)
        if workout_type in ["Walking", "Running"]:
            planned_distance = st.number_input(
                "Planned exercise distance (km)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.5,
                format="%.1f"
            )
        else:
            planned_distance = 0.0
        planned_intensity = st.selectbox(
            "Planned training intensity",
            ["Low", "Moderate", "High"]
        )

        run_mode = None
        if workout_type == "Running":
            run_mode = st.selectbox("Running training type", ["General Run", "Easy Run", "Long Run", "Intervals"])

        is_strength = 1 if workout_type == "Strength Training" else 0
        is_walking = 1 if workout_type == "Walking" else 0
        is_running = 1 if workout_type == "Running" else 0

        planned_energy = 80 if planned_intensity == "Low" else 180 if planned_intensity == "Moderate" else 320
        has_hiit = 1 if planned_intensity == "High" and workout_type == "Running" else 0
        num_workouts = 1

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

        if st.button("🚀 Start Smart Exercise Load Evaluation"):
            predicted_hr = model.predict(input_data)[0]

            st.subheader("📊 AI Prediction and Evaluation")
            st.metric(label="Estimated average exercise heart rate", value=f"{predicted_hr:.1f} BPM")

            st.markdown(f"**Training plan: {planned_duration} minutes / {planned_distance:.1f} km / {planned_intensity} intensity**")
            if run_mode:
                st.markdown(f"**Running mode: {run_mode}**")

            st.subheader("💡 Smart Exercise and Health Suggestions")

            if sleep_yesterday < 6.0 or sleep_3d_avg < 6.5:
                st.warning("⚠️ Sleep debt warning: Your recent sleep duration is clearly insufficient.")
                if predicted_hr > 115:
                    st.error("🚨 High cardiopulmonary risk warning: The AI predicts a high exercise heart rate today. With insufficient sleep, high-intensity exercise greatly increases cardiac strain and injury risk. Consider switching to low-intensity walking or stretching, or cut volume by 50%.")
                else:
                    st.info("💡 Although the predicted heart rate is within normal range, your current recovery is weak. Focus on active recovery and avoid pushing heavy loads.")
            else:
                st.success("✅ Recovery status looks good: Your recent sleep is sufficient and body recovery is strong.")
                if is_running or predicted_hr > 115:
                    if run_mode == "Easy Run":
                        st.success("🔥 Suitable for an easy run: Today is good for recovery pace. Maintain a steady rhythm.")
                    elif run_mode == "Long Run":
                        st.success("🔥 Suitable for long-distance training: Keep your pace controlled and steady. A slightly lower average heart rate can be normal.")
                    elif run_mode == "Intervals":
                        st.success("🔥 Suitable for interval training: Average heart rate does not reflect sprint peaks. Use alternating intensity.")
                    else:
                        st.success("🔥 Suitable for high-intensity training: Your body status is strong today, and you can push cardiovascular or aerobic training.")
                else:
                    if run_mode == "Easy Run":
                        st.info("💪 Today is suitable for an easy run. Keep a relaxed pace and breathing.")
                    elif run_mode == "Long Run":
                        st.info("💪 Today is suitable for long endurance running. Start conservatively and avoid going too fast.")
                    elif run_mode == "Intervals":
                        st.info("💡 Ensure a good warm-up for interval training. A lower average heart rate does not mean the workout is too easy.")
                    else:
                        st.info("💪 Your physical condition is excellent today; a regular strength or light exercise session should feel easy.")

# Run with: streamlit run app-en.py
