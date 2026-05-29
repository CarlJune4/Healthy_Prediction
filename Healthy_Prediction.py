import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
import zipfile
from typing import Optional
import csv
import bisect
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_PLOTTING = True
except Exception:
    HAVE_PLOTTING = False


def find_export_xml_in_zip(zip_path: str) -> Optional[str]:
    # If a zip file exists, extract and return the internal member path
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                if member.endswith("export.xml"):
                    z.extract(member, ".")
                    return member

    # Fallback: if user already extracted to apple_health_export/export.xml, return that path
    # check current working directory
    local_path = os.path.join("apple_health_export", "export.xml")
    if os.path.exists(local_path):
        return local_path

    # check relative to this script's directory (e.g. ./Healthy_Prediction/apple_health_export/export.xml)
    script_dir = os.path.dirname(__file__)
    script_local = os.path.join(script_dir, "apple_health_export", "export.xml")
    if os.path.exists(script_local):
        return script_local

    # as a last resort, search for any export.xml under the script directory
    for root, dirs, files in os.walk(script_dir):
        if "export.xml" in files:
            return os.path.join(root, "export.xml")

    return None


def load_records_from_xml(xml_path: str) -> pd.DataFrame:
    # Stream-parse XML to avoid loading entire file into memory.
    records = []
    # keep HR, sleep, and common exercise-related record types for richer CSV generation
    keep_types = {
        "HKQuantityTypeIdentifierHeartRate",
        "HKCategoryTypeIdentifierSleepAnalysis",
        "HKQuantityTypeIdentifierActiveEnergyBurned",
        "HKQuantityTypeIdentifierDistanceWalkingRunning",
        "HKQuantityTypeIdentifierDistanceCycling",
        "HKQuantityTypeIdentifierStepCount",
        "HKQuantityTypeIdentifierFlightsClimbed",
        "HKQuantityTypeIdentifierRestingHeartRate",
        "HKQuantityTypeIdentifierWalkingHeartRateAverage",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        "HKQuantityTypeIdentifierVO2Max",
    }
    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag.endswith("Record"):
            t = elem.get("type")
            if t in keep_types:
                records.append({
                    "type": t,
                    "value": elem.get("value"),
                    "unit": elem.get("unit"),
                    "startDate": elem.get("startDate"),
                    "endDate": elem.get("endDate"),
                    "sourceName": elem.get("sourceName"),
                    "sourceVersion": elem.get("sourceVersion"),
                    "device": elem.get("device"),
                    "metadata": elem.get("metadata"),
                })
            # clear the element to save memory
            elem.clear()

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # keep raw value as string and also provide a numeric conversion when applicable
    df["raw_value"] = df["value"].astype(object)

    # parse datetimes as UTC-aware timestamps to avoid tz-naive/tz-aware comparisons
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    df["endDate"] = pd.to_datetime(df["endDate"], errors="coerce", utc=True)

    # numeric conversion stored separately so categorical values (e.g., sleep) are preserved
    df["numeric_value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def load_workouts_from_xml(xml_path: str) -> pd.DataFrame:
    """Stream-parse workouts from export.xml and return DataFrame with start/end and activity type."""
    workouts = []
    for event, elem in ET.iterparse(xml_path, events=("end",)):
        # Apple Health uses <Workout ... /> elements
        if elem.tag.endswith("Workout"):
            activity = elem.get("workoutActivityType") or elem.get("activityType") or elem.get("workoutActivityTypeCode") or elem.get("activity")
            workouts.append({
                "startDate": elem.get("startDate"),
                "endDate": elem.get("endDate"),
                "activity": activity,
                "sourceName": elem.get("sourceName"),
                "sourceVersion": elem.get("sourceVersion"),
                "device": elem.get("device"),
                "totalDistance": elem.get("totalDistance"),
                "totalDistanceUnit": elem.get("totalDistanceUnit"),
                "totalEnergyBurned": elem.get("totalEnergyBurned"),
                "totalEnergyBurnedUnit": elem.get("totalEnergyBurnedUnit"),
                "duration": elem.get("duration"),
                "metadata": elem.get("metadata"),
            })
            elem.clear()

    wdf = pd.DataFrame(workouts)
    if wdf.empty:
        return wdf
    wdf["startDate"] = pd.to_datetime(wdf["startDate"], errors="coerce", utc=True)
    wdf["endDate"] = pd.to_datetime(wdf["endDate"], errors="coerce", utc=True)
    wdf["duration_minutes"] = (wdf["endDate"] - wdf["startDate"]).dt.total_seconds() / 60
    wdf["totalDistance"] = pd.to_numeric(wdf["totalDistance"], errors="coerce")
    wdf["totalEnergyBurned"] = pd.to_numeric(wdf["totalEnergyBurned"], errors="coerce")
    return wdf


_WORKOUT_ACTIVITY_MAP = {
    # common Apple workoutActivityType numeric codes mapped to names (partial)
    "HKWorkoutActivityTypeRunning": "Running",
    "HKWorkoutActivityTypeWalking": "Walking",
    "HKWorkoutActivityTypeCycling": "Cycling",
    "HKWorkoutActivityTypeSwimming": "Swimming",
    "HKWorkoutActivityTypeTraditionalStrengthTraining": "Strength Training",
    "HKWorkoutActivityTypeHighIntensityIntervalTraining": "HIIT",
    # numeric codes often appear as integers in exports; include common numbers as strings
    "37": "Running",
    "38": "Running",
    "54": "Walking",
    "14": "Cycling",
    "48": "Swimming",
    "52": "Strength Training",
    "58": "Yoga",
}


def map_workout_activity(code: str) -> str:
    if code is None:
        return ""
    code = str(code)
    # try direct mapping
    if code in _WORKOUT_ACTIVITY_MAP:
        return _WORKOUT_ACTIVITY_MAP[code]
    # sometimes code is like 'HKWorkoutActivityTypeRunning' or numeric; fallback
    for k, v in _WORKOUT_ACTIVITY_MAP.items():
        if k.lower() in code.lower():
            return v
    return code


def summarize_sleep(sleep_df: pd.DataFrame) -> None:
    if sleep_df.empty:
        print("睡眠資料為空。無法產生睡眠摘要。")
        return

    # determine tz of startDate series (may be None)
    tz = getattr(sleep_df["startDate"].dt, "tz", None)
    if tz is None:
        two_days_ago = pd.Timestamp.now() - pd.Timedelta(days=2)
    else:
        two_days_ago = pd.Timestamp.now(tz=tz) - pd.Timedelta(days=2)

    recent_sleep = sleep_df[sleep_df["startDate"] > two_days_ago]
    if recent_sleep.empty:
        print("最近兩天沒有睡眠紀錄。")
        return

    # group by the raw value (preserves categorical labels)
    sleep_summary = recent_sleep.groupby("raw_value").apply(
        lambda x: (x["endDate"] - x["startDate"]).sum()
    )

    print("睡眠分佈：")
    for k, v in sleep_summary.items():
        # v is a Timedelta
        hours = v.total_seconds() / 3600
        print(f"  {k}: {hours:.2f} 小時")


def summarize_heart_rate(hr_df: pd.DataFrame) -> None:
    tz = getattr(hr_df["startDate"].dt, "tz", None)
    if tz is None:
        today = pd.Timestamp.now().date()
    else:
        today = pd.Timestamp.now(tz=tz).date()
    today_hr = hr_df[hr_df["startDate"].dt.date == today]
    if today_hr.empty:
        print("今天沒有心率記錄。")
        return

    mean_hr = today_hr["numeric_value"].mean()
    max_hr = today_hr["numeric_value"].max()
    mean_str = f"{mean_hr:.1f}" if pd.notna(mean_hr) else "N/A"
    max_str = f"{int(max_hr)}" if pd.notna(max_hr) else "N/A"
    print(f"\n今天心率統計：\n平均 {mean_str} bpm | 最高 {max_str} bpm")


def compute_sleep_scores(sleep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute sleep duration per sleep_date (use endDate.date as the sleep date).

    Returns DataFrame with index 'sleep_date' (datetime.date) and column 'sleep_hours'.
    """
    if sleep_df.empty:
        return pd.DataFrame(columns=["sleep_date", "sleep_hours"]).set_index("sleep_date")

    # Use endDate.date as the representative date for the sleep session
    sleep_df = sleep_df.copy()
    sleep_df["sleep_date"] = sleep_df["endDate"].dt.date
    sleep_df["duration_hr"] = (sleep_df["endDate"] - sleep_df["startDate"]).dt.total_seconds() / 3600

    # sum durations per sleep_date
    agg = sleep_df.groupby("sleep_date")["duration_hr"].sum().rename("sleep_hours").to_frame()
    return agg


def compute_exercise_metrics(hr_df: pd.DataFrame, activity_threshold: int = 90) -> pd.DataFrame:
    """Compute daily exercise heart-rate metrics.

    - activity_threshold: bpm threshold to consider a HR sample as 'exercise'
    Returns DataFrame indexed by 'date' with columns: mean_exercise_hr, max_hr, samples
    """
    if hr_df.empty:
        return pd.DataFrame(columns=["date", "mean_exercise_hr", "max_hr", "samples"]).set_index("date")

    df = hr_df.copy()
    df["date"] = df["startDate"].dt.date

    # samples considered exercise when numeric_value >= threshold
    exercise_samples = df[df["numeric_value"] >= activity_threshold]

    mean_ex = exercise_samples.groupby("date")["numeric_value"].mean().rename("mean_exercise_hr")
    max_all = df.groupby("date")["numeric_value"].max().rename("max_hr")
    samples = exercise_samples.groupby("date")["numeric_value"].count().rename("samples")

    result = pd.concat([mean_ex, max_all, samples], axis=1).fillna(0)
    return result


def label_hr_samples_with_workouts(hr_df: pd.DataFrame, wdf: pd.DataFrame) -> pd.DataFrame:
    """Label each heart-rate sample with workout activity names if the sample falls within any workout interval.

    Returns copy of hr_df with new column `workout_type` ('' if none, or semicolon-separated names).
    """
    hr = hr_df.copy()
    hr["workout_type"] = ""
    if hr.empty or wdf.empty:
        return hr

    # Sort heart-rate samples by timestamp to allow efficient interval assignment
    hr_sorted = hr.sort_values("startDate").reset_index()
    # keep original index to restore order later
    hr_sorted = hr_sorted.rename(columns={"index": "orig_index"})

    times = hr_sorted["startDate"]
    try:
        import numpy as _np
    except Exception:
        _np = None

    for _, w in wdf.iterrows():
        start = w.get("startDate")
        end = w.get("endDate")
        activity = map_workout_activity(w.get("activity"))
        if pd.isna(start) or pd.isna(end) or not activity:
            continue
        # find slice indices using searchsorted on the sorted timestamps
        left = times.searchsorted(start)
        right = times.searchsorted(end, side="right")
        if right <= left:
            continue
        if _np is not None:
            existing = hr_sorted.loc[left:right - 1, "workout_type"].astype(str).values
            hr_sorted.loc[left:right - 1, "workout_type"] = _np.where(existing == "", activity, existing + ";" + activity)
        else:
            for idx in hr_sorted.loc[left:right - 1].index:
                cur = hr_sorted.at[idx, "workout_type"]
                hr_sorted.at[idx, "workout_type"] = activity if not cur else cur + ";" + activity

    # restore original order
    hr_out = hr_sorted.set_index("orig_index").sort_index()
    hr_out.index.name = hr_df.index.name
    return hr_out


def stream_label_hr_samples_to_csv(xml_path: str, out_csv: str, wdf: pd.DataFrame) -> None:
    """Stream-parse export.xml and write HR samples with workout_type to CSV.

    This avoids loading all HR rows into memory.
    """
    # prepare intervals
    intervals = []
    if not wdf.empty:
        for _, r in wdf.iterrows():
            s = r.get("startDate")
            e = r.get("endDate")
            name = map_workout_activity(r.get("activity"))
            if pd.isna(s) or pd.isna(e) or not name:
                continue
            intervals.append((s.to_datetime64() if hasattr(s, 'to_datetime64') else s, e, name))

    # sort intervals by start
    intervals.sort(key=lambda x: x[0])
    starts = [iv[0] for iv in intervals]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["startDate", "endDate", "numeric_value", "unit", "sourceName", "workout_type"]) 

        for event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag.endswith("Record") and elem.get("type") == "HKQuantityTypeIdentifierHeartRate":
                s = elem.get("startDate")
                e = elem.get("endDate")
                val = elem.get("value")
                unit = elem.get("unit")
                src = elem.get("sourceName")
                # parse start datetime
                try:
                    sd = pd.to_datetime(s, errors="coerce", utc=True)
                except Exception:
                    sd = None

                workout_names = []
                if sd is not None and intervals:
                    # narrow by start times
                    # convert sd to comparable type (Timestamp)
                    # find candidate intervals where start <= sd
                    i = bisect.bisect_right(starts, sd)
                    # check candidates' end >= sd
                    for cand in intervals[:i]:
                        if cand[1] >= sd:
                            workout_names.append(cand[2])

                writer.writerow([s, e, val, unit, src, ";".join(sorted(set(workout_names)))])
                elem.clear()


def correlate_sleep_vs_exercise(sleep_scores: pd.DataFrame, exercise_metrics: pd.DataFrame, sleep_shift_days: int = 0) -> dict:
    """Join sleep_scores and exercise_metrics by date and compute Pearson correlation.

    If `sleep_shift_days` > 0, shift sleep dates forward by that many days (useful to map
    previous-night sleep to next-day exercise performance).

    Returns dict with correlation results and merged DataFrame.
    """
    if sleep_scores.empty or exercise_metrics.empty:
        return {"n": 0, "pearson": None, "merged": pd.DataFrame()}

    # prepare sleep dataframe: ensure it has a column for date
    s = sleep_scores.reset_index().rename(columns={"sleep_date": "sleep_date", "sleep_hours": "sleep_hours"})

    # shift sleep date forward to represent previous-night -> next-day mapping
    if sleep_shift_days:
        s["date"] = pd.to_datetime(s["sleep_date"]) + pd.Timedelta(days=sleep_shift_days)
    else:
        s["date"] = pd.to_datetime(s["sleep_date"]) 
    s["date"] = s["date"].dt.date
    s = s.set_index("date")["sleep_hours"]

    e = exercise_metrics.copy()

    merged = s.to_frame().join(e, how="inner")
    if "samples" in merged.columns:
        merged = merged[merged["samples"] > 0]
    n = len(merged)
    if n == 0:
        return {"n": 0, "pearson": None, "merged": merged}

    pearson = merged["sleep_hours"].corr(merged["mean_exercise_hr"])
    return {"n": n, "pearson": pearson, "merged": merged}


def compute_spearman(merged: pd.DataFrame) -> Optional[float]:
    if merged.empty:
        return None
    return merged["sleep_hours"].corr(merged["mean_exercise_hr"], method="spearman")


def plot_sleep_vs_exercise(merged: pd.DataFrame, out_png: str, pearson: Optional[float] = None, spearman: Optional[float] = None) -> None:
    if merged.empty:
        return
    if not HAVE_PLOTTING:
        print("matplotlib 或 numpy 未安裝，無法輸出散佈圖。請安裝 matplotlib, numpy。")
        return

    x = merged["sleep_hours"].values
    y = merged["mean_exercise_hr"].values

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6, s=10)
    # fit simple linear trend for visualization
    try:
        coeff = np.polyfit(x, y, 1)
        poly = np.poly1d(coeff)
        xs = np.linspace(min(x), max(x), 100)
        plt.plot(xs, poly(xs), color="red", linewidth=1)
    except Exception:
        pass

    plt.xlabel("前一晚睡眠時數 (hours)")
    plt.ylabel("隔天平均運動心率 (bpm)")
    title = "睡眠 vs 隔天運動心率"
    stats = []
    if pearson is not None:
        stats.append(f"Pearson={pearson:.3f}")
    if spearman is not None:
        stats.append(f"Spearman={spearman:.3f}")
    if stats:
        title += " — " + ", ".join(stats)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main(zip_path: str = "export.zip"):
    xml_member = find_export_xml_in_zip(zip_path)
    if xml_member is None:
        print(f"找不到 {zip_path} 或其中的 export.xml，請確認檔案存在。")
        sys.exit(1)
    # xml_member may be a local path (apple_health_export/export.xml) or a member path extracted to '.'
    if os.path.exists(xml_member):
        xml_path = xml_member
    else:
        xml_path = os.path.join(".", xml_member)
    df = load_records_from_xml(xml_path)
    if df.empty:
        print("未從 export.xml 讀到任何紀錄。請確認檔案內容。")
        return

    # 篩出心率與睡眠
    hr = df[df["type"] == "HKQuantityTypeIdentifierHeartRate"].copy()
    sleep = df[df["type"] == "HKCategoryTypeIdentifierSleepAnalysis"].copy()

    summarize_sleep(sleep)
    summarize_heart_rate(hr)

    # 新增：計算睡眠分數與運動心率指標，並做相關性分析
    sleep_scores = compute_sleep_scores(sleep)
    exercise_metrics = compute_exercise_metrics(hr)
    # 把「前一晚睡眠」對應到「隔天運動表現」進行相關性分析
    corr = correlate_sleep_vs_exercise(sleep_scores, exercise_metrics, sleep_shift_days=1)

    if corr["n"] == 0 or corr["pearson"] is None:
        print("無法計算相關性（缺乏重疊日期或樣本）。")
    else:
        print(f"\n樣本數: {corr['n']}，前一晚睡眠時數 vs 隔天平均運動心率 Pearson r = {corr['pearson']:.3f}")
        out_dir = os.path.dirname(xml_path)
        # attach workouts: parse workouts and aggregate types per date
        wdf = load_workouts_from_xml(xml_path)
        # per-sample labeling: stream and write each HR sample with workout_type
        out_hr_csv = os.path.join(out_dir, "hr_samples_with_workout_types.csv")
        try:
            stream_label_hr_samples_to_csv(xml_path, out_hr_csv, wdf)
            print(f"每筆心率樣本（含 workout_type）已寫入：{out_hr_csv}")
        except Exception as ex:
            print("串流寫出每筆樣本失敗，改回批次方式（較慢）", ex)
            hr_labeled = label_hr_samples_with_workouts(hr, wdf)
            hr_labeled.to_csv(out_hr_csv, index=False)
            print(f"每筆心率樣本（含 workout_type）已寫入（批次）：{out_hr_csv}")

        # also aggregate workout types and summary metrics per date for the merged summary
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
        else:
            workout_daily = pd.DataFrame(
                columns=["workout_types", "num_workouts", "total_workout_minutes", "total_workout_distance", "total_workout_energy", "has_running", "has_walking", "has_hiit"]
            ).astype({
                "workout_types": str,
                "num_workouts": int,
                "total_workout_minutes": float,
                "total_workout_distance": float,
                "total_workout_energy": float,
                "has_running": int,
                "has_walking": int,
                "has_hiit": int,
            })

        merged = corr["merged"].copy()
        merged = merged.join(workout_daily, how="left")
        merged["workout_types"] = merged["workout_types"].fillna("")
        merged["num_workouts"] = merged["num_workouts"].fillna(0).astype(int)
        merged["total_workout_minutes"] = merged["total_workout_minutes"].fillna(0.0)
        merged["total_workout_distance"] = merged["total_workout_distance"].fillna(0.0)
        merged["total_workout_energy"] = merged["total_workout_energy"].fillna(0.0)
        merged["has_running"] = merged["has_running"].fillna(0).astype(int)
        merged["has_walking"] = merged["has_walking"].fillna(0).astype(int)
        merged["has_hiit"] = merged["has_hiit"].fillna(0).astype(int)

        out_csv = os.path.join(out_dir, "sleep_hr_correlation_prevnight_to_nextday_with_workouts.csv")
        merged.to_csv(out_csv)
        print(f"合併資料（含 workout 類型）已寫入：{out_csv}")

        # compute Spearman and plot scatter
        spearman_r = compute_spearman(merged)
        print(f"Spearman r = {spearman_r:.3f}" if spearman_r is not None else "Spearman 無法計算")
        out_png = os.path.join(out_dir, "sleep_hr_scatter_prevnight_to_nextday.png")
        plot_sleep_vs_exercise(merged, out_png, pearson=corr["pearson"], spearman=spearman_r)
        print(f"散佈圖已寫入：{out_png}")


if __name__ == "__main__":
    main()