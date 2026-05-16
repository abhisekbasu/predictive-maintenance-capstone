import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Engine Health Predictor",
    page_icon="🔧",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="abhisekbasu/predictive-maintenance-model",
        filename="best_engine_model_v1.joblib"
    )
    summary_path = hf_hub_download(
        repo_id="abhisekbasu/predictive-maintenance-model",
        filename="model_summary.json"
    )
    model = joblib.load(model_path)
    with open(summary_path) as f:
        summary = json.load(f)
    return model, summary

model, summary = load_model()

CAP_THRESHOLDS = {
    "Coolant_Temperature": {"upper": 91.783},
    "Lub_Oil_Pressure":    {"lower": 0.858},
    "Fuel_Pressure":       {"lower": 1.380},
    "Coolant_Pressure":    {"lower": 0.730},
}
LOG_COLS  = ["Fuel_Pressure", "Coolant_Pressure", "Lub_Oil_Temperature"]

def preprocess(df):
    df = df.copy()

    # Outlier capping
    df["Coolant_Temperature"] = df["Coolant_Temperature"].clip(
        upper=CAP_THRESHOLDS["Coolant_Temperature"]["upper"])
    df["Lub_Oil_Pressure"] = df["Lub_Oil_Pressure"].clip(
        lower=CAP_THRESHOLDS["Lub_Oil_Pressure"]["lower"])
    df["Fuel_Pressure"] = df["Fuel_Pressure"].clip(
        lower=CAP_THRESHOLDS["Fuel_Pressure"]["lower"])
    df["Coolant_Pressure"] = df["Coolant_Pressure"].clip(
        lower=CAP_THRESHOLDS["Coolant_Pressure"]["lower"])

    # Log1p transformation
    for col in LOG_COLS:
        df[col] = np.log1p(df[col])

    # Sqrt on Lub_Oil_Temperature
    df["Lub_Oil_Temperature"] = np.sqrt(df["Lub_Oil_Temperature"])

    # Feature engineering
    df["Pressure_Index"]      = df["Lub_Oil_Pressure"] + df["Fuel_Pressure"] + df["Coolant_Pressure"]
    df["Thermal_Index"]       = df["Lub_Oil_Temperature"] + df["Coolant_Temperature"]
    df["Pressure_Temp_Ratio"] = df["Pressure_Index"] / (df["Thermal_Index"] + 1e-9)
    df["RPM_Pressure_Ratio"]  = df["Engine_RPM"] / (df["Fuel_Pressure"] + 1e-9)
    df["Thermal_Deviation"]   = df["Coolant_Temperature"] - df["Lub_Oil_Temperature"]

    return df

st.title("🔧 Engine Health Predictor")
st.markdown(
    "Enter real-time engine sensor readings below to assess "
    "engine health and determine if maintenance is required."
)
st.divider()

# Sensor input form
col1, col2 = st.columns(2)

with col1:
    engine_rpm     = st.number_input("Engine RPM",              min_value=0.0,   max_value=3000.0, value=750.0,  step=10.0)
    lub_oil_press  = st.number_input("Lub Oil Pressure (bar)",  min_value=0.0,   max_value=10.0,   value=3.2,    step=0.1)
    fuel_press     = st.number_input("Fuel Pressure (bar)",     min_value=0.0,   max_value=25.0,   value=6.2,    step=0.1)

with col2:
    coolant_press  = st.number_input("Coolant Pressure (bar)",  min_value=0.0,   max_value=10.0,   value=2.2,    step=0.1)
    lub_oil_temp   = st.number_input("Lub Oil Temp (°C)",       min_value=60.0,  max_value=100.0,  value=77.0,   step=0.5)
    coolant_temp   = st.number_input("Coolant Temp (°C)",       min_value=50.0,  max_value=100.0,  value=78.0,   step=0.5)

st.divider()

if st.button("Predict Engine Health", type="primary", use_container_width=True):
    # Build input dataframe
    input_df = pd.DataFrame([{
        "Engine_RPM":           engine_rpm,
        "Lub_Oil_Pressure":     lub_oil_press,
        "Fuel_Pressure":        fuel_press,
        "Coolant_Pressure":     coolant_press,
        "Lub_Oil_Temperature":  lub_oil_temp,
        "Coolant_Temperature":  coolant_temp,
    }])

    # Preprocess
    processed = preprocess(input_df)

    # Predict
    prediction   = model.predict(processed)[0]
    probability  = model.predict_proba(processed)[0]
    confidence   = probability[prediction] * 100

    st.divider()

    if prediction == 0:
        st.success(f"Engine Status: NORMAL")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.markdown("The engine is operating within healthy parameters. No immediate maintenance required.")
    else:
        st.error(f"Engine Status: MAINTENANCE REQUIRED")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.markdown("The engine shows patterns consistent with developing failure. Schedule inspection immediately.")

    # Show engineered features
    with st.expander("View Engineered Health Indicators"):
        indicators = {
            "Pressure Index":      round(float(processed["Pressure_Index"].values[0]), 4),
            "Thermal Index":       round(float(processed["Thermal_Index"].values[0]), 4),
            "Pressure/Temp Ratio": round(float(processed["Pressure_Temp_Ratio"].values[0]), 4),
            "RPM/Pressure Ratio":  round(float(processed["RPM_Pressure_Ratio"].values[0]), 4),
            "Thermal Deviation":   round(float(processed["Thermal_Deviation"].values[0]), 4),
        }
        for k, v in indicators.items():
            st.metric(k, v)

st.divider()
st.caption(
    f"Model: Random Forest | "
    f"Recall: {summary['test_metrics']['recall']:.4f} | "
    f"ROC-AUC: {summary['test_metrics']['roc_auc']:.4f} | "
    f"Training records: {summary['training_records']:,}"
)
