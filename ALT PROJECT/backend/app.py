import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configure the web page
st.set_page_config(page_title="Machine Health Dashboard", page_icon="🏭", layout="wide")

# Simple, descriptive names for the machine types
MACHINE_TYPES = {
    "L": "Low Quality (L-Type) - Standard tools, wears out faster",
    "M": "Medium Quality (M-Type) - Good balance of durability",
    "H": "High Quality (H-Type) - Heavy duty, lasts the longest",
}

# Ensure this path matches where your CSV is saved
DATA_PATH = r"C:\Users\Vivek Akhil\Downloads\ai 2020.csv"
API_URL = "http://127.0.0.1:8000/predict"

@st.cache_data
def load_and_age_data(selected_year: int):
    # Load the original 2020 dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Map the simple descriptive names
    df["Machine Name"] = df["Type"].map(lambda t: MACHINE_TYPES[t].split(" - ")[0])

    # Calculate how many years have passed since the 2020 baseline
    years_passed = max(0, selected_year - 2020)

    # Simulate aging: tools wear out faster, machines run hotter, and require more torque
    wear_increase = 1.0 + (years_passed * 0.08)
    temp_increase = years_passed * 0.45
    torque_increase = 1.0 + (years_passed * 0.02)

    df["Tool wear [min]"] = np.clip(df["Tool wear [min]"] * wear_increase, 0, 320)
    df["Process temperature [K]"] = df["Process temperature [K]"] + temp_increase
    df["Torque [Nm]"] = np.clip(df["Torque [Nm]"] * torque_increase, 10, 90)

    # Determine if the increased age and stress cause new failures
    extra_failures = (
        (df["Tool wear [min]"] > 215) | 
        ((df["Process temperature [K]"] - df["Air temperature [K]"] < 7.5)) | 
        (df["Torque [Nm]"] > 65)
    ).astype(int)

    # Combine original failures with new projected failures
    df["Machine failure"] = np.maximum(df["Machine failure"], extra_failures)
    
    return df

# ==========================================
# SIDEBAR: Simple User Controls
# ==========================================
st.sidebar.title("⚙️ Analysis Settings")
st.sidebar.markdown("**Step 1: Choose the year you want to analyze.**")

selected_year = st.sidebar.slider(
    "Select Year:",
    min_value=2020,
    max_value=2030,
    value=2020, # Defaults to 2020 as requested
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Step 2: Choose machine types to view.**")
selected_types = st.sidebar.multiselect(
    "Machine Types:",
    options=["Low Quality (L-Type)", "Medium Quality (M-Type)", "High Quality (H-Type)"],
    default=["Low Quality (L-Type)", "Medium Quality (M-Type)", "High Quality (H-Type)"]
)

# Load and filter data based on user selection
df_all = load_and_age_data(selected_year)
df_filtered = df_all[df_all["Machine Name"].isin(selected_types)]


# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
st.title(f"🏭 Factory Health Dashboard for {selected_year}")
st.markdown("Monitor your factory's overall equipment health and predict specific machine breakdowns.")

# Key Performance Indicators (KPIs)
total_machines = len(df_filtered)
broken_machines = int(df_filtered["Machine failure"].sum())
broken_percent = (broken_machines / total_machines * 100) if total_machines > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Machines Monitored", f"{total_machines:,}")
col2.metric("Expected Failures", f"{broken_machines}", f"{broken_percent:.1f}% of total")
col3.metric("Average Tool Wear", f"{df_filtered['Tool wear [min]'].mean():.0f} minutes")

st.markdown("---")

tab_overview, tab_predict = st.tabs(["📊 Factory Overview", "🔮 Predict a Machine's Health"])

# --- TAB 1: DYNAMIC DATA ANALYSIS ---
with tab_overview:
    st.subheader(f"Data Analysis & Trends for {selected_year}")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Dynamic Histogram for Tool Wear
        fig_wear = px.histogram(
            df_filtered, 
            x="Tool wear [min]", 
            color="Machine failure",
            title=f"Tool Wear Distribution in {selected_year}",
            labels={"Machine failure": "Is it broken?", "Tool wear [min]": "Tool Wear (minutes)"},
            color_discrete_map={0: "#10b981", 1: "#ef4444"} # Green for good, Red for broken
        )
        st.plotly_chart(fig_wear, use_container_width=True)

    with c2:
        # Dynamic Breakdown of Failure Types
        modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        friendly_modes = ["Tool Worn Out", "Overheated", "Power Issue", "Too Much Strain", "Random Break"]
        counts = [df_filtered[m].sum() for m in modes]
        
        fig_reasons = px.bar(
            x=friendly_modes, y=counts,
            title=f"Primary Causes of Machine Failure in {selected_year}",
            labels={"x": "Reason for Failure", "y": "Number of Machines"},
            color=counts, color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_reasons, use_container_width=True)

# --- TAB 2: LIVE PREDICTION ---
with tab_predict:
    st.subheader("Check a Single Machine's Status")
    st.markdown("Enter the machine's current sensor readings to see if it is safe to use or about to break.")
    
    form_col, result_col = st.columns(2)
    
    with form_col:
        machine_type = st.selectbox("Select Machine Type", ["L", "M", "H"], format_func=lambda x: MACHINE_TYPES[x])
        air_temp = st.number_input("Room Temperature (Kelvin)", value=298.5, step=0.1)
        proc_temp = st.number_input("Machine Temperature (Kelvin)", value=309.2, step=0.1)
        rpm = st.number_input("Speed (RPM)", value=1420.0, step=10.0)
        torque = st.number_input("Power/Torque (Nm)", value=52.0, step=1.0)
        tool_wear = st.number_input("How long has the tool been used? (minutes)", value=180.0, step=1.0)
        
        check_button = st.button("Check Machine Health", type="primary")
        
    with result_col:
        if check_button:
            # Package the data for the backend API
            payload = {
                "machine_type": machine_type,
                "air_temp": air_temp,
                "process_temp": proc_temp,
                "rpm": rpm,
                "torque": torque,
                "tool_wear": tool_wear
            }
            
            try:
                # Send request to FastAPI
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    
                    risk = result["risk_level"]
                    chance_to_fail = result["failure_probability"] * 100
                    time_left = result["estimated_rul_minutes"]
                    
                    # Display simple, clear alerts based on risk level
                    if risk == "CRITICAL":
                        st.error(f"🚨 **DANGER!** This machine has a **{chance_to_fail:.1f}%** chance of breaking. Stop it now!")
                    elif risk == "WARNING":
                        st.warning(f"⚠️ **WARNING.** This machine has a **{chance_to_fail:.1f}%** chance of breaking. Check it soon.")
                    else:
                        st.success(f"✅ **ALL GOOD.** Only a **{chance_to_fail:.1f}%** chance of breaking. Keep working!")
                        
                    st.info(f"⏳ **Estimated time before tool needs replacing:** {time_left:.0f} minutes remaining.")
                    
                    # Explainability: Show why it might fail in simple terms
                    st.markdown("### Top factors affecting this machine right now:")
                    factors = result["feature_importance"]
                    
                    friendly_factors = {
                        "Air temperature [K]": "Room Temperature",
                        "Process temperature [K]": "Machine Temperature",
                        "Rotational speed [rpm]": "Operating Speed",
                        "Torque [Nm]": "Torque/Power load",
                        "Tool wear [min]": "Tool Wear duration",
                        "Temp_Difference": "Heat Build-up inside the machine",
                        "Mechanical_Power": "Overall Mechanical Strain"
                    }
                    
                    for exact_name, friendly_name in friendly_factors.items():
                        if exact_name in factors:
                            score = factors[exact_name]
                            if score > 0.05: # Only show the highly impactful factors
                                st.write(f"- **{friendly_name}** is a major factor.")

                else:
                    st.error("Oops! The backend server returned an error.")
            except Exception as e:
                st.error("Could not connect to the prediction server. Make sure `uvicorn main:app --reload --port 8000` is running in another terminal.")
        else:
            st.write("👈 Fill out the form on the left and click the button to see the health results here.")