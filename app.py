# ============================================================
#  Predictive Maintenance — Unified Streamlit App
#  Supports two datasets:
#    1. Jet Engine  (AI4I 2020)
#    2. Naval Vessel (UCI Naval Propulsion)
#
#  Run with:  streamlit run app.py
#  Requires:
#    - model_artifacts.pkl        (from jet_engine.py)
#    - xgb_naval_model.pkl        (from naval_xgboost_failure.py)
# ============================================================

import time
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Predictive Maintenance Suite",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
#  HYBRID RISK SCORER
#  Combines XGBoost model probability with physics-based rules.
#  This ensures every slider produces a visible, meaningful change.
#  Formula:  final_prob = 0.4 * model_prob + 0.6 * rule_score
# ============================================================

def jet_rule_score(air_temp, process_temp, rpm, torque, tool_wear):
    """
    Physics-based failure risk for jet engine (0.0 to 1.0).
    Each parameter contributes independently; scores are weighted.
    Thresholds derived from AI4I 2020 dataset failure conditions.
    """
    scores = []

    # Air temperature — normal ~300 K, risk rises above 301 K
    scores.append(np.clip((air_temp - 298.0) / (305.0 - 298.0), 0, 1) ** 1.5)

    # Process temperature — normal ~310 K, risk rises above 311 K
    scores.append(np.clip((process_temp - 306.0) / (316.0 - 306.0), 0, 1) ** 1.5)

    # RPM — both very high AND very low are risky
    rpm_risk = abs(rpm - 1500) / (2886 - 1500)
    scores.append(np.clip(rpm_risk, 0, 1) ** 1.2)

    # Torque — mechanical stress, critical above 65 Nm
    scores.append(np.clip((torque - 20.0) / (76.6 - 20.0), 0, 1) ** 1.8)

    # Tool wear — linear degradation, critical at 200+ min
    scores.append(np.clip(tool_wear / 253.0, 0, 1) ** 1.2)

    weights = [0.15, 0.15, 0.15, 0.25, 0.30]
    return float(np.dot(scores, weights))


def naval_rule_score(inputs):
    """
    Physics-based failure risk for naval propulsion (0.0 to 1.0).
    Mirrors real GT propulsion degradation physics.
    """
    scores = []

    # Compressor decay kMc — 1.0=new, 0.95=critical lower bound
    kmc = inputs["GT_Compressor_decay_state_coefficient"]
    scores.append(np.clip((1.0 - kmc) / (1.0 - 0.950), 0, 1) ** 0.8)

    # Turbine decay kMt — 1.0=new, 0.975=critical lower bound
    kmt = inputs["GT_Turbine_decay_state_coefficient"]
    scores.append(np.clip((1.0 - kmt) / (1.0 - 0.975), 0, 1) ** 0.8)

    # HP Turbine exit temperature — normal 445-700 C, danger above 730 C
    t48 = inputs["Hight_Pressure_HP_Turbine_exit_temperature_T48_C"]
    scores.append(np.clip((t48 - 445.0) / (790.0 - 445.0), 0, 1) ** 2.0)

    # Ship speed — extreme speed increases mechanical stress
    vs = inputs["Ship_speed_v"]
    scores.append(np.clip((vs - 3) / (27 - 3), 0, 1) ** 2.5)

    # Fuel flow — high flow signals inefficiency or overload
    mf = inputs["Fuel_flow_mf_kg_s"]
    scores.append(np.clip((mf - 0.05) / (1.20 - 0.05), 0, 1) ** 1.5)

    # GT shaft torque — very high torque = mechanical overload
    gtt = inputs["Gas_Turbine_GT_shaft_torque_GTT_kN_m"]
    scores.append(np.clip((gtt - 300) / (70000 - 300), 0, 1) ** 1.5)

    # Compressor outlet pressure — high = compressor stress
    p2 = inputs["GT_Compressor_outlet_air_pressure_P2_bar"]
    scores.append(np.clip((p2 - 5.0) / (15.0 - 5.0), 0, 1) ** 1.8)

    # Turbine injection control — high TIC compensates for degradation
    tic = inputs["Turbine_Injecton_Control_TIC"]
    scores.append(np.clip((tic - 5.0) / (30.0 - 5.0), 0, 1) ** 1.2)

    # Decay coefficients weighted most heavily
    weights = [0.22, 0.22, 0.18, 0.08, 0.08, 0.08, 0.07, 0.07]
    return float(np.dot(scores, weights))


def hybrid_prob(model_prob, rule_score, model_weight=0.4):
    """Blend model output with rule-based score."""
    return float(np.clip(model_weight * model_prob + (1 - model_weight) * rule_score, 0.0, 1.0))


def risk_label_color(prob):
    if prob < 0.30:
        return "🟢 LOW",      "success"
    elif prob < 0.55:
        return "🟡 MEDIUM",   "warning"
    elif prob < 0.75:
        return "🟠 HIGH",     "error"
    else:
        return "🔴 CRITICAL", "error"


# ============================================================
#  ARTIFACT LOADERS
# ============================================================

@st.cache_resource
def load_jet_artifacts():
    try:
        with open("model_artifacts.pkl", "rb") as f:
            return pickle.load(f), None
    except FileNotFoundError:
        return None, "model_artifacts.pkl not found. Run jet_engine.py first."
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_naval_artifacts():
    try:
        with open("xgb_naval_model.pkl", "rb") as f:
            return pickle.load(f), None
    except FileNotFoundError:
        return None, "xgb_naval_model.pkl not found. Run naval_xgboost_failure.py first."
    except Exception as e:
        return None, str(e)


# ============================================================
#  NAVAL FEATURE DEFINITIONS
# ============================================================

NAVAL_DISPLAY_NAMES = {
    "Lever_position":                                      "Lever Position (LP)",
    "Ship_speed_v":                                        "Ship Speed (knots)",
    "Gas_Turbine_GT_shaft_torque_GTT_kN_m":               "GT Shaft Torque (kN·m)",
    "GT_rate_of_revolutions_GTn_rpm":                      "GT Revolutions (rpm)",
    "Gas_Generator_rate_of_revolutions_GGn_rpm":           "GG Revolutions (rpm)",
    "Starboard_Propeller_Torque_Ts_kN":                    "Starboard Prop Torque (kN)",
    "Port_Propeller_Torque_Tp_kN":                         "Port Prop Torque (kN)",
    "Hight_Pressure_HP_Turbine_exit_temperature_T48_C":    "HP Turbine Exit Temp (°C)",
    "GT_Compressor_inlet_air_temperature_T1_C":            "Compressor Inlet Temp (°C)",
    "GT_Compressor_outlet_air_temperature_T2_C":           "Compressor Outlet Temp (°C)",
    "HP_Turbine_exit_pressure_P48_bar":                    "HP Turbine Exit Pressure (bar)",
    "GT_Compressor_inlet_air_pressure_P1_bar":             "Compressor Inlet Pressure (bar)",
    "GT_Compressor_outlet_air_pressure_P2_bar":            "Compressor Outlet Pressure (bar)",
    "GT_exhaust_gas_pressure_Pexh_bar":                    "Exhaust Gas Pressure (bar)",
    "Turbine_Injecton_Control_TIC":                        "Turbine Injection Control (%)",
    "Fuel_flow_mf_kg_s":                                   "Fuel Flow (kg/s)",
    "GT_Compressor_decay_state_coefficient":               "★ Compressor Decay (kMc)",
    "GT_Turbine_decay_state_coefficient":                  "★ Turbine Decay (kMt)",
}

NAVAL_RANGES = {
    "Lever_position":                                      (1.0,   9.0,   5.0),
    "Ship_speed_v":                                        (3,     27,    15),
    "Gas_Turbine_GT_shaft_torque_GTT_kN_m":               (300,   70000, 20000),
    "GT_rate_of_revolutions_GTn_rpm":                      (1300,  2100,  1600),
    "Gas_Generator_rate_of_revolutions_GGn_rpm":           (6000,  10000, 8000),
    "Starboard_Propeller_Torque_Ts_kN":                    (5,     600,   180),
    "Port_Propeller_Torque_Tp_kN":                         (5,     600,   180),
    "Hight_Pressure_HP_Turbine_exit_temperature_T48_C":    (445,   790,   600),
    "GT_Compressor_inlet_air_temperature_T1_C":            (288,   303,   288),
    "GT_Compressor_outlet_air_temperature_T2_C":           (540,   680,   600),
    "HP_Turbine_exit_pressure_P48_bar":                    (1.0,   2.7,   1.8),
    "GT_Compressor_inlet_air_pressure_P1_bar":             (0.990, 1.00,  0.998),
    "GT_Compressor_outlet_air_pressure_P2_bar":            (5.0,   15.0,  9.0),
    "GT_exhaust_gas_pressure_Pexh_bar":                    (1.010, 1.040, 1.020),
    "Turbine_Injecton_Control_TIC":                        (5.0,   30.0,  15.0),
    "Fuel_flow_mf_kg_s":                                   (0.05,  1.20,  0.40),
    "GT_Compressor_decay_state_coefficient":               (0.950, 1.00,  0.975),
    "GT_Turbine_decay_state_coefficient":                  (0.975, 1.00,  0.990),
}

def naval_model_predict(model, row_dict):
    df_row = pd.DataFrame([row_dict])
    if hasattr(model, "feature_names_in_"):
        for c in model.feature_names_in_:
            if c not in df_row.columns:
                df_row[c] = 0.0
        df_row = df_row[model.feature_names_in_]
    return float(model.predict_proba(df_row)[0][1])


# ============================================================
#  NAVAL SIMULATION SCENARIOS
# ============================================================

NAVAL_SCENARIOS = {
    "Normal Cruise (15 kn)": {
        "Lever_position": (4.5, 5.5), "Ship_speed_v": (12, 18),
        "Gas_Turbine_GT_shaft_torque_GTT_kN_m": (15000, 25000),
        "GT_rate_of_revolutions_GTn_rpm": (1500, 1700),
        "Gas_Generator_rate_of_revolutions_GGn_rpm": (7500, 8500),
        "Starboard_Propeller_Torque_Ts_kN": (150, 220),
        "Port_Propeller_Torque_Tp_kN": (150, 220),
        "Hight_Pressure_HP_Turbine_exit_temperature_T48_C": (560, 630),
        "GT_Compressor_inlet_air_temperature_T1_C": (288, 292),
        "GT_Compressor_outlet_air_temperature_T2_C": (580, 625),
        "HP_Turbine_exit_pressure_P48_bar": (1.6, 1.9),
        "GT_Compressor_inlet_air_pressure_P1_bar": (0.997, 0.999),
        "GT_Compressor_outlet_air_pressure_P2_bar": (8.0, 10.0),
        "GT_exhaust_gas_pressure_Pexh_bar": (1.018, 1.022),
        "Turbine_Injecton_Control_TIC": (10.0, 15.0),
        "Fuel_flow_mf_kg_s": (0.28, 0.45),
        "GT_Compressor_decay_state_coefficient": (0.975, 1.000),
        "GT_Turbine_decay_state_coefficient": (0.988, 1.000),
    },
    "High Speed (27 kn)": {
        "Lever_position": (7.5, 9.0), "Ship_speed_v": (24, 27),
        "Gas_Turbine_GT_shaft_torque_GTT_kN_m": (55000, 70000),
        "GT_rate_of_revolutions_GTn_rpm": (1900, 2100),
        "Gas_Generator_rate_of_revolutions_GGn_rpm": (9000, 10000),
        "Starboard_Propeller_Torque_Ts_kN": (500, 600),
        "Port_Propeller_Torque_Tp_kN": (500, 600),
        "Hight_Pressure_HP_Turbine_exit_temperature_T48_C": (700, 790),
        "GT_Compressor_inlet_air_temperature_T1_C": (295, 303),
        "GT_Compressor_outlet_air_temperature_T2_C": (645, 680),
        "HP_Turbine_exit_pressure_P48_bar": (2.3, 2.7),
        "GT_Compressor_inlet_air_pressure_P1_bar": (0.997, 0.999),
        "GT_Compressor_outlet_air_pressure_P2_bar": (13.0, 15.0),
        "GT_exhaust_gas_pressure_Pexh_bar": (1.030, 1.040),
        "Turbine_Injecton_Control_TIC": (22.0, 30.0),
        "Fuel_flow_mf_kg_s": (0.90, 1.20),
        "GT_Compressor_decay_state_coefficient": (0.960, 0.985),
        "GT_Turbine_decay_state_coefficient": (0.980, 0.994),
    },
    "Compressor Degradation": {
        "Lever_position": (4.0, 6.0), "Ship_speed_v": (9, 18),
        "Gas_Turbine_GT_shaft_torque_GTT_kN_m": (10000, 30000),
        "GT_rate_of_revolutions_GTn_rpm": (1400, 1800),
        "Gas_Generator_rate_of_revolutions_GGn_rpm": (7000, 9000),
        "Starboard_Propeller_Torque_Ts_kN": (100, 300),
        "Port_Propeller_Torque_Tp_kN": (100, 300),
        "Hight_Pressure_HP_Turbine_exit_temperature_T48_C": (640, 730),
        "GT_Compressor_inlet_air_temperature_T1_C": (288, 295),
        "GT_Compressor_outlet_air_temperature_T2_C": (600, 660),
        "HP_Turbine_exit_pressure_P48_bar": (1.5, 2.1),
        "GT_Compressor_inlet_air_pressure_P1_bar": (0.994, 0.999),
        "GT_Compressor_outlet_air_pressure_P2_bar": (7.0, 12.0),
        "GT_exhaust_gas_pressure_Pexh_bar": (1.018, 1.030),
        "Turbine_Injecton_Control_TIC": (16.0, 25.0),
        "Fuel_flow_mf_kg_s": (0.45, 0.85),
        "GT_Compressor_decay_state_coefficient": (0.950, 0.962),
        "GT_Turbine_decay_state_coefficient": (0.980, 0.992),
    },
    "Turbine Failure Imminent": {
        "Lever_position": (5.0, 7.5), "Ship_speed_v": (18, 27),
        "Gas_Turbine_GT_shaft_torque_GTT_kN_m": (35000, 65000),
        "GT_rate_of_revolutions_GTn_rpm": (1750, 2100),
        "Gas_Generator_rate_of_revolutions_GGn_rpm": (8500, 10000),
        "Starboard_Propeller_Torque_Ts_kN": (350, 580),
        "Port_Propeller_Torque_Tp_kN": (350, 580),
        "Hight_Pressure_HP_Turbine_exit_temperature_T48_C": (745, 790),
        "GT_Compressor_inlet_air_temperature_T1_C": (294, 303),
        "GT_Compressor_outlet_air_temperature_T2_C": (650, 680),
        "HP_Turbine_exit_pressure_P48_bar": (2.2, 2.7),
        "GT_Compressor_inlet_air_pressure_P1_bar": (0.994, 0.999),
        "GT_Compressor_outlet_air_pressure_P2_bar": (11.5, 15.0),
        "GT_exhaust_gas_pressure_Pexh_bar": (1.028, 1.040),
        "Turbine_Injecton_Control_TIC": (23.0, 30.0),
        "Fuel_flow_mf_kg_s": (0.80, 1.20),
        "GT_Compressor_decay_state_coefficient": (0.955, 0.970),
        "GT_Turbine_decay_state_coefficient": (0.975, 0.980),
    },
}


# ============================================================
#  SIDEBAR — DATASET SELECTOR
# ============================================================

st.sidebar.markdown("## 🗂️ Dataset")
dataset_choice = st.sidebar.radio(
    "Select system to monitor:",
    ["✈️  Jet Engine (AI4I)", "⚓  Naval Vessel (Propulsion)"],
)
IS_JET   = dataset_choice.startswith("✈️")
IS_NAVAL = dataset_choice.startswith("⚓")
st.sidebar.markdown("---")


# ╔══════════════════════════════════════════════════════════╗
# ║                  JET ENGINE DATASET                      ║
# ╚══════════════════════════════════════════════════════════╝
if IS_JET:

    st.title("✈️ Jet Engine — Predictive Failure Detection")
    st.caption("Dataset: AI4I 2020  |  Model: XGBoost + Physics Rules  |  Every slider affects risk in real time")

    artifacts, err = load_jet_artifacts()
    if err:
        st.error(f"❌ {err}")
        st.stop()

    model          = artifacts["model"]
    subtype_models = artifacts["subtype_models"]
    preprocessor   = artifacts["preprocessor"]
    FEATURE_NAMES  = artifacts["feature_names"]
    FAILURE_LABELS = artifacts["failure_labels"]
    explainer      = artifacts.get("explainer")

    tab1, tab2 = st.tabs(["🎛️ Manual Prediction", "🤖 Auto Simulation"])

    # ── TAB 1: MANUAL ────────────────────────────────────────
    with tab1:
        st.info("💡 **Tip:** Drag **Tool Wear** toward 253 min, **Torque** above 60 Nm, or **Temperatures** to their maximums to trigger HIGH risk.")
        st.markdown("---")

        st.sidebar.header("⚙️ Jet Engine Parameters")
        machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])
        air_temp     = st.sidebar.slider("Air Temperature (K)",     295.0, 305.0, 300.0, step=0.1)
        process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0, step=0.1)
        rpm          = st.sidebar.slider("Rotational Speed (RPM)",  1168,  2886,  1500)
        torque       = st.sidebar.slider("Torque (Nm)",             3.8,   76.6,  40.0,  step=0.1)
        tool_wear    = st.sidebar.slider("Tool Wear (min)",         0,     253,   100)

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Business Cost Assumptions**")
        unplanned_cost   = st.sidebar.number_input("Unplanned Breakdown Cost (₹)", value=500000, step=10000)
        maintenance_cost = st.sidebar.number_input("Planned Maintenance Cost (₹)", value=50000,  step=5000)

        # Model prediction
        input_df = pd.DataFrame([{
            "Type":                    machine_type,
            "Air temperature [K]":     air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]":  rpm,
            "Torque [Nm]":             torque,
            "Tool wear [min]":         tool_wear,
        }])
        input_transformed = preprocessor.transform(input_df)
        model_prob  = model.predict_proba(input_transformed)[0][1]
        rule_score  = jet_rule_score(air_temp, process_temp, rpm, torque, tool_wear)
        failure_prob = hybrid_prob(model_prob, rule_score, model_weight=0.4)
        risk_lbl, _ = risk_label_color(failure_prob)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔴 Failure Risk")
            if failure_prob >= 0.55:
                st.error(f"**{risk_lbl}**\n\n{failure_prob*100:.1f}% failure probability")
            elif failure_prob >= 0.30:
                st.warning(f"**{risk_lbl}**\n\n{failure_prob*100:.1f}% failure probability")
            else:
                st.success(f"**{risk_lbl}**\n\n{failure_prob*100:.1f}% failure probability")
            st.progress(float(failure_prob))
            st.caption(f"Model: {model_prob*100:.1f}%  |  Physics rules: {rule_score*100:.1f}%  |  Combined: {failure_prob*100:.1f}%")

            st.markdown("**Parameter contributions:**")
            contribs = {
                "Air Temp":  np.clip((air_temp - 298.0) / 7.0, 0, 1),
                "Proc Temp": np.clip((process_temp - 306.0) / 10.0, 0, 1),
                "RPM":       np.clip(abs(rpm - 1500) / 1386, 0, 1),
                "Torque":    np.clip((torque - 20.0) / 56.6, 0, 1),
                "Tool Wear": np.clip(tool_wear / 253.0, 0, 1),
            }
            for name, val in contribs.items():
                icon = "🔴" if val > 0.7 else "🟡" if val > 0.4 else "🟢"
                st.write(f"{icon} {name}: {val*100:.0f}%")
                st.progress(float(val))

        with col2:
            st.subheader("⚠️ Failure Type Breakdown")
            breakdown = {}
            for col_name, sub_model in subtype_models.items():
                prob = sub_model.predict_proba(input_transformed)[0][1]
                breakdown[FAILURE_LABELS[col_name]] = prob
            if breakdown:
                bdf = pd.DataFrame(list(breakdown.items()),
                                   columns=["Failure Type","Probability"]).sort_values("Probability", ascending=False)
                for _, row in bdf.iterrows():
                    pct   = row["Probability"] * 100
                    color = "🔴" if pct > 20 else "🟡" if pct > 5 else "🟢"
                    st.write(f"{color} **{row['Failure Type']}**: {pct:.1f}%")
                    st.progress(float(row["Probability"]))
            else:
                st.info("No subtype models available.")

        with col3:
            st.subheader("💰 Business Impact")
            if failure_prob >= 0.55:
                saving = unplanned_cost - maintenance_cost
                st.metric("Saving if Maintained Now", f"₹{saving:,.0f}",
                          f"Avoid ₹{unplanned_cost:,.0f} breakdown")
                st.warning(f"Schedule maintenance — saves **₹{saving:,.0f}**.")
            elif failure_prob >= 0.30:
                st.metric("Risk Level", "MEDIUM", "Monitor closely")
                st.info("Elevated risk. Plan maintenance within 72 hours.")
            else:
                st.metric("Risk Level", "LOW", "No action needed")
                st.info("Machine operating normally.")

        st.markdown("---")

        if SHAP_AVAILABLE and explainer:
            st.subheader("🔍 SHAP Explanation")
            shap_vals = explainer.shap_values(input_transformed)
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=input_transformed[0],
                    feature_names=FEATURE_NAMES
                ), show=False
            )
            st.pyplot(fig); plt.close()

        with st.expander("📊 Global Feature Importance"):
            try:
                st.image(plt.imread("shap_summary.png"), use_column_width=True)
            except FileNotFoundError:
                st.info("Run jet_engine.py to generate shap_summary.png")

        with st.expander("📈 ROC Curve — All Models"):
            try:
                st.image(plt.imread("roc_auc_comparison.png"), use_column_width=True)
            except FileNotFoundError:
                st.info("Run jet_engine.py to generate roc_auc_comparison.png")

        with st.expander("📉 Failure Type Distribution"):
            try:
                st.image(plt.imread("failure_type_distribution.png"), use_column_width=True)
            except FileNotFoundError:
                st.info("Run jet_engine.py to generate failure_type_distribution.png")

    # ── TAB 2: JET SIMULATION ────────────────────────────────
    with tab2:
        st.markdown("Live simulation — hybrid risk updates every second.")
        st.markdown("---")

        for key, default in [("jet_running", False), ("jet_history", []),
                              ("jet_alarms", 0), ("jet_tick", 0)]:
            if key not in st.session_state:
                st.session_state[key] = default

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            jet_scenario = st.selectbox("Scenario", [
                "Normal Operation", "High Load", "Tool Degradation", "Overheating"])
            st.caption({
                "Normal Operation": "🟢 Stable — low risk expected",
                "High Load":        "🟡 Elevated RPM and torque",
                "Tool Degradation": "🟠 Wear approaching critical",
                "Overheating":      "🔴 Thermal runaway",
            }[jet_scenario])
        with c2:
            jet_thresh = st.slider("Alarm Threshold (%)", 10, 90, 40, key="jt") / 100
        with c3:
            st.write(""); st.write("")
            b1, b2 = st.columns(2)
            if b1.button("▶ Start", type="primary", use_container_width=True, key="js"):
                st.session_state.jet_running = True
                st.session_state.jet_history = []
                st.session_state.jet_alarms  = 0
                st.session_state.jet_tick    = 0
            if b2.button("⏹ Stop", use_container_width=True, key="jstop"):
                st.session_state.jet_running = False

        JET_SC = {
            "Normal Operation": {"air":(298,300),"proc":(308,311),"rpm":(1400,1600),"torque":(30,45), "wear":(10,80)},
            "High Load":        {"air":(301,304),"proc":(312,315),"rpm":(2000,2500),"torque":(60,72), "wear":(100,180)},
            "Tool Degradation": {"air":(299,302),"proc":(310,313),"rpm":(1300,1700),"torque":(55,72), "wear":(190,253)},
            "Overheating":      {"air":(303,305),"proc":(313,316),"rpm":(2200,2886),"torque":(68,76), "wear":(160,230)},
        }

        alarm_ph, metrics_ph, chart_ph = st.empty(), st.empty(), st.empty()

        if st.session_state.jet_running:
            s = JET_SC[jet_scenario]
            r = {
                "air":    round(random.uniform(*s["air"]),   2),
                "proc":   round(random.uniform(*s["proc"]),  2),
                "rpm":    int(random.uniform(*s["rpm"])),
                "torque": round(random.uniform(*s["torque"]),2),
                "wear":   round(random.uniform(*s["wear"]),  1),
                "type":   random.choice(["L","M","H"]),
            }
            inp = pd.DataFrame([{
                "Type": r["type"], "Air temperature [K]": r["air"],
                "Process temperature [K]": r["proc"], "Rotational speed [rpm]": r["rpm"],
                "Torque [Nm]": r["torque"], "Tool wear [min]": r["wear"],
            }])
            mp   = model.predict_proba(preprocessor.transform(inp))[0][1]
            rs   = jet_rule_score(r["air"], r["proc"], r["rpm"], r["torque"], r["wear"])
            prob = hybrid_prob(mp, rs, 0.4)

            if prob >= jet_thresh:
                st.session_state.jet_alarms += 1
            st.session_state.jet_history.append({"tick": st.session_state.jet_tick, "risk": prob*100})
            if len(st.session_state.jet_history) > 40:
                st.session_state.jet_history = st.session_state.jet_history[-40:]
            st.session_state.jet_tick += 1
            hist = st.session_state.jet_history

            with alarm_ph.container():
                if prob >= jet_thresh:
                    st.error(f"🚨 **ALARM!** Risk: **{prob*100:.1f}%** | Threshold: {jet_thresh*100:.0f}% | Alarms: {st.session_state.jet_alarms}")
                else:
                    st.success(f"✅ **NORMAL** — Risk: **{prob*100:.1f}%** | Alarms: {st.session_state.jet_alarms}")

            with metrics_ph.container():
                m1,m2,m3,m4,m5,m6 = st.columns(6)
                m1.metric("🔴 Risk",    f"{prob*100:.1f}%")
                m2.metric("🌡️ Air",    f"{r['air']} K")
                m3.metric("🌡️ Proc",   f"{r['proc']} K")
                m4.metric("⚙️ RPM",    f"{r['rpm']}")
                m5.metric("🔩 Torque", f"{r['torque']} Nm")
                m6.metric("🛠️ Wear",   f"{r['wear']} min")

            with chart_ph.container():
                ticks = [h["tick"] for h in hist]
                risks = [h["risk"] for h in hist]
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
                fc = "#ff4444" if prob >= jet_thresh else "#00cc66"
                ax.fill_between(ticks, risks, alpha=0.2, color=fc)
                ax.plot(ticks, risks, color="#00d4ff", lw=2.5, marker='o', markersize=3)
                ax.axhline(jet_thresh*100, color="#ff4444", ls="--", lw=1.5, label=f"Threshold {jet_thresh*100:.0f}%")
                at = [h["tick"] for h in hist if h["risk"] >= jet_thresh*100]
                ar = [h["risk"] for h in hist if h["risk"] >= jet_thresh*100]
                if at: ax.scatter(at, ar, color="#ff4444", s=80, zorder=5)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Reading #", color="#aaa"); ax.set_ylabel("Risk %", color="#aaa")
                ax.set_title("Live Failure Risk — Jet Engine", color="#ddd")
                ax.tick_params(colors="#888"); ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
                for sp in ax.spines.values(): sp.set_edgecolor("#333")
                st.pyplot(fig); plt.close()

            time.sleep(1); st.rerun()
        else:
            alarm_ph.info("▶ Press **Start** to begin simulation.")
            if st.session_state.jet_history:
                risks = [h["risk"] for h in st.session_state.jet_history]
                st.markdown("---"); st.subheader("📊 Last Simulation Summary")
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("Readings", len(risks))
                s2.metric("Alarms",   st.session_state.jet_alarms)
                s3.metric("Peak",     f"{max(risks):.1f}%")
                s4.metric("Average",  f"{np.mean(risks):.1f}%")


# ╔══════════════════════════════════════════════════════════╗
# ║                NAVAL VESSEL DATASET                      ║
# ╚══════════════════════════════════════════════════════════╝
elif IS_NAVAL:

    st.title("⚓ Naval Vessel — Propulsion Failure Detection")
    st.caption("Dataset: UCI Naval Propulsion (11,934 records)  |  Model: XGBoost + Physics Rules  |  Every slider affects risk in real time")

    naval_artifacts, err = load_naval_artifacts()
    if err:
        st.error(f"❌ {err}")
        st.stop()

    naval_model = naval_artifacts["model"] if isinstance(naval_artifacts, dict) else naval_artifacts

    tab1, tab2 = st.tabs(["🎛️ Manual Prediction", "🤖 Auto Simulation"])

    # ── TAB 1: MANUAL ────────────────────────────────────────
    with tab1:
        st.info("💡 **Tip:** Drag **★ Compressor Decay (kMc)** or **★ Turbine Decay (kMt)** toward 0.950 / 0.975, "
                "or push **HP Turbine Exit Temp** above 730 °C to trigger HIGH / CRITICAL risk.")
        st.markdown("---")

        st.sidebar.header("⚙️ Naval Propulsion Parameters")
        user_inputs = {}
        for feat, display in NAVAL_DISPLAY_NAMES.items():
            lo, hi, default = NAVAL_RANGES[feat]
            lo, hi, default = float(lo), float(hi), float(default)
            rng  = hi - lo
            step = 0.0001 if rng < 0.05 else 0.001 if rng < 0.1 else 0.01 if rng < 1 else 0.1 if rng < 10 else 1.0
            user_inputs[feat] = st.sidebar.slider(display, lo, hi, default, step=step)

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Business Cost**")
        nav_unplanned = st.sidebar.number_input("Unplanned Repair ($)", value=2000000, step=50000)
        nav_planned   = st.sidebar.number_input("Planned Maintenance ($)", value=150000, step=10000)

        model_prob   = naval_model_predict(naval_model, user_inputs)
        rule_score   = naval_rule_score(user_inputs)
        failure_prob = hybrid_prob(model_prob, rule_score, model_weight=0.35)
        risk_lbl, _ = risk_label_color(failure_prob)

        kmc = user_inputs["GT_Compressor_decay_state_coefficient"]
        kmt = user_inputs["GT_Turbine_decay_state_coefficient"]
        t48 = user_inputs["Hight_Pressure_HP_Turbine_exit_temperature_T48_C"]
        vs  = user_inputs["Ship_speed_v"]
        mf  = user_inputs["Fuel_flow_mf_kg_s"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔴 Failure Risk")
            if failure_prob >= 0.55:
                st.error(f"**{risk_lbl}**\n\n{failure_prob*100:.1f}% failure probability")
            elif failure_prob >= 0.30:
                st.warning(f"**{risk_lbl}**\n\n{failure_prob*100:.1f}% failure probability")
            else:
                st.success(f"**{risk_lbl}**\n\n{failure_prob*100:.1f}% failure probability")
            st.progress(float(failure_prob))
            st.caption(f"Model: {model_prob*100:.1f}%  |  Physics rules: {rule_score*100:.1f}%  |  Combined: {failure_prob*100:.1f}%")

            st.markdown("**Key risk drivers:**")
            contribs = {
                "Compressor Decay": np.clip((1.0-kmc)/(1.0-0.950), 0, 1),
                "Turbine Decay":    np.clip((1.0-kmt)/(1.0-0.975), 0, 1),
                "Turbine Temp":     np.clip((t48-445)/(790-445), 0, 1),
                "Ship Speed":       np.clip((vs-3)/(27-3), 0, 1),
                "Fuel Flow":        np.clip((mf-0.05)/(1.20-0.05), 0, 1),
            }
            for name, val in contribs.items():
                icon = "🔴" if val > 0.7 else "🟡" if val > 0.4 else "🟢"
                st.write(f"{icon} {name}: {val*100:.0f}%")
                st.progress(float(val))

        with col2:
            st.subheader("🔩 Component Health")
            def health_bar(label, val, lo, hi):
                pct = (val - lo) / (hi - lo)
                icon = "🟢" if pct > 0.7 else "🟡" if pct > 0.4 else "🔴"
                st.write(f"{icon} **{label}**: {val:.4f}")
                st.progress(float(np.clip(pct, 0, 1)))

            health_bar("Compressor Decay (kMc)", kmc, 0.950, 1.000)
            health_bar("Turbine Decay (kMt)",    kmt, 0.975, 1.000)
            health_bar("Combined Degradation",   kmc*kmt, 0.926, 1.000)
            st.caption("1.0 = new  |  lower = degraded")

            st.markdown("---")
            st.markdown("**Thermal status:**")
            t48_pct  = (t48 - 445) / (790 - 445)
            t48_icon = "🔴 OVERTEMP" if t48_pct > 0.8 else "🟡 Elevated" if t48_pct > 0.5 else "🟢 Normal"
            st.write(f"{t48_icon} — HP Turbine: **{t48:.0f} °C**")
            st.progress(float(np.clip(t48_pct, 0, 1)))

        with col3:
            st.subheader("💰 Business Impact")
            if failure_prob >= 0.55:
                saving = nav_unplanned - nav_planned
                st.metric("Saving if Maintained Now", f"${saving:,.0f}",
                          f"Avoid ${nav_unplanned:,.0f} repair")
                st.warning(f"Schedule dock maintenance — saves **${saving:,.0f}**.")
            elif failure_prob >= 0.30:
                st.metric("Risk Level", "MEDIUM", "Plan inspection")
                st.info("Elevated. Schedule inspection within 48 hours.")
            else:
                st.metric("Risk Level", "LOW", "No action needed")
                st.info("Propulsion plant operating normally.")

        st.markdown("---")
        st.subheader("📊 Feature Importance (XGBoost)")
        try:
            imp        = naval_model.feature_importances_
            feat_names = naval_model.feature_names_in_ if hasattr(naval_model, "feature_names_in_") else list(user_inputs.keys())
            fi         = pd.Series(imp, index=feat_names).sort_values(ascending=True).tail(10)
            labels     = [NAVAL_DISPLAY_NAMES.get(f, f).replace("★ ", "") for f in fi.index]
            colors     = ["#e05252" if any(x in f for x in ["decay","TIC","T48","kmc","kmt"]) else "#4a9fd4" for f in fi.index]
            fig, ax    = plt.subplots(figsize=(9, 4))
            ax.barh(labels, fi.values, color=colors, edgecolor="none")
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 10 Feature Importance — Naval XGBoost")
            for i, v in enumerate(fi.values):
                ax.text(v+0.001, i, f"{v:.3f}", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption("🔴 Red = health-critical  |  🔵 Blue = operational")
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")

    # ── TAB 2: NAVAL SIMULATION ───────────────────────────────
    with tab2:
        st.markdown("Live propulsion monitoring — hybrid risk updates every second.")
        st.markdown("---")

        for key, default in [("nav_running", False), ("nav_history", []),
                              ("nav_alarms", 0), ("nav_tick", 0)]:
            if key not in st.session_state:
                st.session_state[key] = default

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            nav_scenario = st.selectbox("Scenario", list(NAVAL_SCENARIOS.keys()))
            st.caption({
                "Normal Cruise (15 kn)":    "🟢 Stable mid-range — low risk",
                "High Speed (27 kn)":       "🟡 Max load — thermal/mechanical stress",
                "Compressor Degradation":   "🟠 kMc dropping — wear accelerating",
                "Turbine Failure Imminent": "🔴 Overtemp + low kMt — alarm expected",
            }.get(nav_scenario, ""))
        with c2:
            nav_thresh = st.slider("Alarm Threshold (%)", 10, 90, 35, key="nt") / 100
        with c3:
            st.write(""); st.write("")
            b1, b2 = st.columns(2)
            if b1.button("▶ Start", type="primary", use_container_width=True, key="ns"):
                st.session_state.nav_running = True
                st.session_state.nav_history = []
                st.session_state.nav_alarms  = 0
                st.session_state.nav_tick    = 0
            if b2.button("⏹ Stop", use_container_width=True, key="nstop"):
                st.session_state.nav_running = False

        nav_alarm_ph, nav_metrics_ph, nav_chart_ph = st.empty(), st.empty(), st.empty()

        if st.session_state.nav_running:
            s       = NAVAL_SCENARIOS[nav_scenario]
            reading = {feat: round(random.uniform(*rng), 4) for feat, rng in s.items()}

            mp   = naval_model_predict(naval_model, reading)
            rs   = naval_rule_score(reading)
            prob = hybrid_prob(mp, rs, model_weight=0.35)

            if prob >= nav_thresh:
                st.session_state.nav_alarms += 1
            st.session_state.nav_history.append({"tick": st.session_state.nav_tick, "risk": prob*100})
            if len(st.session_state.nav_history) > 40:
                st.session_state.nav_history = st.session_state.nav_history[-40:]
            st.session_state.nav_tick += 1
            hist = st.session_state.nav_history

            with nav_alarm_ph.container():
                if prob >= nav_thresh:
                    st.error(f"🚨 **PROPULSION ALARM!** Risk: **{prob*100:.1f}%** | Threshold: {nav_thresh*100:.0f}% | Alarms: {st.session_state.nav_alarms}")
                else:
                    st.success(f"✅ **NORMAL** — Risk: **{prob*100:.1f}%** | Alarms: {st.session_state.nav_alarms}")

            with nav_metrics_ph.container():
                m1,m2,m3,m4,m5,m6 = st.columns(6)
                m1.metric("🔴 Risk",         f"{prob*100:.1f}%")
                m2.metric("⚓ Speed",         f"{reading['Ship_speed_v']:.0f} kn")
                m3.metric("🌡️ Turbine Temp", f"{reading['Hight_Pressure_HP_Turbine_exit_temperature_T48_C']:.0f} °C")
                m4.metric("⚙️ GT RPM",       f"{reading['GT_rate_of_revolutions_GTn_rpm']:.0f}")
                m5.metric("🔧 kMc",          f"{reading['GT_Compressor_decay_state_coefficient']:.4f}")
                m6.metric("🔧 kMt",          f"{reading['GT_Turbine_decay_state_coefficient']:.4f}")

            with nav_chart_ph.container():
                ticks = [h["tick"] for h in hist]
                risks = [h["risk"] for h in hist]
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
                fc = "#ff4444" if prob >= nav_thresh else "#00cc66"
                ax.fill_between(ticks, risks, alpha=0.2, color=fc)
                ax.plot(ticks, risks, color="#00d4ff", lw=2.5, marker='o', markersize=3)
                ax.axhline(nav_thresh*100, color="#ff4444", ls="--", lw=1.5, label=f"Threshold {nav_thresh*100:.0f}%")
                at = [h["tick"] for h in hist if h["risk"] >= nav_thresh*100]
                ar = [h["risk"] for h in hist if h["risk"] >= nav_thresh*100]
                if at: ax.scatter(at, ar, color="#ff4444", s=80, zorder=5)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Reading #", color="#aaa"); ax.set_ylabel("Risk %", color="#aaa")
                ax.set_title("Live Propulsion Risk — Naval Vessel", color="#ddd")
                ax.tick_params(colors="#888"); ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
                for sp in ax.spines.values(): sp.set_edgecolor("#333")
                st.pyplot(fig); plt.close()

            time.sleep(1); st.rerun()
        else:
            nav_alarm_ph.info("▶ Press **Start** to begin live propulsion monitoring.")
            if st.session_state.nav_history:
                risks = [h["risk"] for h in st.session_state.nav_history]
                st.markdown("---"); st.subheader("📊 Last Simulation Summary")
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("Readings", len(risks))
                s2.metric("Alarms",   st.session_state.nav_alarms)
                s3.metric("Peak",     f"{max(risks):.1f}%")
                s4.metric("Average",  f"{np.mean(risks):.1f}%")
