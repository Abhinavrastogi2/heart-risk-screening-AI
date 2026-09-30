import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

st.set_page_config(
    page_title="Heart Risk Screening Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
    }
    .risk-moderate {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        color: #92400E;
        padding: 1rem;
        border-radius: 8px;
    }
    .risk-low {
        background-color: #F0FDF4;
        border-left: 6px solid #10B981;
        color: #065F46;
        padding: 1rem;
        border-radius: 8px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load Models & Configuration
@st.cache_resource
def load_models_and_config():
    base_dir = os.path.dirname(__file__)
    home_path = os.path.join(base_dir, "models", "home_model.joblib")
    clinic_path = os.path.join(base_dir, "models", "clinic_model.joblib")
    config_path = os.path.join(base_dir, "models", "config.json")
    metrics_path = os.path.join(base_dir, "models", "metrics_data.json")

    home_model = joblib.load(home_path) if os.path.exists(home_path) else None
    clinic_model = joblib.load(clinic_path) if os.path.exists(clinic_path) else None

    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    return home_model, clinic_model, config, metrics_data

@st.cache_data
def load_raw_dataset():
    paths = [
        r"C:\Users\rasto\Downloads\heart_disease_uci.csv",
        "heart_disease_uci.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

home_model, clinic_model, config, metrics_data = load_models_and_config()
df_raw = load_raw_dataset()

# Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=64)
st.sidebar.title("Heart Disease Risk AI")
st.sidebar.markdown("**Dual-Mode Clinical & Home Screening**")

mode = st.sidebar.radio(
    "Select Mode / Dashboard Section:",
    [
        "🏡 Home Mode (Patient Screen)",
        "🏥 Clinic Mode (Clinical Screen)",
        "📊 Model Benchmarks & Metrics",
        "🔍 Feature Importance & Explainability",
        "⚖️ Subgroup Fairness & Robustness",
        "📈 Dataset Overview & EDA"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Screening Goal:** Focus on **Recall (Sensitivity)** to ensure sick patients are flagged early.
- **Home Threshold:** 0.43 (Recall: ~93%)
- **Clinic Threshold:** 0.44 (Recall: ~91%)
""")

# Risk Classifier Helper
def get_risk_level(prob, threshold):
    low_cut = threshold / 2.0
    if prob >= threshold:
        return "HIGH", "risk-high", "High probability of heart risk. Clinical consultation recommended."
    elif prob >= low_cut:
        return "MODERATE", "risk-moderate", "Moderate risk detected. Monitor symptoms and consider lifestyle changes."
    else:
        return "LOWER", "risk-low", "Low risk detected. Maintain healthy habits and routine checkups."

# ------------------------------------------------------------------
# PAGE 1: HOME MODE
# ------------------------------------------------------------------
if mode == "🏡 Home Mode (Patient Screen)":
    st.markdown("<div class='main-header'>🏡 Home Heart Risk Checker</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Designed for patients & non-clinical users to screen heart disease risk using simple inputs.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Enter Your Details")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        
        cp = st.selectbox(
            "Chest Pain Type", 
            options=["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"],
            help="Typical angina: pressure/squeezing during exertion. Asymptomatic: no pain."
        )
        
        trestbps = st.number_input(
            "Resting Blood Pressure (mmHg)", 
            min_value=80, max_value=220, value=130,
            help="Typical range: 90 - 140 mmHg"
        )
        
        exang = st.selectbox(
            "Do you experience chest pain during exercise/exertion?",
            options=["False", "True"],
            format_func=lambda x: "Yes" if x == "True" else "No"
        )
        
        with st.expander("Optional Lab Values (If available from recent blood test)"):
            chol_input = st.text_input("Cholesterol (mg/dL) [Leave blank if unknown]", value="")
            fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", options=["Unknown", "False", "True"])

        chol_val = float(chol_input) if chol_input.strip().isdigit() else np.nan
        fbs_val = fbs_input if fbs_input != "Unknown" else np.nan

        predict_btn = st.button("🔍 Assess My Heart Risk", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎯 Risk Assessment Result")
        if predict_btn and home_model:
            input_df = pd.DataFrame([{
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "exang": exang,
                "chol": chol_val,
                "fbs": fbs_val
            }])

            prob = home_model.predict_proba(input_df)[0, 1]
            thr = config.get("home_threshold", 0.43)
            risk_label, css_class, msg = get_risk_level(prob, thr)

            st.markdown(f"""
            <div class="{css_class}">
                <h2 style="margin:0; font-size:1.8rem;">Risk Level: {risk_label}</h2>
                <h3 style="margin:5px 0; font-size:1.3rem;">Estimated Probability: {prob:.1%}</h3>
                <p style="margin-top:10px; font-weight:500;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Key Contributing Factors")
            st.write(f"- **Chest Pain Type:** {cp.title()} (Major factor in home screening)")
            st.write(f"- **Exercise Angina:** {'Yes' if exang == 'True' else 'No'}")
            st.write(f"- **Age / Sex:** {age} year old {sex}")

            if risk_label == "HIGH":
                st.warning("⚠️ **Important:** This screening tool indicates elevated risk. Please consult a qualified cardiologist or healthcare provider for comprehensive evaluation.")
            elif risk_label == "MODERATE":
                st.info("ℹ️ **Recommendation:** Schedule a routine health checkup and keep track of blood pressure and cholesterol levels.")
            else:
                st.success("✅ **Good News:** Your inputs indicate low risk based on the home screening model. Keep up a healthy lifestyle!")

            st.caption("🚨 **Emergency Notice:** If you are experiencing acute chest pain, shortness of breath, or numbness, call emergency services immediately.")
        else:
            st.info("Fill out the form on the left and click **Assess My Heart Risk** to see your results.")

# ------------------------------------------------------------------
# PAGE 2: CLINIC MODE
# ------------------------------------------------------------------
elif mode == "🏥 Clinic Mode (Clinical Screen)":
    st.markdown("<div class='main-header'>🏥 Clinic Decision Support Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Designed for nurses and doctors incorporating ECG and stress-test data.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🩺 Patient Clinical Data")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 100, 58, key="c_age")
            sex = st.selectbox("Sex", ["Male", "Female"], key="c_sex")
            cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"], key="c_cp")
            trestbps = st.number_input("Resting BP (mmHg)", 80, 220, 140, key="c_bp")
            exang = st.selectbox("Exercise Angina", ["False", "True"], key="c_exang")
            chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 240, key="c_chol")
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["False", "True"], key="c_fbs")
        
        with c2:
            st.markdown("**ECG & Stress Test Data**")
            restecg = st.selectbox(
                "Resting ECG Results",
                options=["normal", "st-t abnormality", "lv hypertrophy"],
                help="lv hypertrophy: left ventricular hypertrophy"
            )
            thalch = st.number_input("Max Heart Rate Achieved (thalch)", 60, 220, 135)
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.5, step=0.1)
            slope = st.selectbox("ST Slope", options=["upsloping", "flat", "downsloping"])

        calc_clinic = st.button("🏥 Run Clinical Screening Model", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Comparative Risk Analysis")

        if calc_clinic and clinic_model and home_model:
            home_data = pd.DataFrame([{
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "exang": exang, "chol": chol, "fbs": fbs
            }])

            clinic_data = pd.DataFrame([{
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "exang": exang, "chol": chol, "fbs": fbs,
                "restecg": restecg, "thalch": thalch, "oldpeak": oldpeak, "slope": slope
            }])

            prob_home = home_model.predict_proba(home_data)[0, 1]
            prob_clinic = clinic_model.predict_proba(clinic_data)[0, 1]

            thr_h = config.get("home_threshold", 0.43)
            thr_c = config.get("clinic_threshold", 0.44)

            lbl_h, css_h, _ = get_risk_level(prob_home, thr_h)
            lbl_c, css_c, _ = get_risk_level(prob_clinic, thr_c)

            st.markdown(f"""
            <div style="display:flex; gap:15px; margin-bottom:15px;">
                <div style="flex:1;" class="{css_h}">
                    <h4 style="margin:0;">🏡 Home Model</h4>
                    <p style="font-size:1.4rem; font-weight:bold; margin:5px 0;">{prob_home:.1%}</p>
                    <span style="font-weight:600;">Status: {lbl_h}</span>
                </div>
                <div style="flex:1;" class="{css_c}">
                    <h4 style="margin:0;">🏥 Clinic Model</h4>
                    <p style="font-size:1.4rem; font-weight:bold; margin:5px 0;">{prob_clinic:.1%}</p>
                    <span style="font-weight:600;">Status: {lbl_c}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            delta = prob_clinic - prob_home
            st.metric(
                label="Clinical Refinement Delta (Clinic vs. Home)",
                value=f"{prob_clinic:.1%}",
                delta=f"{delta:+.1%} probability shift after ECG & stress test"
            )

            st.markdown("### 📋 Clinical Diagnostic Summary")
            if prob_clinic >= thr_c:
                st.error("🚨 **High Risk Flagged**: Patient meets clinical decision threshold. Recommend further diagnostic testing (e.g. coronary angiography, echocardiogram).")
            else:
                st.success("✅ **Low / Moderate Risk Flagged**: Patient below clinical threshold. Maintain routine monitoring.")

        else:
            st.info("Input patient parameters and click **Run Clinical Screening Model**.")

# ------------------------------------------------------------------
# PAGE 3: MODEL BENCHMARKS & METRICS
# ------------------------------------------------------------------
elif mode == "📊 Model Benchmarks & Metrics":
    st.markdown("<div class='main-header'>📊 Model Performance & CV Benchmarks</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Rigorous 5-Fold Stratified Cross-Validation & Test Set Evaluation</div>", unsafe_allow_html=True)

    if metrics_data:
        t1, t2 = st.tabs(["🏡 Home Model Performance", "🏥 Clinic Model Performance"])

        with t1:
            st.subheader("Home Model (7 Features)")
            h_eval = metrics_data["evaluation"]["home"]
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("ROC-AUC", f"{h_eval['roc_auc']:.3f}")
            m2.metric("Recall (Sensitivity)", f"{h_eval['recall']:.1%}")
            m3.metric("Accuracy", f"{h_eval['accuracy']:.1%}")
            m4.metric("Precision", f"{h_eval['precision']:.1%}")
            m5.metric("Decision Threshold", f"{h_eval['threshold']}")

            st.markdown("#### 5-Fold Stratified Cross-Validation Benchmarks")
            df_h_cv = pd.DataFrame(metrics_data["comparison"]["home"])
            st.dataframe(df_h_cv.style.highlight_max(axis=0, subset=["roc_auc_mean", "recall_mean"]), use_container_width=True)

        with t2:
            st.subheader("Clinic Model (11 Features)")
            c_eval = metrics_data["evaluation"]["clinic"]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("ROC-AUC", f"{c_eval['roc_auc']:.3f}")
            m2.metric("Recall (Sensitivity)", f"{c_eval['recall']:.1%}")
            m3.metric("Accuracy", f"{c_eval['accuracy']:.1%}")
            m4.metric("Precision", f"{c_eval['precision']:.1%}")
            m5.metric("Decision Threshold", f"{c_eval['threshold']}")

            st.markdown("#### 5-Fold Stratified Cross-Validation Benchmarks")
            df_c_cv = pd.DataFrame(metrics_data["comparison"]["clinic"])
            st.dataframe(df_c_cv.style.highlight_max(axis=0, subset=["roc_auc_mean", "recall_mean"]), use_container_width=True)

# ------------------------------------------------------------------
# PAGE 4: FEATURE IMPORTANCE
# ------------------------------------------------------------------
elif mode == "🔍 Feature Importance & Explainability":
    st.markdown("<div class='main-header'>🔍 Model Explainability & Feature Importance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Permutation importance on untouched test set (ROC-AUC drop when shuffled)</div>", unsafe_allow_html=True)

    if metrics_data and "feature_importance" in metrics_data:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏡 Home Model Importance")
            imp_h = pd.Series(metrics_data["feature_importance"]["home"]).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            imp_h.plot(kind="barh", color="#3B82F6", ax=ax)
            ax.set_xlabel("Mean drop in ROC-AUC")
            ax.set_title("Home Model Feature Importance")
            st.pyplot(fig)

        with c2:
            st.subheader("🏥 Clinic Model Importance")
            imp_c = pd.Series(metrics_data["feature_importance"]["clinic"]).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            imp_c.plot(kind="barh", color="#10B981", ax=ax)
            ax.set_xlabel("Mean drop in ROC-AUC")
            ax.set_title("Clinic Model Feature Importance")
            st.pyplot(fig)

# ------------------------------------------------------------------
# PAGE 5: SUBGROUP FAIRNESS
# ------------------------------------------------------------------
elif mode == "⚖️ Subgroup Fairness & Robustness":
    st.markdown("<div class='main-header'>⚖️ Subgroup Fairness & Hospital Robustness</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Auditing performance disparities across gender and hospital sources</div>", unsafe_allow_html=True)

    if metrics_data and "fairness" in metrics_data:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clinic Model Performance by Sex")
            df_sex = pd.DataFrame(metrics_data["fairness"]["clinic_by_sex"])
            st.dataframe(df_sex, use_container_width=True)

        with col2:
            st.subheader("Clinic Model Performance by Hospital")
            df_hosp = pd.DataFrame(metrics_data["fairness"]["clinic_by_hosp"])
            st.dataframe(df_hosp, use_container_width=True)

        st.subheader("🏨 Leave-One-Hospital-Out Generalization Audit")
        st.write("Models trained on 3 hospitals and tested on the 4th held-out hospital to check transferability:")
        df_loho = pd.DataFrame(metrics_data["loho"])
        st.dataframe(df_loho, use_container_width=True)

# ------------------------------------------------------------------
# PAGE 6: DATASET OVERVIEW
# ------------------------------------------------------------------
elif mode == "📈 Dataset Overview & EDA":
    st.markdown("<div class='main-header'>📈 Heart Disease UCI Dataset Explorer</div>", unsafe_allow_html=True)

    if df_raw is not None:
        st.subheader("Dataset Preview (920 Patients)")
        st.dataframe(df_raw.head(10), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Patients", len(df_raw))
        c2.metric("Hospitals Included", df_raw['dataset'].nunique() if 'dataset' in df_raw else 4)
        c3.metric("Missing Value Columns", sum(df_raw.isnull().sum() > 0))

        st.subheader("Key Distribution Plots")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(data=df_raw, x="age", hue="num", kde=True, ax=axes[0])
        axes[0].set_title("Age Distribution by Disease Severity")

        sns.countplot(data=df_raw, x="cp", hue="num", ax=axes[1])
        axes[1].set_title("Chest Pain Type vs. Disease")
        axes[1].tick_params(axis="x", rotation=25)

        sns.countplot(data=df_raw, x="sex", hue="num", ax=axes[2])
        axes[2].set_title("Sex Distribution vs. Disease")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error("Dataset `heart_disease_uci.csv` not found.")

# Global Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.8rem; padding: 1.5rem 0 0.5rem 0;'>"
    "© 2026 Heart Disease Risk AI Screening Tool • Made in 2026<br>"
    "<span style='font-size: 0.75rem; color: #94A3B8;'>Dual-Mode Home & Clinical Decision Support System • Educational & Decision Support Use Only</span>"
    "</div>",
    unsafe_allow_html=True
)

