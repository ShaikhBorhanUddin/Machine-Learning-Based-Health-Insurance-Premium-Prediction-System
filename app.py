import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('Models/xgboost_model_cpu.pkl')
except FileNotFoundError:
    st.error("Model file 'xgboost_model_cpu.pkl' not found. Please ensure it's in the same directory as app.py.")
    st.stop()

st.set_page_config(page_title="Medical Insurance Premium Predictor", layout="wide")
st.title("🏥 Medical Insurance Premium Prediction")
st.markdown("Enter the details below to predict the annual medical insurance premium.")

# ================= FEATURE ENGINEERING HELPERS =================

BMI_BINS = [0, 18.5, 25, 30, 35, 40, float('inf')]
BMI_LABELS = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
def get_bmi_group(bmi_value):
    return pd.cut(pd.Series([bmi_value]), bins=BMI_BINS, labels=BMI_LABELS, right=False).iloc[0]

def get_bp_category(systolic_bp, diastolic_bp):
    if systolic_bp < 90 or diastolic_bp < 60:
        return "Low Blood Pressure"
    elif 90 <= systolic_bp <= 129 and 60 <= diastolic_bp <= 80:
        return "Normal"
    elif 120 <= systolic_bp <= 129 and diastolic_bp < 80:
        return "Elevated"
    elif 130 <= systolic_bp <= 139 or 81 <= diastolic_bp <= 89:
        return "Hypertension Stage 1"
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        return "Hypertension Stage 2"

LDL_BINS = [0, 100, 130, 160, 190, np.inf]
LDL_LABELS = ['Optimal', 'Near Optimal', 'Borderline High', 'High', 'Very High']
def get_ldl_category(ldl_value):
    return pd.cut(pd.Series([ldl_value]), bins=LDL_BINS, labels=LDL_LABELS, right=False).iloc[0]

HBA1C_BINS = [0, 5.7, 6.5, np.inf]
HBA1C_LABELS = ['Normal', 'Prediabetes', 'Diabetes']
def get_hba1c_category(hba1c_value):
    return pd.cut(pd.Series([hba1c_value]), bins=HBA1C_BINS, labels=HBA1C_LABELS, right=False).iloc[0]

ANNUAL_MEDICAL_COST_BINS = [0, 500, 2000, 5000, 10000, np.inf]
ANNUAL_MEDICAL_COST_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
def get_annual_medical_cost_category(cost_value):
    return pd.cut(pd.Series([cost_value]), bins=ANNUAL_MEDICAL_COST_BINS, labels=ANNUAL_MEDICAL_COST_LABELS, right=False).iloc[0]

DEDUCTIBLE_BINS = [0, 1000, 2000, 3000, np.inf]
DEDUCTIBLE_LABELS = ['Low', 'Moderate', 'High', 'Too High']
def get_deductible_category(deductible_value):
    return pd.cut(pd.Series([deductible_value]), bins=DEDUCTIBLE_BINS, labels=DEDUCTIBLE_LABELS, right=False).iloc[0]

PLAN_TYPE_SUGGESTIONS = {
    "Health Maintenance Organization": "Choose HMO if you want the lowest monthly costs and don't mind using a primary care doctor to manage your care.",
    "Preferred Provider Organization": "Choose PPO if you want the freedom to see specialists without referrals and access out-of-network care.",
    "Exclusive Provider Organization": "Choose EPO if you want lower premiums like an HMO but don't want referrals for specialists.",
    "Point-of-Service": "Choose POS if you want the cost savings of an HMO but want the option to go out-of-network."
}

# ================= PERSONAL & HEALTH DETAILS =================

st.header("Personal & Health Details")
c1, c2, c3, c4 = st.columns(4)

with c1:
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Divorced', 'Single', 'Widowed'])
    household_size = st.slider("Household Size", 1, 10, 2)

with c2:
    dependents = st.slider("Dependents", 0, 9, 1)
    education = st.selectbox("Education Level", ['College', 'Doctorate', 'High School Dropout', 'High School', 'Masters', 'Bachelors'])
    employment_status = st.selectbox("Employment Status", ['Employed', 'Self-employed', 'Retired', 'Unemployed'])
    income = st.number_input("Yearly Income", 10000, 10000000, 12000)

with c3:
    region = st.selectbox("Geographical Region", ['North', 'Central', 'West', 'East', 'South'])
    urban_rural = st.selectbox("Area", ['Suburban', 'Urban', 'Rural'])
    smoker = st.selectbox("Smoking Habit", ['Never', 'Former', 'Current'])
    alcohol_freq = st.selectbox("Alcohol Consumption", ['Never', 'Weekly', 'Daily', 'Occasional'])

with c4:
    bmi = st.slider("BMI", 12.0, 50.0, 25.0, format="%.1f")
    st.text_input("BMI Group (Auto-calculated)", value=str(get_bmi_group(bmi)), disabled=True)
    systolic_bp = st.slider("Systolic BP", 60, 260, 120)
    diastolic_bp = st.slider("Diastolic BP", 40, systolic_bp - 10, min(80, systolic_bp - 10))
    st.text_input("Blood Pressure Category (Auto-calculated)", value=get_bp_category(systolic_bp, diastolic_bp), disabled=True)

# ================= MEDICAL HISTORY & POLICY DETAILS =================

st.header("Medical History & Policy Details")
c5, c6, c7, c8 = st.columns(4)

with c5:
    visits_last_year = st.slider("Visits Last Year", 0, 25, 5)
    hospitalizations_last_3yrs = st.selectbox("Hospitalizations in Last 3 Years", [0,1,2,3],
        format_func=lambda x: {0:"Never",1:"Last Year",2:"Last 2 Years",3:"Every Year"}[x])
    days_hospitalized_last_3yrs = st.slider("Days Hospitalized Last 3 Years", 0, 30, 5)
    medication_count = st.number_input("Medication Count", 0, 10, 1)
    annual_medical_cost = st.number_input("Last Year Medical Cost", min_value=0.0, value=1000.0, format="%.2f")
    st.text_input("Medical Cost Category (Auto-calculated)", value=str(get_annual_medical_cost_category(annual_medical_cost)), disabled=True)

with c6:
    hypertension = st.selectbox("Hypertension", [0,1], format_func=lambda x: "Yes" if x else "No")
    diabetes = st.selectbox("Diabetes", [0,1], format_func=lambda x: "Yes" if x else "No")
    cardiovascular_disease = st.selectbox("Cardiovascular Disease", [0,1], format_func=lambda x: "Yes" if x else "No")
    mental_health = st.selectbox("Mental Health", [0,1], format_func=lambda x: "Yes" if x else "No")

with c7:
    asthma = st.selectbox("Asthma", [0,1], format_func=lambda x: "Yes" if x else "No")
    copd = st.selectbox("COPD", [0,1], format_func=lambda x: "Yes" if x else "No")
    kidney_disease = st.selectbox("Kidney Disease", [0,1], format_func=lambda x: "Yes" if x else "No")
    cancer_history = st.selectbox("Cancer History", [0,1], format_func=lambda x: "Yes" if x else "No")

with c8:
    deductible = st.number_input("Deductible", 0, 10000, 500)
    st.text_input("Deductible Level (Auto-calculated)", value=str(get_deductible_category(deductible)), disabled=True)
    copay = st.slider("Copay", 0, 100, 20)
    policy_term_years = st.slider("Policy Term (Years)", 1, 10, 1)
    plan_type = st.selectbox("Plan Type", list(PLAN_TYPE_SUGGESTIONS.keys()))
    st.text_area("Plan Type Explanation (Auto-generated)", value=PLAN_TYPE_SUGGESTIONS[plan_type], disabled=True, height=100)
    network_tier = st.selectbox("Network Tier", ['Platinum','Gold','Silver','Bronze'])

# ================= PREDICTION =================

if st.button("Predict Annual Premium", type="primary"):
    input_data = pd.DataFrame([[age, sex, region, urban_rural, income, education, marital_status,
                                employment_status, household_size, dependents, bmi, smoker,
                                alcohol_freq, visits_last_year, hospitalizations_last_3yrs,
                                days_hospitalized_last_3yrs, medication_count, systolic_bp,
                                diastolic_bp, ldl, hba1c, plan_type, network_tier, deductible,
                                copay, policy_term_years, annual_medical_cost, hypertension,
                                diabetes, asthma, copd, cardiovascular_disease, cancer_history,
                                kidney_disease, liver_disease, arthritis, mental_health,
                                had_major_procedure]],
                              columns=[
                                  'age', 'sex', 'region', 'urban_rural', 'income', 'education',
                                  'marital_status', 'employment_status', 'household_size',
                                  'dependents', 'bmi', 'smoker', 'alcohol_freq',
                                  'visits_last_year', 'hospitalizations_last_3yrs',
                                  'days_hospitalized_last_3yrs', 'medication_count',
                                  'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c',
                                  'plan_type', 'network_tier', 'deductible', 'copay',
                                  'policy_term_years', 'annual_medical_cost',
                                  'hypertension', 'diabetes', 'asthma', 'copd',
                                  'cardiovascular_disease', 'cancer_history',
                                  'kidney_disease', 'liver_disease', 'arthritis',
                                  'mental_health', 'had_major_procedure'
                              ])

    # dtype enforcement
    for col in ['hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease',
                'cancer_history', 'kidney_disease', 'liver_disease', 'arthritis',
                'mental_health', 'had_major_procedure']:
        input_data[col] = input_data[col].astype(np.int64)

    for col in ['age', 'household_size', 'dependents', 'visits_last_year',
                'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs',
                'medication_count', 'deductible', 'copay', 'policy_term_years']:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce').astype(np.int64)

    for col in ['income', 'bmi', 'systolic_bp', 'diastolic_bp', 'ldl',
                'hba1c', 'annual_medical_cost']:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce').astype(np.float64)

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Annual Premium: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("Developed by Shaikh Borhan Uddin")

