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

PLAN_TYPE_SUGGESTIONS = {
    "Health Maintenance Organization": (
        "Choose HMO if you want the lowest monthly costs and don't mind using a "
        "primary care doctor to manage your care."
    ),
    "Preferred Provider Organization": (
        "Choose PPO if you want the freedom to see specialists without referrals "
        "and access out-of-network care."
    ),
    "Exclusive Provider Organization": (
        "Choose EPO if you want lower premiums like an HMO but don't want referrals "
        "for specialists."
    ),
    "Point-of-Service": (
        "Choose POS if you want the cost savings of an HMO but want the option to "
        "go out-of-network."
    )
}

st.header("Personal & Health Details")

col1, col2, col3, col4 = st.columns(4)

# ================= PERSONAL INFO (COL 1) =================

with col1:
    st.subheader(" ")

    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Divorced', 'Single', 'Widowed'])
    household_size = st.slider("Household Size", 1, 10, 2)
    dependents = st.slider("Dependents", 0, 9, 1)

# ================= PERSONAL INFO (COL 2) =================

with col2:
    st.subheader(" ")

    education = st.selectbox("Education Level", ['College', 'Doctorate', 'High School Dropout', 'High School', 'Masters', 'Bachelors'])
    income = st.number_input("Yearly Income", 10000, 10000000, 12000)
    employment_status = st.selectbox("Employment Status", ['Employed', 'Self-employed', 'Retired', 'Unemployed'])
    region = st.selectbox("Geographical Region", ['North', 'Central', 'West', 'East', 'South'])
    urban_rural = st.selectbox("Area", ['Suburban', 'Urban', 'Rural'])

# ================= HEALTH METRICS (COL 3) =================

with col3:
    st.subheader(" ")

    bmi = st.slider("BMI", 12.0, 50.0, 25.0, format="%.1f")
    bmi_group = get_bmi_group(bmi)
    st.text_input("BMI Group (Auto-calculated)", value=str(bmi_group), disabled=True)
    smoker = st.selectbox("Smoking Habit", ['Never', 'Former', 'Current'])
    alcohol_freq = st.selectbox("Alcohol Consumption", ['Never', 'Weekly', 'Daily', 'Occasional'])
    systolic_bp = st.slider("Systolic BP", 60, 260, 120)
    diastolic_bp = st.slider("Diastolic BP", min_value=40, max_value=systolic_bp - 10, value=min(80, systolic_bp - 10))
    bp_category = get_bp_category(systolic_bp, diastolic_bp)
    st.text_input("Blood Pressure Category (Auto-calculated)", value=bp_category, disabled=True)
    
# ================= HEALTH METRICS (COL 4) =================

with col4:
    st.subheader(" ")

    ldl = st.number_input("LDL", min_value=0.0, value=100.0, format="%.1f")
    ldl_category = get_ldl_category(ldl)
    st.text_input("LDL Category (Auto-calculated)", value=str(ldl_category), disabled=True)
    hba1c = st.number_input("HbA1c", min_value=0.0, value=5.5, format="%.2f")
    hba1c_category = get_hba1c_category(hba1c)
    st.text_input("HbA1c Category (Auto-calculated)", value=str(hba1c_category), disabled=True)
    annual_medical_cost = st.number_input("Annual Medical Cost", min_value=0.0, value=1000.0, format="%.2f")
    annual_medical_cost_category = get_annual_medical_cost_category(annual_medical_cost)
    st.text_input("Annual Medical Cost Category (Auto-calculated)", value=str(annual_medical_cost_category), disabled=True)
    
st.header("Medical History & Policy Details")

col5, col16, col17, col18 = st.columns(4)

with col5:
    st.subheader(" ")

    visits_last_year = st.slider("Visits Last Year", min_value=0, max_value=20, value=5)
    hospitalizations_last_3yrs_options = {0: "Never", 1: "Last Year", 2: "Last 2 Years", 3: "Every Year"}
    hospitalizations_last_3yrs = st.selectbox("Hospitalizations in Last 3 Years", options=list(hospitalizations_last_3yrs_options.keys()), format_func=lambda x: hospitalizations_last_3yrs_options[x])
    days_hospitalized_last_3yrs = st.slider("Days Hospitalized Last 3 Years", min_value=0, max_value=30, value=0)
    hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col16:
    st.subheader(" ")

    asthma = st.selectbox("Asthma", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    copd = st.selectbox("COPD", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cardiovascular_disease = st.selectbox("Cardiovascular Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cancer_history = st.selectbox("Cancer History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    kidney_disease = st.selectbox("Kidney Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col17:
    st.subheader(" ")

    liver_disease = st.selectbox("Liver Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    arthritis = st.selectbox("Arthritis", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    mental_health = st.selectbox("Mental Health", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    medication_count = st.number_input("Medication Count", min_value=0, max_value=10, value=1)
    had_major_procedure = st.selectbox("Had Major Procedure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col18:
    st.subheader(" ")

    deductible = st.number_input("Deductible", min_value=0, max_value=5000, value=500)
    copay = st.number_input("Copay", min_value=0, max_value=100, value=20)
    policy_term_years = st.slider("Policy Term (Years)", min_value=1, max_value=10, value=1)
    plan_type = st.selectbox("Plan Type", options=['Preferred Provider Organization', 'Point-of-Service', 'Health Maintenance Organization', 'Exclusive Provider Organization'])
    plan_suggestion = PLAN_TYPE_SUGGESTIONS.get(plan_type, "")
    st.text_area("Plan Type Recommendation (Auto-generated)", value=plan_suggestion, disabled=True, height=100)
    network_tier = st.selectbox("Network Tier", options=['Platinum', 'Gold', 'Silver', 'Bronze'])


if st.button("Predict Annual Premium", type="primary"):
    # Create a DataFrame from inputs, ensuring correct column order and dtypes
    input_data = pd.DataFrame([[age, sex, region, urban_rural, income, education, marital_status,
                                employment_status, household_size, dependents, bmi, smoker,
                                alcohol_freq, visits_last_year, hospitalizations_last_3yrs,
                                days_hospitalized_last_3yrs, medication_count, systolic_bp,
                                diastolic_bp, ldl, hba1c, plan_type, network_tier, deductible,
                                copay, policy_term_years, annual_medical_cost, hypertension,
                                diabetes, asthma, copd, cardiovascular_disease, cancer_history,
                                kidney_disease, liver_disease, arthritis, mental_health,
                                had_major_procedure]],
                              columns=['age', 'sex', 'region', 'urban_rural', 'income', 'education',
                                       'marital_status', 'employment_status', 'household_size',
                                       'dependents', 'bmi', 'smoker', 'alcohol_freq',
                                       'visits_last_year', 'hospitalizations_last_3yrs',
                                       'days_hospitalized_last_3yrs', 'medication_count',
                                       'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c', 'plan_type',
                                       'network_tier', 'deductible', 'copay', 'policy_term_years',
                                       'annual_medical_cost', 'hypertension', 'diabetes', 'asthma',
                                       'copd', 'cardiovascular_disease', 'cancer_history',
                                       'kidney_disease', 'liver_disease', 'arthritis',
                                       'mental_health', 'had_major_procedure'])

    # Ensure dtypes are correct for prediction
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







