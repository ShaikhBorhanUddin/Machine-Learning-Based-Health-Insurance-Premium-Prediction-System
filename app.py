import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Assuming the model file 'xgboost_model_cpu.pkl' is in the same directory as app.py
try:
    model = joblib.load('Models/xgboost_model_cpu.pkl')
except FileNotFoundError:
    st.error("Model file 'xgboost_model_cpu.pkl' not found. Please ensure it's in the same directory as app.py.")
    st.stop()

st.set_page_config(page_title="Medical Insurance Premium Predictor", layout="wide")
st.title("🏥 Medical Insurance Premium Prediction")
st.markdown("Enter the details below to predict the annual medical insurance premium.")

st.header("Personal & Health Details")

col1, col2, col3, col4 = st.columns(4)

# ================= PERSONAL INFO (COL 1) =================
with col1:
    st.subheader(" ")

    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ['Female', 'Male', 'Other'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Divorced', 'Single', 'Widowed'])
    household_size = st.slider("Household Size", 1, 10, 2)
    dependents = st.slider("Dependents", 0, 9, 1)

# ================= PERSONAL INFO (COL 2) =================
with col2:
    st.subheader(" ")

    education = st.selectbox("Education Level", ['Doctorate', 'High School Dropout', 'High School', 'College', 'Masters', 'Bachelors'])
    income = st.number_input("Income in USD", min_value=0.0, value=1000000.0, format="%.2f")
    employment_status = st.selectbox("Employment Status", ['Retired', 'Employed', 'Self-employed', 'Unemployed'])
    region = st.selectbox("Region", ['North', 'Central', 'West', 'East', 'South'])
    urban_rural = st.selectbox("Geography", ['Suburban', 'Urban', 'Rural'])


# ================= HEALTH METRICS (COL 3) =================
with col3:
    st.subheader(" ")

    bmi = st.slider("BMI", 12.0, 50.0, 25.0, format="%.1f")
    smoker = st.selectbox("Smoking Habit", ['Never', 'Former', 'Current'])
    alcohol_freq = st.selectbox("Alcohol Consumption", ['Never', 'Weekly', 'Daily', 'Occasional'])
    systolic_bp = st.slider("Systolic BP", 60, 260, 120)
    diastolic_bp = st.slider("Diastolic BP", min_value=40, max_value=systolic_bp - 1, value=min(80, systolic_bp - 1))

# ================= HEALTH METRICS (COL 4) =================
with col4:
    st.subheader(" ")

    ldl = st.number_input("LDL", min_value=0.0, value=100.0, format="%.1f")
    hba1c = st.number_input("HbA1c", min_value=0.0, value=5.5, format="%.2f")
    medication_count = st.number_input("Medication Count", min_value=0, max_value=10, value=1)
    annual_medical_cost = st.number_input("Annual Medical Cost", min_value=0.0, value=1000.0, format="%.2f")
    
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
    had_major_procedure = st.selectbox("Had Major Procedure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col18:
    st.subheader(" ")

    deductible = st.number_input("Deductible", min_value=0, max_value=5000, value=500)
    copay = st.number_input("Copay", min_value=0, max_value=100, value=20)
    policy_term_years = st.slider("Policy Term (Years)", min_value=1, max_value=10, value=1)
    plan_type = st.selectbox("Plan Type", options=['Preferred Provider Organization', 'Point-of-Service', 'Health Maintenance Organization', 'Exclusive Provider Organization'])
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












