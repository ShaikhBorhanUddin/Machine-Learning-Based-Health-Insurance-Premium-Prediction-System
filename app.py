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

st.set_page_config(page_title="Medical Insurance Premium Predictor", layout="centered")
st.title("🏥 Medical Insurance Premium Prediction")
st.markdown("Enter the details below to predict the annual medical insurance premium.")

st.header("Personal Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", options=['Female', 'Male'])
    marital_status = st.selectbox("Marital Status", options=['Married', 'Divorced', 'Single', 'Widowed'])
    household_size = st.number_input("Household Size", min_value=1, max_value=10, value=2)
    dependents = st.number_input("Dependents", min_value=0, max_value=9, value=1)
    education = st.selectbox("Education", options=['Doctorate', 'High School Dropout', 'High School', 'College', 'Masters', 'Bachelors'])
with col2:
    income = st.number_input("Income", min_value=0.0, value=50000.0, format="%.2f")
    employment_status = st.selectbox("Employment Status", options=['Retired', 'Employed', 'Self-employed', 'Unemployed', 'Student'])
    region = st.selectbox("Region", options=['North', 'Central', 'West', 'East', 'South'])
    urban_rural = st.selectbox("Urban/Rural", options=['Suburban', 'Urban', 'Rural'])
    
st.header("Health Metrics & Habits")
col3, col4 = st.columns(2)
with col3:
    bmi = st.number_input("BMI", min_value=0.0, value=25.0, format="%.1f")
    smoker = st.selectbox("Smoker", options=['Never', 'Former', 'Current'])
    alcohol_freq = st.selectbox("Alcohol Frequency", options=['Never', 'Weekly', 'Daily', 'Occasional', 'Seldom'])
    systolic_bp = st.number_input("Systolic BP", min_value=0.0, value=120.0, format="%.1f")
    diastolic_bp = st.number_input("Diastolic BP", min_value=0.0, value=80.0, format="%.1f")
with col4:
    ldl = st.number_input("LDL", min_value=0.0, value=100.0, format="%.1f")
    hba1c = st.number_input("HbA1c", min_value=0.0, value=5.5, format="%.2f")
    medication_count = st.number_input("Medication Count", min_value=0, max_value=10, value=1)
    annual_medical_cost = st.number_input("Annual Medical Cost", min_value=0.0, value=1000.0, format="%.2f")

st.header("Medical History & Policy Details")
col5, col6 = st.columns(2)
with col5:
    visits_last_year = st.number_input("Visits Last Year", min_value=0, max_value=20, value=5)
    hospitalizations_last_3yrs = st.number_input("Hospitalizations Last 3 Years", min_value=0, max_value=10, value=0)
    days_hospitalized_last_3yrs = st.number_input("Days Hospitalized Last 3 Years", min_value=0, max_value=30, value=0)
    hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    asthma = st.selectbox("Asthma", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    copd = st.selectbox("COPD", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cardiovascular_disease = st.selectbox("Cardiovascular Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col6:
    cancer_history = st.selectbox("Cancer History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    kidney_disease = st.selectbox("Kidney Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    liver_disease = st.selectbox("Liver Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    arthritis = st.selectbox("Arthritis", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    mental_health = st.selectbox("Mental Health", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    had_major_procedure = st.selectbox("Had Major Procedure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    deductible = st.number_input("Deductible", min_value=0, max_value=5000, value=500)
    copay = st.number_input("Copay", min_value=0, max_value=100, value=20)
    policy_term_years = st.number_input("Policy Term (Years)", min_value=1, max_value=10, value=1)
    plan_type = st.selectbox("Plan Type", options=['Preferred Provider Organization', 'Point-of-Service', 'Health Maintenance Organization', 'Exclusive Provider Organization', 'High Deductible Health Plan'])
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
st.markdown("Developed with ❤️ by Your Name/Team")



