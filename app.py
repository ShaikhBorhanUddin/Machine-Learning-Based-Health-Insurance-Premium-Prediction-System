import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration and File Paths ---
# Adjust these paths if your files are in a different location
MODEL_PATH = 'Models/xgboost_model.pkl'
FEATURE_NAMES_PATH = 'Models/feature_names.pkl'

# --- Load Model and Feature Names ---
@st.cache_resource
def load_model(path):
    """Loads the pre-trained model."""
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_data
def load_feature_names(path):
    """Loads the list of feature names."""
    try:
        feature_names = joblib.load(path)
        return feature_names
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return None

model = load_model(MODEL_PATH)
feature_names = load_feature_names(FEATURE_NAMES_PATH)

if model is None or feature_names is None:
    st.stop() # Stop the app if model or features failed to load

# --- Define Feature Types and Categories (based on original training data) ---
categorical_features = [
    'sex', 'region', 'urban_rural', 'education', 'marital_status',
    'employment_status', 'smoker', 'alcohol_freq', 'plan_type', 'network_tier'
]

# These categories should match those seen during training exactly
categories = {
    'sex': ['Female', 'Male'],
    'region': ['North', 'Central', 'South', 'West', 'East'],
    'urban_rural': ['Suburban', 'Urban', 'Rural'],
    'education': ['Doctorate', 'High School Dropout', 'High School', 'College', 'Master', 'Bachelor'],
    'marital_status': ['Married', 'Divorced', 'Single', 'Widowed'],
    'employment_status': ['Retired', 'Employed', 'Self-employed', 'Unemployed'],
    'smoker': ['Never', 'Former', 'Current'],
    'alcohol_freq': ['Never', 'Weekly', 'Daily', 'Occasional'],
    'plan_type': ['Preferred Provider Organization', 'Point-of-Service', 'Health Maintenance Organization', 'Exclusive Provider Organization'],
    'network_tier': ['Platinum', 'Gold', 'Silver', 'Bronze']
}

boolean_features = [
    'hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease',
    'cancer_history', 'kidney_disease', 'liver_disease', 'arthritis',
    'mental_health', 'had_major_procedure'
]

# --- Streamlit App Layout ---
st.set_page_config(page_title="Medical Insurance Premium Predictor", layout="wide")
st.title("🏥 Medical Insurance Premium Predictor")
st.markdown("Enter the patient's details to predict their annual insurance premium.")

# --- Sidebar for user input ---
st.sidebar.header("Patient Information")

user_input = {}

# Group inputs into sections for better UI
st.sidebar.subheader("Demographics")
user_input['age'] = st.sidebar.slider('Age', min_value=15, max_value=80, value=30)
user_input['sex'] = st.sidebar.selectbox('Sex', options=categories['sex'], index=0)
user_input['region'] = st.sidebar.selectbox('Region', options=categories['region'], index=0)
user_input['urban_rural'] = st.sidebar.selectbox('Urban/Rural', options=categories['urban_rural'], index=0)
user_input['income'] = st.sidebar.number_input('Yearly Income ($)', min_value=1100.0, value=1000000.0, step=100.0)
user_input['education'] = st.sidebar.selectbox('Education Level', options=categories['education'], index=3)
user_input['marital_status'] = st.sidebar.selectbox('Marital Status', options=categories['marital_status'], index=0)
user_input['employment_status'] = st.sidebar.selectbox('Employment Status', options=categories['employment_status'], index=1)
user_input['household_size'] = st.sidebar.slider('Household Size', min_value=1, max_value=10, value=2)
user_input['dependents'] = st.sidebar.slider('Number of Dependents', min_value=0, max_value=8, value=0)

st.sidebar.subheader("Health Metrics & Lifestyle")
user_input['bmi'] = st.sidebar.slider('BMI', min_value=12.0, max_value=50.0, value=25.0, step=0.1)
user_input['smoker'] = st.sidebar.selectbox('Smoker Status', options=categories['smoker'], index=0)
user_input['alcohol_freq'] = st.sidebar.selectbox('Alcohol Frequency', options=categories['alcohol_freq'], index=0)
user_input['systolic_bp'] = st.sidebar.slider('Systolic BP (mmHg)', min_value=60.0, max_value=260.0, value=120.0, step=1.0)
user_input['diastolic_bp'] = st.sidebar.slider('Diastolic BP (mmHg)', min_value=40.0, max_value=180.0, value=80.0, step=1.0)
user_input['ldl'] = st.sidebar.slider('LDL Cholesterol (mg/dL)', min_value=30.0, max_value=250.0, value=100.0, step=1.0)
user_input['hba1c'] = st.sidebar.slider('HbA1c (%)', min_value=3.0, max_value=15.0, value=5.5, step=0.1)

st.sidebar.subheader("Medical History & Plan Details")
user_input['visits_last_year'] = st.sidebar.slider('Doctor Visits Last Year', min_value=0, max_value=15, value=2)
user_input['hospitalizations_last_3yrs'] = st.sidebar.slider('Hospitalizations Last 3 Years', min_value=0, max_value=10, value=0)
user_input['days_hospitalized_last_3yrs'] = st.sidebar.slider('Days Hospitalized Last 3 Years', min_value=0, max_value=30, value=0)
user_input['medication_count'] = st.sidebar.slider('Number of Medications', min_value=0, max_value=10, value=1)
user_input['annual_medical_cost'] = st.sidebar.number_input('Annual Medical Cost (excluding premium) ($)', min_value=0.0, value=10000.0, step=100.0)
user_input['deductible'] = st.sidebar.number_input('Deductible ($)', min_value=1000.0, max_value=5000.0, value = 1000.0, step=50.0)
user_input['copay'] = st.sidebar.number_input('Co-pay ($)', min_value=10.0, max_value = 50.0, value=20.0, step=5.0)
user_input['policy_term_years'] = st.sidebar.slider('Policy Term (Years)', min_value=1, max_value=10, value=1, step=1)
user_input['plan_type'] = st.sidebar.selectbox('Plan Type', options=categories['plan_type'], index=0)
user_input['network_tier'] = st.sidebar.selectbox('Network Tier', options=categories['network_tier'], index=0)

st.sidebar.subheader("Pre-existing Conditions & Procedures")
for feature in boolean_features:
    user_input[feature] = st.sidebar.checkbox(feature.replace('_', ' ').title(), value=False)

# --- Prediction button ---
if st.button('Predict Annual Premium'):
    try:
        # Create a DataFrame from user inputs, ensuring correct order and types
        # Convert boolean values to int (0 or 1)
        for bf in boolean_features:
            user_input[bf] = int(user_input[bf])

        input_df = pd.DataFrame([user_input])

        # Ensure the columns are in the same order as feature_names
        input_df = input_df[feature_names]

        # Make prediction
        predicted_premium = model.predict(input_df)[0]

        st.success(f"Predicted Annual Premium: ${predicted_premium:.2f}")
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)

st.markdown("""
--- 
*Disclaimer: This is a predictive model for estimation purposes only and should not be used as actual medical or financial advice.*
""")
