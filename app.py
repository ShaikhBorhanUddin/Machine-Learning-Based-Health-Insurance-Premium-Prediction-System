import streamlit as st
import pandas as pd
import joblib

# ------------------ CONFIG ------------------ #
MODEL_PATH = "Models/xgboost_model_cpu.pkl"

st.set_page_config(
    page_title="Medical Insurance Premium Predictor",
    layout="wide",
    page_icon="🏥"
)

# ------------------ LOAD MODEL ------------------ #
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# 🔥 FORCE XGBoost to CPU (important for Streamlit Cloud deployment)
try:
    model.named_steps["xgboost_model"].set_params(
        predictor="cpu_predictor",
        tree_method="hist"
    )
except Exception:
    pass
# ------------------ CATEGORIES ------------------ #
categories = {
    'sex': ['Female', 'Male', 'Other'],
    'region': ['North', 'Central', 'South', 'West', 'East'],
    'urban_rural': ['Suburban', 'Urban', 'Rural'],
    'education': ['Doctorate', 'High School Dropout', 'High School', 'College', 'Master', 'Bachelor'],
    'marital_status': ['Married', 'Divorced', 'Single', 'Widowed'],
    'employment_status': ['Retired', 'Employed', 'Self-employed', 'Unemployed'],
    'smoker': ['Never', 'Former', 'Current'],
    'alcohol_freq': ['Never', 'Weekly', 'Daily', 'Occasional'],
    'plan_type': [
        'Preferred Provider Organization',
        'Point-of-Service',
        'Health Maintenance Organization',
        'Exclusive Provider Organization'
    ],
    'network_tier': ['Platinum', 'Gold', 'Silver', 'Bronze']
}

boolean_features = [
    'hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease',
    'cancer_history', 'kidney_disease', 'liver_disease', 'arthritis',
    'mental_health', 'had_major_procedure'
]

# ------------------ UI ------------------ #
st.title("🏥 Medical Insurance Premium Predictor")
st.markdown("Enter patient details to estimate the **annual insurance premium**.")

st.sidebar.header("Patient Information")

user_input = {}

# -------- Demographics -------- #
st.sidebar.subheader("Demographics")
user_input['age'] = st.sidebar.slider('Age', 15, 80, 30)
user_input['sex'] = st.sidebar.selectbox('Sex', categories['sex'])
user_input['region'] = st.sidebar.selectbox('Region', categories['region'])
user_input['urban_rural'] = st.sidebar.selectbox('Urban/Rural', categories['urban_rural'])
user_input['income'] = st.sidebar.number_input('Income ($)', 0.0, value=50000.0, step=100.0)
user_input['education'] = st.sidebar.selectbox('Education Level', categories['education'], index=3)
user_input['marital_status'] = st.sidebar.selectbox('Marital Status', categories['marital_status'])
user_input['employment_status'] = st.sidebar.selectbox('Employment Status', categories['employment_status'], index=1)
user_input['household_size'] = st.sidebar.slider('Household Size', 1, 10, 2)
user_input['dependents'] = st.sidebar.slider('Number of Dependents', 0, 8, 0)

# -------- Health Metrics -------- #
st.sidebar.subheader("Health Metrics & Lifestyle")
user_input['bmi'] = st.sidebar.slider('BMI', 15.0, 50.0, 25.0, 0.1)
user_input['smoker'] = st.sidebar.selectbox('Smoker Status', categories['smoker'])
user_input['alcohol_freq'] = st.sidebar.selectbox('Alcohol Frequency', categories['alcohol_freq'])
user_input['systolic_bp'] = st.sidebar.number_input('Systolic BP (mmHg)', 80.0, 200.0, 120.0)
user_input['diastolic_bp'] = st.sidebar.number_input('Diastolic BP (mmHg)', 40.0, 120.0, 80.0)
user_input['ldl'] = st.sidebar.number_input('LDL Cholesterol (mg/dL)', 50.0, 300.0, 100.0)
user_input['hba1c'] = st.sidebar.number_input('HbA1c (%)', 4.0, 15.0, 5.5)

# -------- Medical Utilization -------- #
st.sidebar.subheader("Medical History & Plan")
user_input['visits_last_year'] = st.sidebar.slider('Doctor Visits Last Year', 0, 15, 2)
user_input['hospitalizations_last_3yrs'] = st.sidebar.slider('Hospitalizations Last 3 Years', 0, 10, 0)
user_input['days_hospitalized_last_3yrs'] = st.sidebar.slider('Days Hospitalized Last 3 Years', 0, 30, 0)
user_input['medication_count'] = st.sidebar.slider('Number of Medications', 0, 10, 1)
user_input['annual_medical_cost'] = st.sidebar.number_input('Annual Medical Cost ($)', 0.0, 100000.0, 1000.0, step=100.0)
user_input['deductible'] = st.sidebar.number_input('Deductible ($)', 0.0, 10000.0, 500.0, step=50.0)
user_input['copay'] = st.sidebar.number_input('Co-pay ($)', 0.0, 500.0, 20.0, step=5.0)
user_input['policy_term_years'] = st.sidebar.slider('Policy Term (Years)', 1, 10, 1)
user_input['plan_type'] = st.sidebar.selectbox('Plan Type', categories['plan_type'])
user_input['network_tier'] = st.sidebar.selectbox('Network Tier', categories['network_tier'])

# -------- Conditions -------- #
st.sidebar.subheader("Pre-existing Conditions")
for feature in boolean_features:
    user_input[feature] = int(
        st.sidebar.checkbox(feature.replace('_', ' ').title(), value=False)
    )

# ------------------ PREDICTION ------------------ #
if st.button("Predict Annual Premium"):
    try:
        input_df = pd.DataFrame([user_input])

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted Annual Premium: **${prediction:,.2f}**")
        st.balloons()

        # Optional: show input summary
        with st.expander("🔎 View Input Summary"):
            st.dataframe(input_df)

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

# ------------------ FOOTER ------------------ #
st.markdown("---")
st.caption("⚠️ This tool provides an estimate only and is not medical or financial advice.")





