import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load("term_deposit_model.pkl")

model = load_model()

# UI
st.title("Term Deposit Subscription Predictor")

st.write("Enter client details to predict subscription likelihood.")

# Example features: update these with your actual model features
age = st.slider("Age", 18, 95, 35)
balance = st.number_input("Account Balance", value=1000)
loan = st.selectbox("Has Personal Loan?", ["No", "Yes"])
month = st.selectbox("Last Contact Month", ["may", "jul", "nov", "jan"])

# Convert inputs to model format
loan_encoded = 1 if loan == "Yes" else 0
month_encoded = 1 if month == "may" else 0  # Simplified example

# Create feature array — adjust according to your trained model
X_input = np.array([['balance', 'day', 'campaign', 'pdays', 'previous', 'pdays_flag', 'previous_flag', 'age_group', 'has_missing_info', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_married', 'marital_single', 'education_secondary', 'education_tertiary', 'education_unknown', 'default_1', 'housing_1', 'loan_1', 'contact_telephone', 'contact_unknown', 'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep', 'poutcome_other', 'poutcome_success', 'poutcome_unknown', 'previous_pdays_flag', 'call_frequency_per_month', 'balance_to_age_ratio']])

if st.button("Predict"):
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        st.success(" Client is likely to SUBSCRIBE.")
    else:
        st.warning(" Client is NOT likely to subscribe.")
