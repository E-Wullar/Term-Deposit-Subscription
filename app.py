import streamlit as st
import joblib
import numpy as np
import os

# Load model
@st.cache_resource
def load_model():
    return joblib.load("term_deposit_model.pkl")

model = load_model()

st.title("Term Deposit Subscription Predictor")

age = st.slider("Age", 18, 95, 35)
balance = st.number_input("Account Balance", value=1000)
loan = st.selectbox("Has Personal Loan?", ["No", "Yes"])
month = st.selectbox("Last Contact Month", ["may", "jul", "nov", "jan"])

loan_encoded = 1 if loan == "Yes" else 0
month_encoded = {"may": 1, "jul": 2, "nov": 3, "jan": 4}[month]

# Make sure this matches your training feature set
X_input = np.array([[age, balance, loan_encoded, month_encoded]])

if st.button("Predict"):
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        st.success("Client is likely to SUBSCRIBE.")
    else:
        st.warning("Client is NOT likely to subscribe.")
