import streamlit as st
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("term_deposit_model.pkl")

model = load_model()

st.title("Term Deposit Subscription Predictor")
st.write("Enter client details to predict subscription likelihood.")

# 1. Collect numeric inputs
balance = st.number_input("Account Balance", value=1000)
day = st.slider("Day of Month Contacted", 1, 31, 15)
campaign = st.number_input("Number of Contacts During Campaign", value=1)
pdays = st.number_input("Days Since Last Contact", value=999)
previous = st.number_input("Number of Previous Contacts", value=0)
call_frequency = st.number_input("Estimated Calls per Month", value=1.0)
balance_age_ratio = st.number_input("Balance to Age Ratio", value=20.0)

# 2. Binary/Flag variables
pdays_flag = 1 if pdays != 999 else 0
previous_flag = 1 if previous > 0 else 0
has_missing_info = st.selectbox("Missing Contact Info?", ["No", "Yes"]) == "Yes"

# 3. Age group
age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-60", "60+"])

# 4. Job (One-hot encode)
jobs = ['blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
        'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
selected_job = st.selectbox("Job", jobs)

# 5. Marital
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])

# 6. Education
education = st.selectbox("Education Level", ["secondary", "tertiary", "unknown", "primary"])

# 7. Default, housing, loan
default = st.selectbox("Has Credit Default?", ["No", "Yes"])
housing = st.selectbox("Has Housing Loan?", ["No", "Yes"])
loan = st.selectbox("Has Personal Loan?", ["No", "Yes"])

# 8. Contact method
contact = st.selectbox("Contact Method", ["cellular", "telephone", "unknown"])

# 9. Month
month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])

# 10. Previous Outcome
poutcome = st.selectbox("Outcome of Previous Campaign", ["success", "failure", "other", "unknown"])

# === Construct Feature Vector === #
# Start with all-zero feature vector
features = ['balance', 'day', 'campaign', 'pdays', 'previous', 'pdays_flag', 'previous_flag', 'age_group', 'has_missing_info',
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed',
    'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_married',
    'marital_single', 'education_secondary', 'education_tertiary', 'education_unknown', 'default_1', 'housing_1',
    'loan_1', 'contact_telephone', 'contact_unknown', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
    'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
    'poutcome_other', 'poutcome_success', 'poutcome_unknown', 'previous_pdays_flag', 'call_frequency_per_month',
    'balance_to_age_ratio']

input_dict = dict.fromkeys(features, 0)

# Set numeric values
input_dict['balance'] = balance
input_dict['day'] = day
input_dict['campaign'] = campaign
input_dict['pdays'] = pdays
input_dict['previous'] = previous
input_dict['pdays_flag'] = pdays_flag
input_dict['previous_flag'] = previous_flag
input_dict['has_missing_info'] = int(has_missing_info)
input_dict['call_frequency_per_month'] = call_frequency
input_dict['balance_to_age_ratio'] = balance_age_ratio
input_dict['age_group'] = age_group  # Optional if model one-hot encoded it — else remove or encode manually

# One-hot encode job
job_col = f"job_{selected_job}"
if job_col in input_dict:
    input_dict[job_col] = 1

# Marital
if marital == "married":
    input_dict['marital_married'] = 1
elif marital == "single":
    input_dict['marital_single'] = 1

# Education
if education == "secondary":
    input_dict['education_secondary'] = 1
elif education == "tertiary":
    input_dict['education_tertiary'] = 1
elif education == "unknown":
    input_dict['education_unknown'] = 1

# Default/Housing/Loan
input_dict['default_1'] = int(default == "Yes")
input_dict['housing_1'] = int(housing == "Yes")
input_dict['loan_1'] = int(loan == "Yes")

# Contact method
if contact == "telephone":
    input_dict['contact_telephone'] = 1
elif contact == "unknown":
    input_dict['contact_unknown'] = 1

# Month
month_col = f"month_{month}"
if month_col in input_dict:
    input_dict[month_col] = 1

# Poutcome
if poutcome == "success":
    input_dict['poutcome_success'] = 1
elif poutcome == "other":
    input_dict['poutcome_other'] = 1
elif poutcome == "unknown":
    input_dict['poutcome_unknown'] = 1

# Previous pdays flag (could be same as pdays_flag depending on your encoding)
input_dict['previous_pdays_flag'] = pdays_flag

# Build final input array
X_input = np.array([[input_dict[feature] for feature in features]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        st.success("✅ Client is likely to SUBSCRIBE.")
    else:
        st.warning("❌ Client is NOT likely to subscribe.")
