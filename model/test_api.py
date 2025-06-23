import requests

url = "http://127.0.0.1:5000/predict"

sample_input = {
    "balance": 1500,
    "day": 15,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "pdays_flag": 0,
    "previous_flag": 0,
    "age_group": 2,  # for example: 0=young, 1=middle, 2=older (depends on how you encoded it)
    "has_missing_info": 0,

    # One-hot encoded job types
    "job_blue-collar": 0,
    "job_entrepreneur": 0,
    "job_housemaid": 0,
    "job_management": 1,
    "job_retired": 0,
    "job_self-employed": 0,
    "job_services": 0,
    "job_student": 0,
    "job_technician": 0,
    "job_unemployed": 0,
    "job_unknown": 0,

    # Marital status
    "marital_married": 1,
    "marital_single": 0,

    # Education
    "education_secondary": 0,
    "education_tertiary": 1,
    "education_unknown": 0,

    # Binary indicators
    "default_1": 0,
    "housing_1": 1,
    "loan_1": 0,

    # Contact type
    "contact_telephone": 0,
    "contact_unknown": 0,

    # Month of last contact
    "month_aug": 0,
    "month_dec": 0,
    "month_feb": 0,
    "month_jan": 0,
    "month_jul": 0,
    "month_jun": 0,
    "month_mar": 0,
    "month_may": 1,
    "month_nov": 0,
    "month_oct": 0,
    "month_sep": 0,

    # Outcome of previous marketing campaign
    "poutcome_other": 0,
    "poutcome_success": 0,
    "poutcome_unknown": 1,

    # Derived features
    "previous_pdays_flag": 0,
    "call_frequency_per_month": 0.5,
    "balance_to_age_ratio": 50.0
}

response = requests.post(url, json=sample_input)
print(response.json())
