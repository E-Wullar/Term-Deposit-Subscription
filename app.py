from flask import Flask, request, jsonify
import joblib
import pandas as pd
import xgboost as xgb

# Load model and features
model = joblib.load("model/term_deposit_model.pkl")
features = joblib.load("model/model_features.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Term Deposit Subscription Prediction API is live!"

def predict():
    try:
        # Get JSON input
        input_data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Keep only trained model features
        input_df = input_df[features]

        # Ensure numeric input
        input_df = input_df.apply(pd.to_numeric)

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
    "prediction": str(prediction),
    "probability": float(round(probability, 4))  
})

        

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
