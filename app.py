from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Dummy root route for health check
@app.route('/')
def index():
    return "Term Deposit Subscription API is live"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model = joblib.load('model/term_deposit.pkl')
    features = joblib.load('model/model_features.pkl')
    X = [data.get(f, 0) for f in features]
    pred = model.predict([X])[0]
    prob = model.predict_proba([X])[0][1]
    return jsonify({'prediction': str(pred), 'probability': prob})
