from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model and feature list once at startup
MODEL_PATH = 'model/term_deposit_model.pkl'
FEATURES_PATH = 'model/model_features.pkl'

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

@app.route('/')
def index():
    return "Term Deposit Subscription API is live"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X = [data.get(f, 0) for f in features]
        pred = model.predict([X])[0]
        prob = model.predict_proba([X])[0][1]
        return jsonify({'prediction': int(pred), 'probability': round(prob, 4)})
    except Exception as e:
        return jsonify({'error': str(e)})
