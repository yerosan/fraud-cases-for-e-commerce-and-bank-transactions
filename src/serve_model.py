from flask import Flask, request, jsonify
import joblib  # Assuming you saved the model with joblib
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
app = Flask(__name__)

# Load your trained model
model = joblib.load("../models/RandomForestClassifier_model.pkl")
data="../Data/test.csv"
# Setup logging
handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Fraud Detection API!"})
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
        # Convert boolean fields to integers (0 or 1)
    data['source_Direct'] = int(data['source_Direct'])
    data['source_SEO'] = int(data['source_SEO'])
    data['browser_FireFox'] = int(data['browser_FireFox'])
    data['browser_IE'] = int(data['browser_IE'])
    data['browser_Opera'] = int(data['browser_Opera'])
    data['browser_Safari'] = int(data['browser_Safari'])
    data['sex_M'] = int(data['sex_M'])

    # Prepare the input for the model (ensure it's in the right format)
    input_features = [
        data["Unnamed_0.1"],
        data["Unnamed_0"],
        data["purchase_value"],
        data["age"],
        data["ip_address"],
        data["signup_hour"],
        data["purchase_hour"],
        data["signup_day"],
        data["purchase_day"],
        data["hour_of_day"],
        data["day_of_week"],
        data["country_encoded"],
        data["transaction_count"],
        data["source_Direct"],
        data["source_SEO"],
        data["browser_FireFox"],
        data["browser_IE"],
        data["browser_Opera"],
        data["browser_Safari"],
        data["sex_M"]
    ]
    print("----------__________", input_features)
    app.logger.info(f"Incoming request data: {input_features}")
    
    try:
        features = data

        # Make prediction (ensure model expects a 2D array)
        prediction = model.predict([input_features])

        # Convert the prediction to a native Python type
        prediction_value = prediction[0].item() if isinstance(prediction[0], np.generic) else prediction[0]

        return jsonify({"prediction": prediction_value})
        # prediction = model.predict([input_features])  # Replace with actual input format
        # app.logger.info(f"Prediction: {prediction[0]}")
        # return jsonify({"prediction": prediction[0]})
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
