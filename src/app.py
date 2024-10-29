from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load fraud data
data = pd.read_csv("../Data/Fraud_Data.csv")

# Helper functions to calculate insights
def get_summary_stats():
    total_transactions = len(data)
    fraud_cases = data[data['class'] == 1].shape[0]
    fraud_percentage = (fraud_cases / total_transactions) * 100
    return {
        "total_transactions": total_transactions,
        "fraud_cases": fraud_cases,
        "fraud_percentage": fraud_percentage
    }

def get_fraud_trends():
    fraud_trend = data[data['class'] == 1].groupby('date')['class'].count().reset_index()
    fraud_trend.columns = ['date', 'fraud_cases']
    return fraud_trend.to_dict(orient='records')

def get_fraud_by_geography():
    return data[data['class'] == 1].groupby('country')['class'].count().reset_index().to_dict(orient='records')

def get_fraud_by_device_browser():
    return data[data['class'] == 1].groupby(['device', 'browser'])['class'].count().reset_index().to_dict(orient='records')

# Flask endpoints
@app.route('/api/summary_stats', methods=['GET'])
def summary_stats():
    return jsonify(get_summary_stats())

@app.route('/api/fraud_trends', methods=['GET'])
def fraud_trends():
    return jsonify(get_fraud_trends())

@app.route('/api/fraud_by_geography', methods=['GET'])
def fraud_by_geography():
    return jsonify(get_fraud_by_geography())

@app.route('/api/fraud_by_device_browser', methods=['GET'])
def fraud_by_device_browser():
    return jsonify(get_fraud_by_device_browser())

if __name__ == "__main__":
    app.run(debug=True, port=5000)
