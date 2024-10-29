# Fraud Detection for E-commerce and Bank Transactions

## Overview
This project provides a machine learning pipeline and a visualization dashboard for detecting and analyzing fraud in e-commerce and bank transactions. It uses machine learning models to classify transactions as fraudulent or legitimate and offers an interactive web-based dashboard to visualize insights.

## Key Features
- **End-to-end ML Pipeline**: Includes data preprocessing, feature engineering, model training, and evaluation.
- **Explainability**: Utilizes SHAP and LIME for model interpretability.
- **Real-time Prediction API**: Flask API to serve the trained model for predictions on new data.
- **Interactive Dashboard**: Dash-based dashboard for exploring insights and trends.

## Project Structure

```bash
├── .vscode/                
│   └── settings.json       
├── .github/                
│   └── workflows/          
│       ├── unittests.yml   
├── .gitignore              
├── README.md               # Project Overview and Instructions
├── requirements.txt        # Dependencies
├── src/                    
│   ├── __init__.py         
├── notebooks/              
│   ├── __init__.py         
│   └── EDA_and_Preprocessing.ipynb   # Jupyter notebook for EDA
├── tests/                  
│   ├── __init__.py         
└── scripts/                
    ├── __init__.py         
    └── data_preprocessing.py   # Script for data preprocessing
    ├── model_bulding.py.py        # Model training and evaluation
    |   featureEnginerring.py      # feature engineering

├── src/
│   ├── dashboard.py            # Interactive Dash dashboard
│   └── serve_model.py          # Flask app for serving the model API
├── data/
│   ├── raw/                    # Raw data files
├── models/                     # Saved trained models
├── requirements.txt            # Dependencies for the project
├── README.md                   # Project documentation
└── Dockerfile                  # Docker configuration for deployment

```
## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation

#### Clone the Repository:
```bash
git clone https://github.com/yerosan/fraud-cases-for-e-commerce-and-bank-transactions.git
cd fraud-cases-for-e-commerce-and-bank-transactions
```
### Install Dependencies:
``` bash
pip install -r requirements.txt
```
### Set Up and Configure the Environment:
Place your data files in the data/raw/ directory. Ensure your configuration (e.g., API keys, paths) is set if needed.
## Model Insights
Four models were trained and evaluated for fraud detection, with the following performance metrics:

| Model                      | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| DecisionTreeClassifier      | 0.8971   | 0.4679    | 0.5737 | 0.5154   |
| RandomForestClassifier      | 0.9562   | 0.9963    | 0.5424 | 0.7024   |
| GradientBoostingClassifier  | 0.9064   | 0.9600    | 0.0195 | 0.0382   |
| MLPClassifier               | 0.9046   | 0.0000    | 0.0000 | 0.0000   |

The **RandomForestClassifier** demonstrated the best balance of accuracy and F1 score and was saved as the final model (`models/RandomForestClassifier_model.pkl`).

## API Endpoints
The Flask API provides several endpoints for model predictions and insights:
- **GET /predict:** Predicts if a transaction is fraudulent.
- **GET /summary:** Provides summary statistics on transactions.
- **GET /fraud-trends:** Returns fraud trends over time for visualization.

### Sample Request
To make a prediction, send a POST request to `/predict`:
```json
{
  "purchase_value": 120.5,
  "age": 30,
  "ip_address": "192.168.0.1",
  ...
}
```
## Dashboard Insights
The dashboard includes:
- **Summary Boxes:** Displays total transactions, fraud cases, and fraud percentages.
- **Line Chart:** Shows fraud trends over time.
- **Geographic Map:** Highlights fraud distribution by region.
- **Bar Chart:** Compares fraud cases by device and browser type.

## Model Explainability
Model explainability is achieved through SHAP and LIME:
- **SHAP Summary and Force Plots:** Highlights the most influential features.
- **LIME:** Explains individual predictions locally.
