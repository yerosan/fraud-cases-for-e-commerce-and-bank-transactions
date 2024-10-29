# src/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import tensorflow as tf
import joblib  # Assuming you saved the model with joblib
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM

class ModelTrainer:
    
    def __init__(self, data, model_name='FraudDetection'):
        self.data = data
        self.model_name = model_name

    def prepare_data(self, target_col='class'):
        """Separate features and target, perform train-test split."""
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return self.X_train,self.X_test
    def train_sklearn_model(self, model):
        """Train a scikit-learn model, log results in MLflow, and return the trained model."""
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        # Specify the directory to save the model
        model_directory = "../models"
        os.makedirs(model_directory, exist_ok=True)

        # Save the trained model to disk
        model_filename = f"{model_directory}/{model.__class__.__name__}_model.pkl"
        joblib.dump(model, model_filename)  # Save model as .pkl file
        print(f"Model saved as {model_filename}")
        
        # Calculate and log metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"{model.__class__.__name__} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"{model.__class__.__name__}"):
            mlflow.log_param("model_type", model.__class__.__name__)
            mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})
        
        # Return the trained model instance
        return model


    def train_neural_network(self, model_type="CNN"):
        """Train a neural network model and log results in MLflow."""
        # Define CNN or RNN model architectures
        model = Sequential()
        if model_type == "CNN":
            model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.X_train.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        elif model_type in ["RNN", "LSTM"]:
            if model_type == "LSTM":
                model.add(LSTM(64, input_shape=(self.X_train.shape[1], 1)))
            else:
                model.add(tf.keras.layers.SimpleRNN(64, input_shape=(self.X_train.shape[1], 1)))
        
        # Add dense layers and output
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train and evaluate the model
        model.fit(self.X_train, self.y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
        _, accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Log to MLflow
        with mlflow.start_run(run_name=model_type):
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("accuracy", accuracy)
    
    def run_all_models(self):
        """Train and evaluate all models, logging results to MLflow."""
        models = [
            LogisticRegression(max_iter=200),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            MLPClassifier(max_iter=300)
        ]
        
        for model in models:
            self.train_sklearn_model(model)
        
        # Train deep learning models
        self.train_neural_network("CNN")
        self.train_neural_network("RNN")
        self.train_neural_network("LSTM")
