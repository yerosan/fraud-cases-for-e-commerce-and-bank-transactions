# src/feature_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce

class FeatureEngineer:
    def __init__(self, fraud_data):
        self.fraud_data = fraud_data
        self.scaler = StandardScaler()

    def create_time_features(self):
        self.fraud_data['signup_time'] = pd.to_datetime(self.fraud_data['signup_time'])
        self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'])
        """Create features based on the time of transaction."""
        self.fraud_data['hour_of_day'] = self.fraud_data['purchase_time'].dt.hour
        self.fraud_data['day_of_week'] = self.fraud_data['purchase_time'].dt.dayofweek
        self.fraud_data.dropna(inplace=True)
        # Initialize TargetEncoder
        target_encoder = ce.TargetEncoder(cols=['country'])
        self.fraud_data['country_encoded'] = target_encoder.fit_transform(self.fraud_data['country'], self.fraud_data['class'])

    def transaction_frequency(self):
        """Calculate transaction frequency for each user."""
        self.fraud_data['transaction_count'] = self.fraud_data.groupby('user_id').cumcount() + 1

    def normalize_and_scale(self):
        """Normalize and scale numerical features."""
        numerical_cols = ['purchase_value', 'age', 'transaction_count']
        self.fraud_data[numerical_cols] = self.scaler.fit_transform(self.fraud_data[numerical_cols])

    def encode_categorical_features(self):
        """Encode categorical features using one-hot encoding."""
        categorical_cols = ['source', 'browser', 'sex']
        self.fraud_data = pd.get_dummies(self.fraud_data, columns=categorical_cols, drop_first=True)

    def get_feature_data(self):
        self.fraud_data.drop(columns=['user_id','signup_time','purchase_time','device_id','country'], inplace=True)
        """Return the data with all features engineered, scaled, and encoded."""
        return self.fraud_data
