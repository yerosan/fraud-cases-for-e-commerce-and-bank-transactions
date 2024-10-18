import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessing:
    def __init__(self):
         pass
    
    def data_reading(self,data_path):
        return pd.read_csv(data_path)
    
    def handle_missing_values(self,df):
        # Dropping rows with missing values (can also use df.fillna() for imputation)
        missing_values = df.isnull().sum()
        print("Missing values per column:\n", missing_values)
        df_clean = df.dropna()
        return df_clean
    

    def clean_data(self,df):
        # Remove duplicate rows
        df = df.drop_duplicates()

        # Convert the 'signup_time' and 'purchase_time' columns to datetime
        # Convert to datetime
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Verify the changes
        print(df[['signup_time', 'purchase_time']].dtypes)

        
        return df
    
    def Summary_statistics(self, df):
        # Summary statistics for numeric columns
        print(df.describe())
    def Visualize_class_distribution(self, df):
        # Visualize the class distribution
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        sns.countplot(x='class', data=df, palette="Set1")
        plt.title('Fraud vs Non-Fraud Transaction Distribution')
        plt.xlabel('Transaction Class (0: Non-Fraud, 1: Fraud)')
        plt.ylabel('Count')
        plt.show()
    def class_distribution_percentages(self, df):
        # Print class distribution percentages
        class_distribution = df['class'].value_counts(normalize=True) * 100
        print(f'Class distribution (fraud vs non-fraud):\n{class_distribution}')

    def Univariate_analysis(self, df):
        # Univariate analysis for fraud_data
        plt.figure(figsize=(10, 6))
        sns.histplot(df['purchase_value'], bins=50, kde=True)
        plt.title('Distribution of Purchase Values')

    def Bivariate_analysis(self, df):
        # Bivariate analysis: Fraud class vs. device
        sns.countplot(x='device_id', hue='class', data=df)
        plt.title('Fraud Class Across Different Devices')
        plt.xticks(rotation=90)
        plt.show()
    
    def Visualize_top_10_countries(self,df):
        # Visualize the top 10 countries by transaction count
        plt.figure(figsize=(14, 7))
        country_counts = df['browser'].value_counts().head(10)
        sns.barplot(x=country_counts.index, y=country_counts.values, palette="coolwarm")
        plt.title('Top 10 Countries by Number of Transactions')
        plt.xlabel('Country')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()
    
    def  Visualize_fraud_rates_by_country(self,df):

        # Visualize fraud rates by country
        plt.figure(figsize=(14, 7))
        fraud_by_country = df.groupby('country')['class'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=fraud_by_country.index, y=fraud_by_country.values, palette="Reds")
        plt.title('Top 10 Countries with Highest Fraud Rates')
        plt.xlabel('Country')
        plt.ylabel('Fraud Rate')
        plt.xticks(rotation=45)
        plt.show()
    
    def time_based_insights(self, df):
        # Create hour of the day and day of the week features
        df['signup_hour'] = df['signup_time'].dt.hour
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['signup_day'] = df['signup_time'].dt.dayofweek
        df['purchase_day'] = df['purchase_time'].dt.dayofweek

        # Plot fraud cases by hour of purchase
        plt.figure(figsize=(10, 6))
        sns.countplot(x='purchase_hour', hue='class', data=df, palette="Set2")
        plt.title('Fraud Cases by Hour of Purchase')
        plt.xlabel('Hour of Purchase')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()

        # Plot fraud cases by day of purchase
        plt.figure(figsize=(10, 6))
        sns.countplot(x='purchase_day', hue='class', data=df, palette="Set3")
        plt.title('Fraud Cases by Day of Purchase')
        plt.xlabel('Day of the Week (0: Monday, 6: Sunday)')
        plt.ylabel('Number of Transactions')
        plt.show()

    def analyze_categorical_variables (self,df):

        # Analyze source vs fraud
        plt.figure(figsize=(10, 6))
        sns.countplot(x='source', hue='class', data=df, palette="Set1")
        plt.title('Fraud Cases by Source of Traffic')
        plt.xlabel('Source')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()

        # Analyze browser vs fraud
        plt.figure(figsize=(10, 6))
        sns.countplot(x='browser', hue='class', data=df, palette="Set2")
        plt.title('Fraud Cases by Browser')
        plt.xlabel('Browser')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()

        # Analyze sex vs fraud
        plt.figure(figsize=(8, 5))
        sns.countplot(x='sex', hue='class', data=df, palette="Set3")
        plt.title('Fraud Cases by Gender')
        plt.xlabel('Sex')
        plt.ylabel('Number of Transactions')
        plt.show()


    def feature_engineering(self,df):
        # Transaction frequency
        df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        
        # Time-based features: Extract hour of day and day of the week
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek

        return df


    def encode_categorical(df, categorical_columns):
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        return df_encoded



    

    


