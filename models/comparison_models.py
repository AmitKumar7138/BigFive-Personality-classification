import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results = []

    def preprocess_data(self, df):
        # The last 5 columns are labels ('cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN')
        X = df.iloc[:, :-5]  # Extract word2vecs
        y = df[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]  # Extract labels
        return X, y

    def train_svm(self, X_train, y_train, X_test, y_test):
        svm_model = OneVsRestClassifier(
            SVC(kernel='rbf', C=1, gamma=0.1, degree=1))
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        self.evaluate_and_save_results("SVM", y_test, svm_predictions)

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        xgb_model = xgb.XGBClassifier(
            max_depth=10, n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        xgb_predictions = xgb_model.predict(X_test)
        # Ensure predictions are integers
        xgb_predictions = xgb_predictions.astype(int)
        self.evaluate_and_save_results("XGBoost", y_test, xgb_predictions)

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        rf_model = RandomForestClassifier(
            n_estimators=100, criterion='gini', min_samples_split=2)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        self.evaluate_and_save_results("Random_Forest", y_test, rf_predictions)

    def evaluate_and_save_results(self, Model, y_true, y_pred):
        # Convert each row in the arrays into a separate list inside a larger list
        # Convert each row in the DataFrame y_true into a list
        y_values_list = [list(row) for _, row in y_true.iterrows()]

        # Convert each row in the NumPy array y_pred into a list
        predicted_values_list = [list(row) for row in y_pred]

        # Create pandas Series from these lists of lists
        y_values_series = pd.Series(y_values_list)
        predicted_values_series = pd.Series(predicted_values_list)

        # Create DataFrame
        df_vals = pd.DataFrame({
            'Y_values': y_values_series,
            'predicted_values': predicted_values_series
        })

        # Save the DataFrame as a CSV file
        df_vals.to_csv(f'{Model}_output.csv', index=False)
