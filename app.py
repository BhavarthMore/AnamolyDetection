import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Define the Custom Isolation Forest class
class CustomIsolationForest:
    def __init__(self, contamination=0.1, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        
    def load(self, model_path):
        self.model = joblib.load(model_path)
        
    def predict(self, X):
        # Predict the outliers (-1) and inliers (1)
        raw_predictions = self.model.predict(X)
        # Convert Isolation Forest output to binary labels: -1 -> 1 (fraud), 1 -> 0 (normal)
        converted_predictions = np.where(raw_predictions == -1, 1, 0)
        return converted_predictions

# Title of the app
st.title("Anomaly Detection with Isolation Forest")

# File uploader for the test dataset
uploaded_file = st.file_uploader("Upload your test dataset (CSV file)")

if uploaded_file is not None:
    # Read the uploaded CSV file
    creditcard_test = pd.read_csv(uploaded_file)

    # Create a section for the dataframe header
    st.header('Header of Dataframe')
    st.write(creditcard_test.head())

    # Check and clean the data
    for column in creditcard_test.columns:
        if not creditcard_test[column].dtype == float:
            creditcard_test[column] = pd.to_numeric(creditcard_test[column], errors='coerce')
        creditcard_test[column].fillna(creditcard_test[column].mean(), inplace=True)
    
    # Load the saved Isolation Forest model
    try:
        loaded_model = CustomIsolationForest(contamination=0.01)  # Use the contamination rate you originally used
        loaded_model.load('custom_isolation_forest_model.pkl')
        st.write("Model loaded successfully.")
    except Exception as e:
        st.write(f"Error loading model: {e}")
        st.stop()

    # Predict anomalies using the loaded model
    try:
        predictions_lof = loaded_model.predict(creditcard_test)
        # Convert predictions to a DataFrame
        predictions_df = pd.DataFrame(predictions_lof, columns=['IsolationForest_Predictions'])
        st.write("Predictions:")
        st.dataframe(predictions_df)
        
        # Provide a button to download the predictions as a CSV file
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='isolation_forest_predictions.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.write(f"Error during prediction: {e}")
