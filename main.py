# -*- coding: utf-8 -*-
"""
Customer Churn Prediction Web App using ANN + Preprocessor
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

@st.cache_resource
def load_ann():
    return keras.models.load_model("churn_ann_model.keras")

ann = load_ann()
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

best_threshold = 0.55

def churn_prediction(input_dict):
    df_input = pd.DataFrame([input_dict])  

    input_processed = preprocessor.transform(df_input)

    prob = ann.predict(input_processed)[0][0]

    if prob > best_threshold:
        return f" Likely to Churn"
    else:
        return f"Likely to Stay"

st.set_page_config(layout="wide")
st.title("📊 Customer Churn Prediction")

col1, space, col2 = st.columns([1.5, 0.2, 1.5])

with col1:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=None)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], index=None)

with col2:
    PaymentMethod = st.selectbox("Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"], index=None)
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=None)
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=None)
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=None)

diagnosis = ""

if st.button("Predict Churn"):
    dropdowns = [Contract, PaperlessBilling, PaymentMethod, InternetService, OnlineSecurity, TechSupport]

    if any(option is None for option in dropdowns):
        st.error("Please fill all fields before prediction.")
    else:
        try:
            input_dict = {
                "gender": "Female",          
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": "No",   
                "DeviceProtection": "No",   
                "TechSupport": TechSupport,
                "StreamingTV": "No",          
                "StreamingMovies": "No",    
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges
            }

            diagnosis = churn_prediction(input_dict)
            st.success(diagnosis)

        except Exception as e:
            st.error(f"Error in prediction: {e}")





