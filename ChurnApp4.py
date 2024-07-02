#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import pickle

# Load the XGBoost model
try:
    with open('xgb_model.pickle', 'rb') as file:
        xgb_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def predict_churn(data):
    # Ensure the order and names of columns match the model's expectations
    expected_features = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance',
                         'products_number', 'credit_card', 'active_member', 'estimated_salary']
    
    try:
        # Make predictions
        predictions = xgb_model.predict(data[expected_features])
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Custom CSS for background color and logo
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f4f4;
        color: white;
    }
    .stButton>button {
        background-color: #0073e6;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
    .logo {
        position: fixed;
        top: 10px;
        left: 10px;
        width: 150px;
    }
    </style>
    <img src="nicelogo.jpeg" class="logo">
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title("Customer Churn Prediction")

# Input fields
st.write("## Enter Customer Data")
credit_score = st.number_input("Credit Score", value=700)
country = st.selectbox("Country", options=[0, 1, 2], format_func=lambda x: ["France", "Spain", "Germany"][x])
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: ["Male", "Female"][x])
age = st.number_input("Age", value=42)
tenure = st.number_input("Tenure", value=5)
balance = st.number_input("Balance", value=10000.0)
products_number = st.number_input("Number of Products", value=2)
credit_card = st.selectbox("Has Credit Card?", options=[0, 1])
active_member = st.selectbox("Active Member?", options=[0, 1])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# Create a DataFrame from user input
user_data = pd.DataFrame({
    'credit_score': [credit_score],
    'country': [country],
    'gender': [gender],
    'age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'products_number': [products_number],
    'credit_card': [credit_card],
    'active_member': [active_member],
    'estimated_salary': [estimated_salary]
})

# Convert data types
user_data = user_data.astype({
    'credit_score': 'int64',
    'country': 'int64',
    'gender': 'int64',
    'age': 'int64',
    'tenure': 'int64',
    'balance': 'float64',
    'products_number': 'int64',
    'credit_card': 'int64',
    'active_member': 'int64',
    'estimated_salary': 'float64'
})

# Predict churn
if st.button("Predict Churn"):
    prediction = predict_churn(user_data)
    if prediction is not None:
        if prediction[0] == 1:
            st.write("Prediction: Customer will not be
