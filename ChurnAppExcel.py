import streamlit as st
import pandas as pd
import pickle
import base64  # Import base64 for encoding the file

# Load the XGBoost model
try:
    with open('xgb_model.pickle', 'rb') as file:
        xgb_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def preprocess_data(data):
    # Mapping country names to numeric values
    country_mapping = {'France': 1, 'Spain': 2, 'Germany': 0}
    data['country'] = data['country'].map(country_mapping)
    
    # Mapping gender to numeric values
    gender_mapping = {'Female': 0, 'Male': 1}
    data['gender'] = data['gender'].map(gender_mapping)
    
    return data

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
        color: black;
    }
    .stButton>button {
        background-color: #0073e6;
        color: black;
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
    .prediction-text {
        color: black;
        font-size: 20px;
        font-weight: bold;
    }
    .download-button {
        color: black !important;
        background-color: #0073e6;
        border: none;
        border-radius: 5px;
        padding: 10px;
        text-decoration: none;
    }
    .download-button:hover {
        background-color: #005bb5;
    }
    .filename {
        color: black;
    }
    </style>
    <img src="logo.jpeg" class="logo">
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title("Bank Customer Churn Prediction")

# Option selection
option = st.radio("Choose an option", ("Single Record Input", "Upload Excel/CSV File"))

if option == "Single Record Input":
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
                st.markdown('<p class="prediction-text">Prediction: Customer will not be retained</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="prediction-text">Prediction: Customer will be retained</p>', unsafe_allow_html=True)

elif option == "Upload Excel/CSV File":
    # File upload
    uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        # Preprocess the data (handle categorical variables)
        data = preprocess_data(data)

        # Convert data types
        data = data.astype({
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
        predictions = predict_churn(data)
        
        if predictions is not None:
            # Add predictions to the DataFrame
            data['churn'] = predictions
            data['churn'] = data['churn'].replace({1: 'Yes', 0: 'No'})
            
            # Allow user to download the result as an Excel file
            st.write("## Download Predictions")
            output_file_name = "churn_predictions.xlsx"
            data.to_excel(output_file_name, index=False)
            
            with open(output_file_name, "rb") as file:
                st.markdown(f"""
                    <a href="data:file/xlsx;base64,{base64.b64encode(file.read()).decode()}" 
                       class="download-button" download="{output_file_name}">
                       Download Excel file
                    </a>
                    """, unsafe_allow_html=True)
            
            # Optionally display the data with predictions
            st.write("## Data with Predictions")
            st.dataframe(data)
