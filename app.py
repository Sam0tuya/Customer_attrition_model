import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# -----------------------
# Load trained model
# -----------------------
BASE_DIR = Path(__file__).parent
# Ensure the models folder exists or adjust path as necessary
model = joblib.load(BASE_DIR / "models" / "churn_model.pkl")

# -----------------------
# Load authentication config
# -----------------------
with open("users.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name="customer_attrition_cookie",
    key="customer_attrition_key",
    cookie_expiry_days=30
)

# -----------------------
# LOGIN & MAIN APP LOGIC
# -----------------------
st.title("Customer Attrition Prediction System")

# Login Widget
authenticator.login(location='main')

# Handle Login Status
if st.session_state['authentication_status']:
    # --- SUCCESSFUL LOGIN ---
    
    # 1. Logout Button in Sidebar
    authenticator.logout(location='sidebar') 
    st.sidebar.write(f'Logged in as *{st.session_state["name"]}*')

    # 2. Main App Content
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.markdown("---")
    st.subheader("Prediction Model")
    st.write("Enter customer details to predict if they are likely to churn (leave):")

    # -----------------------
    # User Inputs
    # -----------------------
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        streaming_tv = st
