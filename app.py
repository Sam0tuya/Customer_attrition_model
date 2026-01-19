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
# APP HEADER
# -----------------------
st.title("Customer Attrition Prediction System")

# -----------------------
# LOGIN WIDGET
# -----------------------
# This renders the login form on the main page
authenticator.login(location='main')

# -----------------------
# MAIN LOGIC
# -----------------------

# CASE 1: User is Logged In
if st.session_state['authentication_status']:
    
    # 1. Sidebar Logout & Info
    authenticator.logout(location='sidebar') 
    st.sidebar.write(f'Logged in as *{st.session_state["name"]}*')

    # 2. Main Application Content
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.markdown("---")
    st.subheader("Prediction Model")
    st.write("Enter customer details to predict if they are likely to churn (leave):")

    # --- INPUT FORM ---
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
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0)

    # --- PREPARE DATA ---
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_dict])

    # --- ENCODING ---
    cat_cols = input_df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()

        if col == "gender":
            le.classes_ = np.array(["Female", "Male"])
        elif col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            le.classes_ = np.array(["No", "Yes"])
        elif col == "MultipleLines":
            le.classes_ = np.array(["No", "No phone service", "Yes"])
        elif col == "InternetService":
            le.classes_ = np.array(["DSL", "Fiber optic", "No"])
        elif col in [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]:
            le.classes_ = np.array(["No", "No internet service", "Yes"])
        elif col == "Contract":
            le.classes_ = np.array(["Month-to-month", "One year", "Two year"])
        elif col == "PaymentMethod":
            le.classes_ = np.array([
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ])

        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError as e:
            st.error(f"Encoding Error in column {col}: {e}")

    # --- PREDICTION ---
    st.markdown("---")
    if st.button("Predict Churn"):
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("Customer is likely to churn ❌")
        else:
            st.success("Customer is likely to stay ✅")

# CASE 2: Login Failed
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')

# CASE 3: No Login Attempt Yet (Show Registration Option)
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')

    # -----------------------
    # REGISTRATION SECTION
    # -----------------------
    st.markdown("---")
    with st.expander("Or Register a New Account"):
        try:
            # FIX: explicitly naming 'form_name' and 'location' fixes the error
            email, username, name = authenticator.register_user(
                form_name='Register User', 
                location='main', 
                pre_authorized=[]
            )
            
            if email:
                # 1. WRITE THE NEW USER TO THE YAML FILE
                with open('users.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
                    
                st.success('User registered successfully! Please log in above.')
        except Exception as e:
            st.error(e)
