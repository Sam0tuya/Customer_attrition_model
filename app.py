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
# LOGIN
# -----------------------
st.title("Customer Attrition Prediction System")

# login returns True / False / None
login_status = authenticator.login("Login", location="main")

# -----------------------
# REGISTRATION (Optional)
# -----------------------
try:
    if authenticator.register_user("Register", preauthorization=False):
        st.success("User registered successfully! Please log in.")
except Exception as e:
    st.error(e)

# -----------------------
# ACCESS CONTROL
# -----------------------
if login_status:

    # Logout button
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Logged in as: {authenticator.credentials['admin']['name']}")

    st.title("Customer Attrition Prediction Model")
    st.write("Enter customer details to predict if they are likely to churn (leave):")

    # -----------------------
    # User Inputs
    # -----------------------
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

    # -----------------------
    # Prepare input
    # -----------------------
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

    # -----------------------
    # Encode categorical columns
    # -----------------------
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

        input_df[col] = le.transform(input_df[col])

    # -----------------------
    # Prediction
    # -----------------------
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("Customer is likely to churn ❌")
        else:
            st.success("Customer is likely to stay ✅")

elif login_status is False:
    st.error("Username or password is incorrect")

elif login_status is None:
    st.warning("Please log in to continue")
