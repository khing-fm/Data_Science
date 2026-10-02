import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))

st.title("Telco Customer Churn Predictor")
st.write("Fill in all the customer's details for the most accurate prediction.")

st.subheader("Customer Profile")
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])

st.subheader("Account Info")
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

st.subheader("Services")
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

if st.button("Predict"):
    row = pd.DataFrame(0, index=[0], columns=model_columns)

    row["gender"] = 1 if gender == "Male" else 0
    row["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
    row["Partner"] = 1 if partner == "Yes" else 0
    row["Dependents"] = 1 if dependents == "Yes" else 0
    row["PhoneService"] = 1 if phone_service == "Yes" else 0
    row["PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly_charges
    row["TotalCharges"] = total_charges

    row["IsHighRiskProfile"] = int(
        internet_service == "Fiber optic" and contract == "Month-to-month" and tech_support == "No"
    )
    row["TotalChargesPerTenure"] = total_charges / tenure if tenure > 0 else total_charges

    add_on_services = [online_security, online_backup, device_protection,
                        tech_support, streaming_tv, streaming_movies, multiple_lines]
    row["NumAddOnServices"] = sum(1 for s in add_on_services if s == "Yes")

    onehot_fields = {
        f"MultipleLines_{multiple_lines}": 1,
        f"InternetService_{internet_service}": 1,
        f"OnlineSecurity_{online_security}": 1,
        f"OnlineBackup_{online_backup}": 1,
        f"DeviceProtection_{device_protection}": 1,
        f"TechSupport_{tech_support}": 1,
        f"StreamingTV_{streaming_tv}": 1,
        f"StreamingMovies_{streaming_movies}": 1,
        f"Contract_{contract}": 1,
        f"PaymentMethod_{payment_method}": 1,
    }
    for col_name, value in onehot_fields.items():
        if col_name in row.columns:
            row[col_name] = value

    if tenure <= 12:
        pass
    elif tenure <= 36:
        row["TenureGroup_Mid_13-36m"] = 1
    else:
        row["TenureGroup_Loyal_37-72m"] = 1

    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]
    churn_probability = model.predict_proba(row_scaled)[0][1]
    stay_probability = 1 - churn_probability

    st.subheader("Result")

    stay_pct = stay_probability * 100

    if stay_pct <= 50:
        color = "#b02a2a"   #red
        label = "Low likelihood of staying"
    elif stay_pct <= 80:
        color = "#c97a1c"   #yellow/orange
        label = "Moderate likelihood of staying"
    else:
        color = "#1a7a3c"   #green
        label = "High likelihood of staying"

    st.markdown(
        f"""
        <div style="background-color:{color}; padding:16px; border-radius:8px;">
            <span style="color:white; font-size:18px; font-weight:bold;">
            {label} - likelihood of staying: {stay_pct:.1f}%
            </span>
        """,
        unsafe_allow_html=True
    )