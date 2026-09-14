import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model/loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature columns
with open("model/feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# Page title
st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details to predict loan approval.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income", min_value=0, value=50000
)

coapplicant_income = st.number_input(
    "Coapplicant Income", min_value=0, value=0
)

loan_amount = st.number_input(
    "Loan Amount", min_value=0, value=200000
)

loan_term = st.selectbox(
    "Loan Amount Term", [120, 180, 240, 300, 360]
)

credit_history = st.selectbox(
    "Credit History", [1.0, 0.0]
)

property_area = st.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

# Prediction button
if st.button("Predict Loan Approval"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    # Convert categorical data into numbers
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Make sure input has same columns as training data
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
