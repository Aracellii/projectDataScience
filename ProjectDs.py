import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("model_churn.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.title("Customer Churn Prediction App")
st.write("Masukkan data berikut untuk memprediksi apakah customer akan churn.")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100, value=40)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
Balance = st.number_input("Balance", min_value=0.0, format="%.2f")
NumOfProducts = st.number_input("Num of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card?", ["No", "Yes"])
IsActiveMember = st.selectbox("Is Active Member?", ["No", "Yes"])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, format="%.2f")

data = {
    "CreditScore": CreditScore,
    "Geography": Geography,
    "Gender": Gender,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary
}

input_df = pd.DataFrame([data])

input_df["HasCrCard"] = input_df["HasCrCard"].map({"No": 0, "Yes": 1})
input_df["IsActiveMember"] = input_df["IsActiveMember"].map({"No": 0, "Yes": 1})
input_df["TenureByAge"] = input_df["Tenure"] / input_df["Age"]
input_df["BalanceSalaryRatio"] = input_df["Balance"] / input_df["EstimatedSalary"].replace(0, 1)
input_df["Age"] 
input_df["Tenure"] 
input_df["NumOfProducts"] 
input_df["IsActiveMember"] 


input_df = pd.get_dummies(input_df)

for col in feature_cols:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[feature_cols]

numerical_cols = list(scaler.feature_names_in_)
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

if st.button("Prediksi"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.error(f"⚠️ Customer kemungkinan CHURN (probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Customer tidak churn (probabilitas: {prob:.2f})")

