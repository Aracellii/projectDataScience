# ===============================================================
# Front.py — FRONTEND STREAMLIT
# ===============================================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ============================================================
# LOAD MODEL, SCALER, DAN FEATURE LIST
# ============================================================
model = joblib.load("model_churn.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.title("🔍 Prediksi Customer Churn Bank")
st.write("Masukkan data nasabah untuk memprediksi apakah nasabah berpotensi churn.")

# ============================
# FORM INPUT PENGGUNA
# ============================

# Kolom input
col1, col2 = st.columns(2)

with col1:
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    Age = st.number_input("Usia", min_value=18, max_value=100, value=40)
    Tenure = st.number_input("Tenure (Tahun)", min_value=0, max_value=10, value=3)
    Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
    NumOfProducts = st.selectbox("Jumlah Produk", [1, 2, 3, 4])

with col2:
    EstimatedSalary = st.number_input("Perkiraan Gaji", min_value=0.0, value=50000.0)
    HasCrCard = st.selectbox("Punya Credit Card?", [0, 1])
    IsActiveMember = st.selectbox("Status Keaktifan", [0, 1])
    Geography = st.selectbox("Lokasi", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    CardType = st.selectbox("Jenis Kartu", ["Blue", "Silver", "Gold", "Platinum"])
    PointEarned = st.number_input("Point Earned", min_value=0, value=500)

# ============================
# FEATURE ENGINEERING
# ============================
BalanceSalaryRatio = Balance / EstimatedSalary if EstimatedSalary > 0 else 0
TenureByAge = Tenure / Age if Age > 0 else 0

# ============================
# BUTTON PREDIKSI
# ============================
if st.button("🔮 Prediksi Churn"):
    
    # Buat dataframe 1 baris
    input_dict = {
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "EstimatedSalary": EstimatedSalary,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "Point Earned": PointEarned,
        "BalanceSalaryRatio": BalanceSalaryRatio,
        "TenureByAge": TenureByAge,
        # Encoding categories
        "Geography_Germany": 1 if Geography == "Germany" else 0,
        "Geography_Spain": 1 if Geography == "Spain" else 0,
        "Gender_Male": 1 if Gender == "Male" else 0,
        "Card Type_Gold": 1 if CardType == "Gold" else 0,
        "Card Type_Platinum": 1 if CardType == "Platinum" else 0,
        "Card Type_Silver": 1 if CardType == "Silver" else 0,
    }

    # Buat DataFrame sesuai urutan fitur model
    input_df = pd.DataFrame([input_dict], columns=feature_cols)

    # Scaling hanya fitur numerik yg butuh
    numerical_cols = [
        "CreditScore","Age","Tenure","Balance",
        "NumOfProducts","EstimatedSalary",
        "BalanceSalaryRatio","TenureByAge","Point Earned"
    ]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Prediksi
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # ============================
    # OUTPUT
    st.subheader("📌 Hasil Prediksi")
    if pred == 1:
        st.error(f"⚠️ Nasabah **berpotensi CHURN** (Probabilitas: {prob:.2f})")
    else:
        st.success(f"💚 Nasabah **TIDAK berpotensi churn** (Probabilitas: {prob:.2f})")

