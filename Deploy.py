import streamlit as st
import numpy as np
import pandas as pd
import joblib

def load_model(model_name):
    filename = model_name.replace(' ', '_') + '_best_model.pkl'
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{filename}' not found. Please ensure it exists in the current directory.")

st.title("Dự Đoán Ung Thư Phổi Với Machine Learning")

# Load scaler
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please ensure it exists.")
    st.stop()

# Danh sách đặc trưng (phải khớp với thứ tự trong Model.ipynb)
feature_names = ['AGE', 'SMOKING', 'EXPOSURE_TO_POLLUTION', 'ENERGY_LEVEL', 'BREATHING_ISSUE',
                 'THROAT_DISCOMFORT', 'OXYGEN_SATURATION', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE']

# Nhập 9 đặc trưng (dữ liệu mẫu cho người mắc bệnh)
AGE = st.number_input("AGE", min_value=0, max_value=120, value=30)
SMOKING = st.selectbox("SMOKING (0 = No, 1 = Yes)", [0, 1], index=1)
EXPOSURE_TO_POLLUTION = st.selectbox("EXPOSURE_TO_POLLUTION (0 = No, 1 = Yes)", [0, 1], index=1)
ENERGY_LEVEL = st.slider("ENERGY_LEVEL (23.26-83.05)", 23.26, 83.05, 23.26)
BREATHING_ISSUE = st.selectbox("BREATHING_ISSUE (0 = No, 1 = Yes)", [0, 1], index=1)
THROAT_DISCOMFORT = st.selectbox("THROAT_DISCOMFORT (0 = No, 1 = Yes)", [0, 1], index=1)
OXYGEN_SATURATION = st.number_input("OXYGEN_SATURATION (%)", min_value=89.92, max_value=99.80, value=90.00)
SMOKING_FAMILY_HISTORY = st.selectbox("SMOKING_FAMILY_HISTORY (0 = No, 1 = Yes)", [0, 1], index=1)
STRESS_IMMUNE = st.selectbox("STRESS_IMMUNE (0 = No, 1 = Yes)", [0, 1], index=1)

model_option = st.selectbox("Chọn model dự đoán", [
    "DecisionTree", "Random Forest", "K-Nearest Neighbor", "Bayes", "XGBoost", "Logistic Regression", "SVM"
])

if st.button("Dự đoán"):
    # Tạo DataFrame với tên đặc trưng
    X_input = pd.DataFrame({
        'AGE': [AGE],
        'SMOKING': [SMOKING],
        'EXPOSURE_TO_POLLUTION': [EXPOSURE_TO_POLLUTION],
        'ENERGY_LEVEL': [ENERGY_LEVEL],
        'BREATHING_ISSUE': [BREATHING_ISSUE],
        'THROAT_DISCOMFORT': [THROAT_DISCOMFORT],
        'OXYGEN_SATURATION': [OXYGEN_SATURATION],
        'SMOKING_FAMILY_HISTORY': [SMOKING_FAMILY_HISTORY],
        'STRESS_IMMUNE': [STRESS_IMMUNE]
    }, columns=feature_names)  # Đảm bảo thứ tự cột đúng

    try:
        X_input_scaled = scaler.transform(X_input)
        model = load_model(model_option)
        pred = model.predict(X_input_scaled)
        result = "YES" if pred[0] == 1 else "NO"
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input_scaled)[0, 1]
        st.write(f"Kết quả dự đoán: **{result}**")
        if proba is not None:
            st.write(f"Xác suất dự đoán mắc bệnh phổi: {proba:.4f}")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")