import streamlit as st
import numpy as np
import pandas as pd
import joblib

# === CSS để làm giao diện đẹp hơn ===
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            padding: 20px;
        }
        h1 {
            color: #1f77b4;
            text-align: center;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Dự Đoán Ung Thư Phổi Bằng Machine Learning")

# === Load scaler ===
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'scaler.pkl'. Vui lòng kiểm tra lại.")
    st.stop()

# === Load model ===
def load_model(model_name):
    filename = model_name.replace(' ', '_') + '_best_model.pkl'
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Không tìm thấy file model: '{filename}'.")

# === Danh sách đặc trưng ===
feature_names = ['AGE', 'SMOKING', 'EXPOSURE_TO_POLLUTION', 'ENERGY_LEVEL', 'BREATHING_ISSUE',
                 'THROAT_DISCOMFORT', 'OXYGEN_SATURATION', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE']

# === Nhập liệu đầu vào (chia layout) ===
st.markdown("### 📋 Nhập thông tin người dùng")

col1, col2, col3 = st.columns(3)

with col1:
    AGE = st.number_input("Tuổi", min_value=0, max_value=120, value=30)
    SMOKING = st.radio("Hút thuốc?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")

with col2:
    EXPOSURE_TO_POLLUTION = st.radio("Ô nhiễm môi trường?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
    BREATHING_ISSUE = st.radio("Khó thở?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
    THROAT_DISCOMFORT = st.radio("Đau họng?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")

with col3:
    ENERGY_LEVEL = st.slider("Mức năng lượng", 23.26, 83.05, 40.0)
    OXYGEN_SATURATION = st.number_input("Độ bão hòa Oxy (%)", min_value=89.92, max_value=99.80, value=95.0)
    SMOKING_FAMILY_HISTORY = st.radio("Gia đình có người hút thuốc?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
    STRESS_IMMUNE = st.radio("Miễn dịch yếu do căng thẳng?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")

# === Chọn mô hình ===
model_option = st.selectbox("🤖 Chọn mô hình dự đoán", [
    "DecisionTree", "Random Forest", "K-Nearest Neighbor", "Bayes", "XGBoost", "Logistic Regression", "SVM"
])

# === Dự đoán ===
if st.button("🔍 Dự đoán"):
    X_input = pd.DataFrame([[
        AGE, SMOKING, EXPOSURE_TO_POLLUTION, ENERGY_LEVEL, BREATHING_ISSUE,
        THROAT_DISCOMFORT, OXYGEN_SATURATION, SMOKING_FAMILY_HISTORY, STRESS_IMMUNE
    ]], columns=feature_names)

    try:
        X_scaled = scaler.transform(X_input)
        model = load_model(model_option)
        pred = model.predict(X_scaled)
        result = "🟢 Không mắc" if pred[0] == 0 else "🔴 Có khả năng mắc"
        st.success(f"**Kết quả:** {result}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0, 1]
            st.info(f"**Xác suất mắc bệnh:** {proba:.2%}")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
