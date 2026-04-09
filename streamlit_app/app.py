import streamlit as st
import requests
import os

# 1. Cấu hình trang và tiêu đề (Ghi điểm UX/UI)
st.set_page_config(page_title="Customer Churn Prediction - Group 4", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("""
This application leverages a **Machine Learning model** to predict the probability of customer attrition (churn). 
*Predictions are generated based on real-time behavioral metrics and demographic data.*
""")

# 2. Cấu hình Endpoint API (Dùng biến môi trường để chạy Docker - Phase 3)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

# 3. Giao diện nhập liệu
st.subheader("📝 Customer Information Input")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        usage_frequency = st.number_input("Usage Frequency (per month)", min_value=0, value=10)
        support_calls = st.number_input("Support Calls", min_value=0, value=1)
        
    with col2:
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, value=0)
        subscription_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])
        contract_length = st.selectbox("Contract Length", options=["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=500.0)
        last_interaction = st.number_input("Days Since Last Interaction", min_value=0, value=5)

    submit_button = st.form_submit_button("🚀 Run Prediction")

# 4. Xử lý Logic khi nhấn nút
if submit_button:
    input_data = {
        "age": int(age),
        "gender": str(gender),
        "tenure": int(tenure),
        "usage_frequency": int(usage_frequency),
        "support_calls": int(support_calls),
        "payment_delay": int(payment_delay),
        "subscription_type": str(subscription_type),
        "contract_length": str(contract_length),
        "total_spend": float(total_spend),
        "last_interaction": int(last_interaction)
    }

    try:
        with st.spinner("Connecting to prediction server..."):
            response = requests.post(BACKEND_URL, json=input_data, timeout=10)
            
        if response.status_code == 200:
            res_data = response.json()
            churn_status = res_data["churn"]
            label = res_data["label"]
            prob = res_data.get("churn_probability")

            st.divider()
            st.subheader("🎯 Prediction Results:")
            
            if churn_status:
                st.error(f"🔴 CUSTOMER STATUS: {label.upper()}")
            else:
                st.success(f"🟢 CUSTOMER STATUS: {label.upper()}")
            
            if prob is not None:
                st.info(f"Churn Probability: **{prob*100:.2f}%**")
                st.progress(prob)
        else:
            st.error(f"❌ API Server Error (Status Code: {response.status_code})")

    except requests.exceptions.ConnectionError:
        st.error("❌ Connection Error. Please ensure FastAPI is running at " + BACKEND_URL)
    except Exception as e:
        st.error(f"⚠️ An unknown error occurred: {str(e)}")

# 5. Thông tin nhóm (Footer)
st.sidebar.markdown("---")
st.sidebar.info("📌 **Group 4 - MLOps Project**")