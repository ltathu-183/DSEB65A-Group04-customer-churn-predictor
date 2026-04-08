import streamlit as st
import requests
import os

# 1. Cấu hình trang và tiêu đề (Ghi điểm UX/UI)
st.set_page_config(page_title="Dự báo Khách hàng Rời bỏ - Nhóm 4", layout="wide")

st.title("📊 Hệ thống Dự báo Khách hàng Rời bỏ (Customer Churn)")
st.markdown("""
Ứng dụng này sử dụng mô hình Machine Learning để dự đoán khả năng một khách hàng sẽ ngừng sử dụng dịch vụ. 
*Kết quả dựa trên dữ liệu hành vi và thông tin cá nhân của khách hàng.*
""")

# 2. Cấu hình Endpoint API (Tư duy MLOps: Dùng biến môi trường để sau này chạy Docker)
# Khi chạy local sẽ dùng localhost, khi Ngọc chạy Docker sẽ đổi qua tên service
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

# 3. Giao diện nhập liệu (Dựa chính xác 100% vào schemas.py của Ly)
st.subheader("📝 Nhập thông tin khách hàng")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Tuổi", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Giới tính", options=["Male", "Female"])
        tenure = st.number_input("Thời gian gắn bó (tháng)", min_value=0, value=12)
        usage_frequency = st.number_input("Tần suất sử dụng (lần/tháng)", min_value=0, value=10)
        support_calls = st.number_input("Số cuộc gọi hỗ trợ", min_value=0, value=1)
        
    with col2:
        payment_delay = st.number_input("Độ trễ thanh toán (ngày)", min_value=0, value=0)
        subscription_type = st.selectbox("Gói dịch vụ", options=["Basic", "Standard", "Premium"])
        contract_length = st.selectbox("Loại hợp đồng", options=["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Tổng chi tiêu ($)", min_value=0.0, value=500.0)
        last_interaction = st.number_input("Số ngày kể từ lần tương tác cuối", min_value=0, value=5)

    submit_button = st.form_submit_button("🚀 Thực hiện Dự đoán")

# 4. Xử lý Logic khi nhấn nút
if submit_button:
    # Chuẩn bị dữ liệu JSON đúng cấu trúc CustomerFeatures của Ly
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
        with st.spinner("Đang kết nối tới máy chủ dự báo..."):
            # Gửi POST request tới API của Ly
            response = requests.post(BACKEND_URL, json=input_data, timeout=10)
            
        if response.status_code == 200:
            res_data = response.json()
            churn_status = res_data["churn"]
            label = res_data["label"]
            prob = res_data.get("churn_probability")

            st.divider()
            st.subheader("🎯 Kết quả dự báo:")
            
            # Hiển thị trực quan theo kết quả (Ghi điểm UX)
            if churn_status:
                st.error(f"🔴 KHÁCH HÀNG CÓ KHẢ NĂNG: {label.upper()}")
            else:
                st.success(f"🟢 KHÁCH HÀNG CÓ KHẢ NĂNG: {label.upper()}")
            
            if prob is not None:
                st.info(f"Xác suất rời bỏ: **{prob*100:.2f}%**")
                st.progress(prob)
                
        else:
            st.error(f"❌ Lỗi từ máy chủ API (Mã lỗi: {response.status_code})")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error("❌ Không thể kết nối tới Backend. Hãy đảm bảo FastAPI đang chạy tại " + BACKEND_URL)
    except Exception as e:
        st.error(f"⚠️ Đã xảy ra lỗi không xác định: {str(e)}")

# 5. Thông tin nhóm (Footer)
st.sidebar.markdown("---")
st.sidebar.info("📌 **Nhóm 4 - MLOps Project**")
