"""Pydantic schemas for churn prediction API (raw customer row, aligned with project CSV)."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Một dòng khách hàng (không gồm CustomerID / Churn), đúng kiểu dữ liệu huấn luyện."""

    age: int = Field(..., ge=0, le=120, description="Tuổi")
    gender: str = Field(..., description="Giới tính, ví dụ: Male, Female")
    tenure: int = Field(..., ge=0, description="Thời gian gắn bó (tháng)")
    usage_frequency: int = Field(..., ge=0, description="Tần suất sử dụng")
    support_calls: int = Field(..., ge=0, description="Số cuộc gọi hỗ trợ")
    payment_delay: int = Field(..., ge=0, description="Độ trễ thanh toán (ngày)")
    subscription_type: str = Field(
        ...,
        description="Gói dịch vụ, ví dụ: Basic, Standard, Premium",
    )
    contract_length: str = Field(
        ...,
        description="Loại hợp đồng, ví dụ: Monthly, Quarterly, Annual",
    )
    total_spend: float = Field(..., ge=0, description="Tổng chi tiêu")
    last_interaction: int = Field(
        ...,
        ge=0,
        description="Ngày kể từ lần tương tác cuối (Last Interaction)",
    )


class ChurnPredictionResponse(BaseModel):
    """Kết quả dự đoán churn."""

    churn: bool = Field(..., description="True nếu mô hình dự đoán churn (lớp dương)")
    label: str = Field(..., description="Nhãn đọc được: Churn hoặc No Churn")
    churn_probability: float | None = Field(
        None,
        description="Xác suất lớp churn nếu mô hình có predict_proba",
    )
