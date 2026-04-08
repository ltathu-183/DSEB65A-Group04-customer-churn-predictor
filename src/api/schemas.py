"""Pydantic schemas for churn prediction API (raw customer row, aligned with project CSV)."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """One customer record (no CustomerID / Churn), matching training data types."""

    age: int = Field(..., ge=0, le=120, description="Customer age")
    gender: str = Field(..., description="Gender, e.g. Male, Female")
    tenure: int = Field(..., ge=0, description="Tenure in months")
    usage_frequency: int = Field(..., ge=0, description="Usage frequency")
    support_calls: int = Field(..., ge=0, description="Number of support calls")
    payment_delay: int = Field(..., ge=0, description="Payment delay in days")
    subscription_type: str = Field(
        ...,
        description="Subscription tier, e.g. Basic, Standard, Premium",
    )
    contract_length: str = Field(
        ...,
        description="Contract type, e.g. Monthly, Quarterly, Annual",
    )
    total_spend: float = Field(..., ge=0, description="Total spend")
    last_interaction: int = Field(
        ...,
        ge=0,
        description="Days since last interaction (Last Interaction)",
    )


class ChurnPredictionResponse(BaseModel):
    """Churn prediction result."""

    churn: bool = Field(
        ...,
        description="True if the model predicts churn (positive class)",
    )
    label: str = Field(..., description="Human-readable label: Churn or No Churn")
    churn_probability: float | None = Field(
        None,
        description="Probability of the churn class if the model supports predict_proba",
    )