from pydantic import BaseModel, Field
from typing import Optional

class CreditFeatures(BaseModel):
    loan_amnt: float = Field(..., description="Loan amount requested")
    term: int = Field(..., description="Loan term in months")
    int_rate: float = Field(..., description="Interest rate")
    grade: str = Field(..., description="Loan grade")
    sub_grade: str = Field(..., description="Loan subgrade")
    emp_length: int = Field(..., description="Employment length in years")
    home_ownership: str = Field(..., description="Home ownership status")
    annual_inc: float = Field(..., description="Annual income")
    verification_status: str = Field(..., description="Verification status")
    purpose: str = Field(..., description="Loan purpose")
    addr_state: str = Field(..., description="State address")
    dti: float = Field(..., description="Debt-to-income ratio")
    inq_last_6mths: int = Field(..., description="Inquiries last 6 months")
    mths_since_last_delinq: Optional[float] = Field(None, description="Months since last delinquency")
    open_acc: int = Field(..., description="Number of open credit accounts")
    revol_bal: float = Field(..., description="Revolving balance")
    total_acc: float = Field(..., description="Total number of credit accounts")
    initial_list_status: str = Field(..., description="Initial list status")
    tot_cur_bal: float = Field(..., description="Total current balance")
    mths_since_earliest_cr_line: float = Field(..., description="Months since earliest credit line")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    credit_score: float = Field(..., description="Calculated credit score")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    minio_storage_path: str = Field(..., description="Path to stored prediction in MinIO")


