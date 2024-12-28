import logging
import os
import datetime
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import boto3
import mlflow
import joblib
import uvicorn
from functools import lru_cache
from typing import Optional
import numpy as np
from sklearn.pipeline import Pipeline
from src.utils_modelling import CatCombiner, CatImputer, DiscretizerCombiner, CatOneHotEncoder, compute_credit_scores
import json
from typing import Optional, Dict, Any
import numpy as np
import json

# Configuração de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuração de ambiente
os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9005'

mlflow.set_tracking_uri("http://localhost:5001")

# Configuração MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API para predição de risco de crédito com pipeline completo",
    version="1.0.0"
)

FEATURE_ORDER = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose',
    'addr_state', 'dti', 'inq_last_6mths', 'mths_since_last_delinq',
    'open_acc', 'revol_bal', 'total_acc', 'initial_list_status',
    'tot_cur_bal', 'mths_since_earliest_cr_line'
]

class CreditFeatures(BaseModel):
    loan_amnt: float
    term: int
    int_rate: float
    grade: str
    sub_grade: str
    emp_length: int
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    addr_state: str
    dti: float
    inq_last_6mths: int
    mths_since_last_delinq: Optional[float]
    open_acc: int
    revol_bal: float
    total_acc: float
    initial_list_status: str
    tot_cur_bal: float
    mths_since_earliest_cr_line: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    credit_score: float
    prediction_timestamp: str
    minio_storage_path: str



@lru_cache(maxsize=1)
def load_model() -> mlflow.pyfunc.PyFuncModel:
    run_id = "5711442d888d4768a92c7baa36563db4"
    logger.debug("Tentando carregar o modelo do MLflow...")
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modelo carregado com sucesso.")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise RuntimeError(f"Erro ao carregar modelo: {e}")

@lru_cache(maxsize=1)
def load_preprocessing_pipeline():
    pipeline_path = "pipeline.pkl"
    logger.debug("Tentando carregar o pipeline de preprocessamento...")
    try:
        pipeline = joblib.load(pipeline_path)
        logger.info("Pipeline carregado com sucesso.")
        return pipeline
    except Exception as e:
        logger.error(f"Erro ao carregar pipeline: {e}")
        raise RuntimeError(f"Erro ao carregar pipeline: {e}")

@lru_cache(maxsize=1)
def load_scorecard() -> pd.DataFrame:
    scorecard_path = "scorecard.csv"
    logger.debug("Tentando carregar o scorecard...")
    try:
        scorecard = pd.read_csv(scorecard_path)
        logger.info("Scorecard carregado com sucesso.")
        return scorecard
    except Exception as e:
        logger.error(f"Erro ao carregar scorecard: {e}")
        raise RuntimeError(f"Erro ao carregar scorecard: {e}")

def reorder_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f"Reordenando features conforme: {FEATURE_ORDER}")
    return df[FEATURE_ORDER]

def process_prediction(prediction: Any) -> tuple[int, float]:
    """
    Process model prediction to return standardized prediction and probability.
    """
    if isinstance(prediction, (list, np.ndarray)):
        prediction = prediction[0]
    if isinstance(prediction, np.floating):
        prediction = float(prediction)
        
    probability = float(prediction)
    final_prediction = 1 if probability >= 0.5 else 0
    
    return final_prediction, probability

def save_to_minio(result: Dict[str, Any], minio_path: str) -> None:
    """
    Save results to MinIO storage.
    """
    try:
        # Converter valores numpy para Python nativo
        result_processed = {
            k: float(v) if isinstance(v, np.floating) else v
            for k, v in result.items()
        }
        
        s3_client.put_object(
            Bucket="predictions",
            Key=minio_path,
            Body=json.dumps(result_processed)
        )
        logger.info("Resultado salvo no MinIO.")
    except Exception as e:
        logger.error(f"Erro ao salvar no MinIO: {e}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CreditFeatures):
    try:
        # Convert input to DataFrame
        logger.debug("Converting input to DataFrame...")
        input_df = pd.DataFrame([features.dict()])
        input_df = reorder_features(input_df)

        # Load and apply preprocessing
        logger.debug("Applying preprocessing pipeline...")
        pipeline = load_preprocessing_pipeline()
        processed_data = pipeline.transform(input_df)

        # Make prediction
        logger.debug("Making prediction...")
        model = load_model()
        raw_prediction = model.predict(processed_data)
        prediction, probability = process_prediction(raw_prediction)

        # Calculate credit score
        logger.debug("Calculating credit score...")
        scorecard = load_scorecard()
        credit_score = compute_credit_scores(
            X=processed_data,
            probas=probability,
            scorecard=scorecard
        )

        # Prepare response
        timestamp = datetime.datetime.now().isoformat()
        minio_path = f"predictions/{timestamp}.json"
        
        result = {
            "prediction": prediction,
            "probability": probability,
            "credit_score": credit_score,
            "features": features.dict(),
            "prediction_timestamp": timestamp
        }

        # Save to MinIO
        logger.debug("Saving to MinIO...")
        save_to_minio(result, minio_path)

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            credit_score=credit_score,
            prediction_timestamp=timestamp,
            minio_storage_path=minio_path
        )

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    try:
        load_model()
        load_preprocessing_pipeline()
        load_scorecard()
        s3_client.list_buckets()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)