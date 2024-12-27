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
from src.utils_modelling import CatCombiner, CatImputer, DiscretizerCombiner, CatOneHotEncoder
import json

# Configuração de logging
logging.basicConfig(level=logging.DEBUG)  # Configura log detalhado
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

# Classes
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
    prediction_timestamp: str
    minio_storage_path: str


# Carregamento de modelos e pipeline
@lru_cache(maxsize=1)
def load_model() -> mlflow.pyfunc.PyFuncModel:
    run_id = "7c0af99bcb044023a853c53483033e07"
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


def reorder_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f"Reordenando features conforme: {FEATURE_ORDER}")
    return df[FEATURE_ORDER]


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CreditFeatures):
    try:
        # Input para DataFrame
        logger.debug("Convertendo entrada para DataFrame...")
        # Usando model_dump ao invés de dict (correção da depreciação)
        input_df = pd.DataFrame([features.model_dump()])

        # Reordenando
        input_df = reorder_features(input_df)
        logger.debug(f"Features reordenadas: {input_df.columns.tolist()}")

        # Pipeline
        logger.debug("Carregando e aplicando pipeline...")
        pipeline = load_preprocessing_pipeline()
        processed_data = pipeline.transform(input_df)
        logger.debug(f"Dados processados: {processed_data.shape}")
        logger.debug(f"Processed data:\n{processed_data}")

        # Modelo
        logger.debug("Carregando modelo para predição...")
        model = load_model()

        # Predição
        logger.debug("Executando predição...")
        prediction = model.predict(processed_data)
        
        # Convertendo tipos para serialização
        final_prediction = int(prediction[0])  # Convertendo np.int64 para int
        probability = float(prediction[0])     # Convertendo para float
        
        # MinIO
        logger.debug("Preparando upload para MinIO...")
        current_time = datetime.datetime.now().isoformat()
        
        # Garantindo que todos os dados sejam serializáveis
        result = {
            "input_data": features.model_dump(),  # Usando model_dump
            "prediction": final_prediction,        # Já convertido para int
            "probability": probability,            # Já convertido para float
            "timestamp": current_time,
        }
        
        # Convertendo qualquer número numpy para Python nativo
        result = {k: v.item() if hasattr(v, 'item') else v 
                 for k, v in result.items()}
        
        minio_path = f"predictions/credit_prediction_{current_time}.json"
        
        s3_client.put_object(
            Bucket="predictions",
            Key=minio_path,
            Body=json.dumps(result)  # Agora todos os dados são serializáveis
        )
        logger.info("Resultado salvo no MinIO.")

        return PredictionResponse(
            prediction=final_prediction,
            probability=probability,
            prediction_timestamp=current_time,
            minio_storage_path=minio_path
        )

    except Exception as e:
        logger.error(f"Erro durante predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        logger.debug("Verificando saúde da aplicação...")
        load_model()
        load_preprocessing_pipeline()
        s3_client.list_buckets()
        logger.info("Health check passou.")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {e}")


# Inicialização
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
