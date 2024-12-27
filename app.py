import mlflow
import logging
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import boto3
import os
import json
import uvicorn
from functools import lru_cache
import time
from typing import Optional
import joblib
from sklearn.pipeline import Pipeline
import numpy as np
from src.utils_modelling import CatImputer, CatCombiner, DiscretizerCombiner,CatOneHotEncoder

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração MLflow
os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9005'
mlflow.set_tracking_uri("http://localhost:5001")

# Configuração MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9005'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'mlflow'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'password')
)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API para predição de risco de crédito com pipeline completo de preprocessamento",
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
    loan_amnt: float = Field(..., description="Valor do empréstimo")
    term: int = Field(..., description="Prazo em meses (36 ou 60)")
    int_rate: float = Field(..., description="Taxa de juros")
    grade: str = Field(..., description="Grade")
    sub_grade: str = Field(..., description="Subgrade")
    emp_length: int = Field(..., description="Tempo de emprego em anos")
    home_ownership: str = Field(..., description="Tipo de propriedade")
    annual_inc: float = Field(..., description="Renda anual")
    verification_status: str = Field(..., description="Status de verificação")
    purpose: str = Field(..., description="Finalidade do empréstimo")
    addr_state: str = Field(..., description="Estado")
    dti: float = Field(..., description="Debt-to-Income ratio")
    inq_last_6mths: int = Field(..., description="Consultas nos últimos 6 meses")
    mths_since_last_delinq: Optional[float] = Field(None, description="Meses desde último atraso")
    open_acc: int = Field(..., description="Número de contas abertas")
    revol_bal: float = Field(..., description="Saldo rotativo")
    total_acc: float = Field(..., description="Total de contas")
    initial_list_status: str = Field(..., description="Status inicial da lista")
    tot_cur_bal: float = Field(..., description="Saldo atual total")
    mths_since_earliest_cr_line: float = Field(..., description="Meses desde a primeira linha de crédito")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    prediction_timestamp: str
    minio_storage_path: str

class ModelLoader:
    def __init__(self, run_id: str = "95e7b8ca498b4398bcd094187e578858", max_retries: int = 5, retry_delay: int = 10):
        self.run_id = run_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._model: Optional[mlflow.pyfunc.PyFuncModel] = None

    def load(self) -> mlflow.pyfunc.PyFuncModel:
        """
        Carrega o modelo do MLflow com retry em caso de falha
        """
        if self._model is not None:
            return self._model

        retries = 0
        last_exception = None

        while retries < self.max_retries:
            try:
                logger.info(f"Tentativa {retries + 1} de {self.max_retries} para carregar o modelo...")
                model_uri = f'runs:/{self.run_id}/model'
                self._model = mlflow.pyfunc.load_model(model_uri)
                logger.info("Modelo carregado com sucesso!")
                return self._model

            except Exception as e:
                last_exception = e
                logger.warning(f"Tentativa {retries + 1} falhou. Erro: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    logger.info(f"Aguardando {self.retry_delay} segundos antes da próxima tentativa...")
                    time.sleep(self.retry_delay)

        logger.error(f"Falha ao carregar o modelo após {self.max_retries} tentativas")
        raise RuntimeError(f"Falha ao carregar o modelo: {str(last_exception)}")

@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    """
    Retorna uma instância única do ModelLoader
    """
    return ModelLoader()

@lru_cache(maxsize=1)
def load_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Carrega o modelo do MLflow usando o ModelLoader
    """
    loader = get_model_loader()
    return loader.load()

@lru_cache(maxsize=1)
def load_preprocessing_pipeline():
    """
    Carrega a pipeline de preprocessamento salva
    """
    try:
        pipeline_path = 'pipeline.pkl'
        pipeline = joblib.load(pipeline_path)
        logger.info("Pipeline de preprocessamento carregada com sucesso")
        return pipeline
    except Exception as e:
        logger.error(f"Erro ao carregar pipeline: {str(e)}")
        raise RuntimeError(f"Falha ao carregar pipeline: {str(e)}")

def upload_to_minio(bucket_name: str, object_name: str, data: dict):
    """Upload dos dados para o MinIO"""
    try:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name_with_timestamp = f"{object_name}_{current_time}.json"

        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_name_with_timestamp,
            Body=json.dumps(data)
        )

        logger.info(f"Dados enviados ao MinIO: {object_name_with_timestamp}")
        return object_name_with_timestamp
    except Exception as e:
        logger.error(f"Erro ao enviar dados ao MinIO: {str(e)}")
        raise

def reorder_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena as features do DataFrame conforme a ordem especificada"""
    return df[FEATURE_ORDER]

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CreditFeatures):
    try:
        # Converter input para DataFrame
        input_df = pd.DataFrame([features.model_dump()])
        
        # Reordenar features
        input_df = reorder_features(input_df)
        
        logger.info(f"Input features order: {input_df.columns.tolist()}")
        
        # Carregar e aplicar a pipeline de preprocessamento
        pipeline = load_preprocessing_pipeline()
        processed_data = pipeline.transform(input_df)
        
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Carregar modelo e fazer predição
        model = load_model()
        
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(processed_data)
            probability = float(prediction_proba[0][1])
            prediction = int(probability >= 0.5)
        else:
            prediction = model.predict(processed_data)
            probability = float(prediction[0])

        # Preparar resultado
        current_time = datetime.datetime.now().isoformat()
        result = {
            "input_data": features.model_dump(),
            "prediction": prediction,
            "probability": probability,
            "timestamp": current_time
        }

        # Upload para MinIO
        minio_path = upload_to_minio(
            bucket_name="predictions",
            object_name="credit_prediction",
            data=result
        )

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            prediction_timestamp=current_time,
            minio_storage_path=minio_path
        )

    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        model = load_model()
        pipeline = load_preprocessing_pipeline()
        s3_client.list_buckets()

        return {
            "status": "healthy",
            "mlflow_model_loaded": True,
            "preprocessing_pipeline_loaded": True,
            "minio_connected": True,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erro durante health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)