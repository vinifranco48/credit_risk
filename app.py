import mlflow
import logging
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import boto3
import os
import json
import uvicorn
from src.utils_modelling import CatImputer, CatCombiner, DiscretizerCombiner, CatOneHotEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class CreditFeatures(BaseModel):
    loan_amnt: float
    term: int
    int_rate: float
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    addr_state: str
    dti: float 
    inq_last_6mths: int
    mths_since_last_delinq: float
    open_acc: int
    revol_bal: float
    total_acc: float
    initial_list_status: str
    tot_curl_bal: float

s3_client = boto3.client(
    endpoint_url = os.getenv('URL_MINIO'),
    aws_access_key_id = os.getenv('MINIO_ACCESS_KEY'),
    aws_secret_acess_key = os.getenv('MINIO_SECRET')

)


def upload_to_minio(bucket_name, object_name, data): 
    try:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name_with_timestamp = f"{object_name}_{current_time}"

        s3_client.put_object(Bucket=bucket_name, Key=object_name_with_timestamp, Body=data)
        logger.info(f"Dados enviados ao Minio: {object_name_with_timestamp}")
    except Exception as e:
        logger.error(f"Erro ao enviar dados ao MinIO: {str(e)}")


@lru_cache
def get_pipeline():
    return Pipeline(steps=[
        ('discretizer_combiner', DiscretizerCombiner()
         'cat_combiner'
         )
    ])