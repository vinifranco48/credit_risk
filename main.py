from fastapi import FastAPI, HTTPException
from models.schemas import CreditFeatures, PredictionResponse
from services.prediction_service import PredictionService
from services.model_services import ModelService
from services.storage_service import StorageService
from services.database_service import MinioToPostgres
import logging
import sys
from src.utils_modelling import (
    compute_credit_scores, 
    FeaturePreprocessor, 
    CatOneHotEncoder, 
    DiscretizerCombiner, 
    CatCombiner, 
    CatImputer
)
import mlflow

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://mlflow-web:5000")

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for credit risk prediction with complete pipeline",
    version="1.0.0"
)
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CreditFeatures):
    try:
        result = await PredictionService.predict(features.dict())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        ModelService.load_model()
        ModelService.load_preprocessing_pipeline()
        ModelService.load_scorecard()
        StorageService.get_client().list_buckets()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.debug(f"AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID')}")
    logger.debug(f"AWS_SECRET_ACCESS_KEY: {os.environ.get('AWS_SECRET_ACCESS_KEY')}")
    
    