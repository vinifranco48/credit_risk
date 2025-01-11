from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import json
import datetime
import logging
from services.model_services import ModelService
from services.storage_service import StorageService
from src.utils_modelling import compute_credit_scores
from services.database_service import MinioToPostgres


logger = logging.getLogger(__name__)

class PredictionService:
    FEATURE_ORDER = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length',
        'home_ownership', 'annual_inc', 'verification_status', 'purpose',
        'addr_state', 'dti', 'inq_last_6mths', 'mths_since_last_delinq',
        'open_acc', 'revol_bal', 'total_acc', 'initial_list_status',
        'tot_cur_bal', 'mths_since_earliest_cr_line'
    ]


    @staticmethod
    def reorder_features(df: pd.DataFrame) -> pd.DataFrame:
        return df[PredictionService.FEATURE_ORDER]

    @staticmethod
    def process_prediction(prediction: Any) -> Tuple[int, float]:
        if isinstance(prediction, (list, np.ndarray)):
            prediction = prediction[0]
        
        if isinstance(prediction, np.floating):
            prediction = float(prediction)

        probability = float(prediction)
        final_prediction = 1 if probability > 0.5 else 0

        return final_prediction, probability

    @staticmethod
    async def predict(features: Dict [str, Any]) -> Dict[str, Any]:
        try:
            input_df = pd.DataFrame([features])
            input_df = PredictionService.reorder_features(input_df)

            timestamp = datetime.datetime.now().isoformat()
            features_path = f"features/{timestamp}.json"
            await StorageService.upload_json(features_path, features, bucket_name="mlflow")

            pipeline = ModelService.load_preprocessing_pipeline()
            processed_data = pipeline.transform(input_df)

            model = ModelService.load_model()
            raw_prediction = model.predict(processed_data)
            prediction, probability = PredictionService.process_prediction(raw_prediction)


            scorecard = ModelService.load_scorecard()
            credit_score = compute_credit_scores(
                X=processed_data,
                probas=probability,
                scorecard=scorecard
            )

            timestamp = datetime.datetime.now().isoformat()
            minio_path = f"predictions/{timestamp}.json"

            result = {
                "prediction": prediction,
                "probability": probability,
                "credit_score": credit_score,
                "prediction_timestamp": timestamp,
                "minio_storage_path": minio_path
            }

            await StorageService.upload_json(minio_path, result)

            return {
                "prediction": prediction,
                "probability": probability,
                "credit_score": credit_score,
                "prediction_timestamp": timestamp,
                "minio_storage_path": minio_path
            }

            migrator = MinioToPostgres()
            migrator.create_tables()
            migrator.process_mlflow_data()
        
        except Exception as e:
            logger.error(f"Error predicting: {e}")
            raise RuntimeError(f"Error predicting: {e}")
