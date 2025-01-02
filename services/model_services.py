import mlflow
import joblib
import pandas as pd
from functools import lru_cache
import logging
from config.settings import Settings
from src.utils_modelling import (
    compute_credit_scores, 
    FeaturePreprocessor, 
    CatOneHotEncoder, 
    DiscretizerCombiner, 
    CatCombiner, 
    CatImputer
)
logger = logging.getLogger(__name__)
settings = Settings()


class ModelService:
    @staticmethod
    @lru_cache(maxsize=1)
    def load_model() -> mlflow.pyfunc.PyFuncModel:

        logger.debug("Loading MLflow model...")
        try:
            model_uri = f"runs:/{settings.MODEL_RUN_ID}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Error loading model: {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_preprocessing_pipeline():
        logger.debug("Loading preprocessing pipeline...")

        try:
            pipeline = joblib.load(settings.PIPELINE_PATH)
            logger.info("Pipeline loaded successfully.")
            return pipeline
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise RuntimeError(f"Error loading pipeline: {e}")
    @staticmethod
    @lru_cache(maxsize=1)
    def load_scorecard() -> pd.DataFrame:
        logger.debug("Loading scorecard...")

        try:
            scorecard = pd.read_csv(settings.SCORECARD_PATH)
            logger.info("Scorecard loaded successfully.")
            return scorecard
        except Exception as e:
            logger.error(f"Error loading scorecard: {e}")
            raise RuntimeError(f"Error loading scorecard: {e}")
