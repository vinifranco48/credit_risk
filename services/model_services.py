import mlflow
import joblib
import pandas as pd
import numpy as np
from functools import lru_cache
import logging
from config.settings import Settings
from typing import Dict, Any, Tuple
from src.preprocessing_ead_lgd import create_ead_lgd_pipeline

logger = logging.getLogger(__name__)  # noqa: F821
settings = Settings()
state_to_region = {
    'ME': 'Northeast', 'NH': 'Northeast', 'VT': 'Northeast', 'MA': 'Northeast', 
    'RI': 'Northeast', 'CT': 'Northeast', 'NY': 'Northeast', 'NJ': 'Northeast', 
    'PA': 'Northeast', 'DC': 'Northeast',
    'OH': 'Midwest', 'MI': 'Midwest', 'IN': 'Midwest', 'IL': 'Midwest', 
    'WI': 'Midwest', 'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest', 
    'ND': 'Midwest', 'SD': 'Midwest', 'NE': 'Midwest', 'KS': 'Midwest',
    'DE': 'South', 'MD': 'South', 'VA': 'South', 'WV': 'South', 'KY': 'South',
    'NC': 'South', 'SC': 'South', 'TN': 'South', 'GA': 'South', 'FL': 'South',
    'AL': 'South', 'MS': 'South', 'AR': 'South', 'LA': 'South', 'TX': 'South',
    'OK': 'South',
    'MT': 'West', 'ID': 'West', 'WY': 'West', 'CO': 'West', 'NM': 'West',
    'AZ': 'West', 'UT': 'West', 'NV': 'West', 'WA': 'West', 'OR': 'West',
    'CA': 'West', 'AK': 'West', 'HI': 'West'
}
class ModelService:
    @staticmethod
    @lru_cache(maxsize=1)
    def load_model() -> mlflow.pyfunc.PyFuncModel:
        """Load the legacy model (PD model)"""
        logger.debug("Loading legacy model...")
        try:
            model = f"runs:/{settings.MODEL_PD}/model"
            return mlflow.pyfunc.load_model(model)
        except Exception as e:
            logger.error(f"Error loading legacy model: {e}")
            raise RuntimeError(f"Error loading legacy model: {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_pd_model() -> mlflow.pyfunc.PyFuncModel:
        logger.debug("Loading PD model...")
        try:
            model = f"runs:/{settings.MODEL_PD}/model"
            return mlflow.pyfunc.load_model(model)
        except Exception as e:
            logger.error(f"Error loading PD model: {e}")
            raise RuntimeError(f"Error loading PD model: {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_lgd_model() -> mlflow.pyfunc.PyFuncModel:
        logger.debug("Loading LGD model...")
        try:
            model = f"runs:/{settings.MODEL_LGD_LINEAR}/model"
            return mlflow.pyfunc.load_model(model)
        except Exception as e:
            logger.error(f"Error loading LGD model: {e}")
            raise RuntimeError(f"Error loading LGD model: {e}")
    
    @staticmethod
    @lru_cache(maxsize=1)
    def load_lgd_logistic_model() -> mlflow.pyfunc.PyFuncModel:
        logger.debug("Loading LGD model...")
        try:
            model = f"runs:/{settings.MODEL_LGD_LOGISTIC}/model"
            return mlflow.pyfunc.load_model(model)
        except Exception as e:
            logger.error(f"Error loading LGD model: {e}")
            raise RuntimeError(f"Error loading LGD model: {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_ead_model() -> mlflow.pyfunc.PyFuncModel:
        logger.debug("Loading EAD model...")
        try:
            model = f"runs:/{settings.MODEL_EAD}/model"
            return mlflow.pyfunc.load_model(model)
        except Exception as e:
            logger.error(f"Error loading EAD model: {e}")
            raise RuntimeError(f"Error loading EAD model: {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_preprocessing_ead_lgd():
        """Load the preprocessing pipeline specific for EAD and LGD models"""
        logger.debug("Loading EAD/LGD preprocessing pipeline...")
        try:
            pipeline = joblib.load(settings.PIPELINE_EAD_LGD)
            return pipeline
        except Exception as e:
            logger.error(f"Error loading EAD/LGD preprocessing pipeline: {e}")
            raise RuntimeError(f"Error loading EAD/LGD preprocessing pipeline: {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_preprocessing_pipeline():
        """Load the main preprocessing pipeline"""
        logger.debug("Loading preprocessing pipeline...")
        try:
            pipeline = joblib.load(settings.PIPELINE_PATH)
            return pipeline
        except Exception as e:
            logger.error(f"Error loading preprocessing pipeline: {e}")
            raise RuntimeError(f"Error loading preprocessing pipeline: {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_scorecard():
        """Load the scorecard for credit score calculation"""
        logger.debug("Loading scorecard...")
        try:
            scorecard = pd.read_csv(settings.SCORECARD_PATH)
            return scorecard
        except Exception as e:
            logger.error(f"Error loading scorecard: {e}")
            raise RuntimeError(f"Error loading scorecard: {e}")

    @staticmethod
    def preprocess_data_for_ead_lgd(data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data specifically for EAD and LGD models"""
        try:
            
            # Load and apply the preprocessing pipeline
            pipeline = ModelService.load_preprocessing_ead_lgd()
            processed_data = pipeline.transform(data)
            
            return processed_data
        except Exception as e:
            logger.error(f"Error preprocessing data for EAD/LGD: {e}")
            raise RuntimeError(f"Error preprocessing data for EAD/LGD: {e}")