import boto3
import json
import logging
from typing import Dict, Any
from config.settings import Settings
import numpy as np

logger = logging.getLogger(__name__)
settings = Settings()


class StorageService:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = boto3.client(
                's3',
                endpoint_url=settings.MLFLOW_S3_ENDPOINT_URL,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
        return cls._client

    @classmethod
    async def save_prediction(cls, result: Dict[str, Any], path: str) -> None:
        try:
            result_processed = {
                k: float(v) if isinstance(v, np.floating) else v
                for k, v in result.items()
            }
            
            cls.get_client().put_object(
                Bucket="predictions",
                Key=path,
                Body=json.dumps(result_processed)
            )
            logger.info(f"Result saved to MinIO at {path}")
        except Exception as e:
            logger.error(f"Error saving to MinIO: {e}")
            raise