from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = 'mlflow'
    AWS_SECRET_ACCESS_KEY: str = 'password'
    MLFLOW_S3_ENDPOINT_URL: str = 'http://localhost:9005'
    MLFLOW_TRACKING_URI: str = 'http://localhost:5001'
    MODEL_RUN_ID: str = "44379cb8e4424cecb8859cb2be4ba31b"
    PIPELINE_PATH: str = "artifact/pipeline.pkl"
    SCORECARD_PATH: str = "artifact/scorecard.csv"

    class Config:
        env_file = ".env"