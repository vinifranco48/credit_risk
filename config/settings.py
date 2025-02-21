from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # AWS e MLflow settings
    AWS_ACCESS_KEY_ID: str = 'mlflow'
    AWS_SECRET_ACCESS_KEY: str = 'password'
    MLFLOW_S3_ENDPOINT_URL: str = 'http://localhost:9005'
    MLFLOW_TRACKING_URI: str = 'http://localhost:5001'
    MODEL_RUN_ID: str = "78cfda90cdfa4811add1af83a4219400"
    PIPELINE_PATH: str = "artifact/pipeline.pkl"
    SCORECARD_PATH: str = "artifact/scorecard.csv"

    # Database settings - usando os mesmos nomes que aparecem no erro
    db_name: str = 'mlflow'
    db_user: str = 'mlflow'
    db_password: str = 'secret'
    db_host: str = 'pgsql'
    db_port: str = '5434'

    # GROQ settings
    groq_api_key: str = 'gsk_LmKMqU47zDl36vAc84G0...tQZwfrX9Io7QRHQP5x07lT7'
    groq_model: str = 'mixtral-8x7b-32768'

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,  # Mudado para False para aceitar variações de case
        extra='allow'  # Adicionado para permitir campos extras
    )