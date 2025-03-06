from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # AWS e MLflow settings
    AWS_ACCESS_KEY_ID: str = 'mlflow'
    AWS_SECRET_ACCESS_KEY: str = 'password'
    MLFLOW_S3_ENDPOINT_URL: str = 'http://localhost:9005'
    MLFLOW_TRACKING_URI: str = 'http://localhost:5001'
    MODEL_PD: str = "6a95bc29784b47dea6a1c6666f4e99d9"
    MODEL_LGD_LINEAR: str = "c9b1ba14a6844f51aa322d9ed2e1f0ee"
    MODEL_LGD_LOGISTIC: str = "99e0f3c0b1c54b1695bde3276ee58540"
    MODEL_EAD: str = "04c3e66bfa71494ba0148b7c93e3f9b0"
    PIPELINE_PATH: str = "artifact/pipeline.pkl"
    SCORECARD_PATH: str = "artifact/scorecard.csv"
    PIPELINE_EAD_LGD: str = "artifact/preprocessor_pipeline.pkl"

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