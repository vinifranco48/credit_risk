import os
import json
import logging
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from typing import Dict, Any

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

logger = logging.getLogger(__name__)

class StorageService:
    _client = None

    @classmethod
    def get_client(cls):
        """
        Retorna o cliente S3/MinIO. Se o cliente não existir, ele é criado.
        """
        if cls._client is None:
            try:
                session = boto3.Session(
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name='us-east-1'
                )
                cls._client = session.client(
                    's3',
                    endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
                    config=Config(
                        signature_version='s3v4',
                        retries={
                            'max_attempts': 3,
                            'mode': 'standard'
                        },
                        connect_timeout=5,
                        read_timeout=5
                    ),
                    verify=False,
                    use_ssl=False
                )
                logger.info("Cliente S3/MinIO criado com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao criar cliente S3/MinIO: {e}")
                raise
        return cls._client

    @classmethod
    def ensure_bucket_exists(cls, bucket_name: str = "mlflow") -> None:
        """
        Verifica se o bucket existe. Se não existir, cria o bucket.
        """
        client = cls.get_client()
        try:
            client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket '{bucket_name}' já existe.")
        except Exception as e:
            try:
                client.create_bucket(Bucket=bucket_name)
                logger.info(f"Bucket '{bucket_name}' criado com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao criar bucket '{bucket_name}': {e}")
                raise

    @classmethod
    async def upload_json(cls, path: str, data: Dict[str, Any], bucket_name: str = "mlflow") -> None:
        """
        Salva um objeto JSON no bucket especificado.
        """
        try:
            cls.ensure_bucket_exists(bucket_name)
            client = cls.get_client()
            client.put_object(
                Bucket=bucket_name,
                Key=path,
                Body=json.dumps(data),
                ContentType='application/json'
            )
            logger.info(f"Objeto salvo em '{bucket_name}/{path}'.")
        except Exception as e:
            logger.error(f"Erro ao salvar objeto no MinIO: {e}")
            raise

    @classmethod
    def test_connection(cls) -> bool:
        """
        Testa a conexão com o MinIO/S3.
        """
        try:
            client = cls.get_client()
            client.list_buckets()
            logger.info("Conexão com MinIO/S3 bem-sucedida.")
            return True
        except Exception as e:
            logger.error(f"Falha na conexão com MinIO/S3: {e}")
            return False

