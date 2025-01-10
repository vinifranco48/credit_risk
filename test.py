from minio import Minio

def list_minio_objects():
    # Configuração MinIO
    minio_client = Minio(
        "localhost:9005",
        access_key="mlflow",
        secret_key="password",
        secure=False
    )
    
    bucket_name = "mlflow"
    
    try:
        objects = minio_client.list_objects(bucket_name, recursive=True)
        print("Objetos encontrados no bucket mlflow:")
        print("-" * 50)
        for obj in objects:
            print(f"Nome: {obj.object_name}")
            print(f"Tamanho: {obj.size} bytes")
            print(f"Última modificação: {obj.last_modified}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Erro ao listar objetos: {e}")

if __name__ == "__main__":
    list_minio_objects()