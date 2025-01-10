import json
from datetime import datetime
from minio import Minio
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any, List
from datetime import datetime, timedelta

class MinioToPostgres:
    def __init__(self):
        # Configuração MinIO
        self.minio_client = Minio(
            "localhost:9005",
            access_key="mlflow",
            secret_key="password",
            secure=False
        )
        
        # Configuração PostgreSQL
        self.pg_conn = psycopg2.connect(
            dbname="mlflow",
            user="mlflow",
            password="secret",
            host="localhost",
            port="5434"
        )
        self.pg_cursor = self.pg_conn.cursor()
        
    def create_tables(self):
        """Cria as tabelas necessárias no PostgreSQL"""
        # Tabela para features
        self.pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS loan_features (
                id SERIAL PRIMARY KEY,
                loan_amnt FLOAT,
                term INTEGER,
                int_rate FLOAT,
                grade VARCHAR(2),
                sub_grade VARCHAR(3),
                emp_length INTEGER,
                home_ownership VARCHAR(10),
                annual_inc FLOAT,
                verification_status VARCHAR(20),
                purpose VARCHAR(30),
                addr_state VARCHAR(2),
                dti FLOAT,
                inq_last_6mths INTEGER,
                mths_since_last_delinq FLOAT,
                open_acc INTEGER,
                revol_bal FLOAT,
                total_acc FLOAT,
                initial_list_status CHAR(1),
                tot_cur_bal FLOAT,
                mths_since_earliest_cr_line FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela para predictions
        self.pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS loan_predictions (
                id SERIAL PRIMARY KEY,
                prediction INTEGER,
                probability FLOAT,
                credit_score FLOAT,
                prediction_timestamp TIMESTAMP,
                minio_storage_path VARCHAR(255),
                feature_id INTEGER REFERENCES loan_features(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.pg_conn.commit()

    def get_all_predictions(self, bucket_name: str) -> List[str]:
        """Recupera todos os arquivos de predictions do bucket"""
        predictions = []
        objects = self.minio_client.list_objects(bucket_name, prefix="predictions/")
        for obj in objects:
            predictions.append(obj.object_name)
        return predictions
    
    def find_matching_prediction(self, feature_timestamp: str, predictions: List[str]) -> str:
        """Encontra o arquivo de prediction mais próximo do timestamp da feature"""
        feature_dt = datetime.fromisoformat(feature_timestamp)
        
        # Procura exatamente 3 segundos após o timestamp da feature
        target_dt = feature_dt + timedelta(seconds=3)
        target_timestamp = target_dt.isoformat()
        
        for pred_path in predictions:
            pred_timestamp = pred_path.split('/')[-1].replace('.json', '')
            if pred_timestamp.startswith(feature_dt.strftime("%Y-%m-%dT%H:%M")):
                return pred_path
                
        return None

    def get_minio_object(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """Recupera e decodifica um objeto JSON do MinIO"""
        try:
            data = self.minio_client.get_object(bucket_name, object_name)
            return json.loads(data.read().decode('utf-8'))
        except Exception as e:
            print(f"Erro ao ler objeto {object_name}: {e}")
            return None

    def insert_feature(self, feature_data: Dict[str, Any]) -> int:
        """Insere dados de features e retorna o ID"""
        sql = """
            INSERT INTO loan_features (
                loan_amnt, term, int_rate, grade, sub_grade, emp_length,
                home_ownership, annual_inc, verification_status, purpose,
                addr_state, dti, inq_last_6mths, mths_since_last_delinq,
                open_acc, revol_bal, total_acc, initial_list_status,
                tot_cur_bal, mths_since_earliest_cr_line
            ) VALUES (
                %(loan_amnt)s, %(term)s, %(int_rate)s, %(grade)s, %(sub_grade)s,
                %(emp_length)s, %(home_ownership)s, %(annual_inc)s,
                %(verification_status)s, %(purpose)s, %(addr_state)s, %(dti)s,
                %(inq_last_6mths)s, %(mths_since_last_delinq)s, %(open_acc)s,
                %(revol_bal)s, %(total_acc)s, %(initial_list_status)s,
                %(tot_cur_bal)s, %(mths_since_earliest_cr_line)s
            ) RETURNING id
        """
        self.pg_cursor.execute(sql, feature_data)
        return self.pg_cursor.fetchone()[0]

    def insert_prediction(self, pred_data: Dict[str, Any], feature_id: int):
        """Insere dados de predição"""
        sql = """
            INSERT INTO loan_predictions (
                prediction, probability, credit_score,
                prediction_timestamp, minio_storage_path, feature_id
            ) VALUES (
                %(prediction)s, %(probability)s, %(credit_score)s,
                %(prediction_timestamp)s, %(minio_storage_path)s, %(feature_id)s
            )
        """
        pred_data['feature_id'] = feature_id
        pred_data['prediction_timestamp'] = datetime.fromisoformat(pred_data['prediction_timestamp'])
        self.pg_cursor.execute(sql, pred_data)

    def process_mlflow_data(self, bucket_name: str = "mlflow"):
        """Processa os dados do bucket MLflow"""
        try:
            # Obtém lista de todos os arquivos de predictions
            all_predictions = self.get_all_predictions(bucket_name)
            print(f"Total de predictions encontradas: {len(all_predictions)}")
            
            # Lista todos os objetos no bucket
            objects = self.minio_client.list_objects(bucket_name, prefix="features/")
            
            for obj in objects:
                if obj.object_name.endswith('.json'):
                    print(f"\nProcessando feature: {obj.object_name}")
                    feature_data = self.get_minio_object(bucket_name, obj.object_name)
                    
                    if feature_data:
                        feature_timestamp = obj.object_name.split('/')[-1].replace('.json', '')
                        matching_prediction = self.find_matching_prediction(feature_timestamp, all_predictions)
                        
                        if matching_prediction:
                            print(f"Encontrada prediction correspondente: {matching_prediction}")
                            pred_data = self.get_minio_object(bucket_name, matching_prediction)
                            
                            if pred_data:
                                feature_id = self.insert_feature(feature_data)
                                self.insert_prediction(pred_data, feature_id)
                                print("Dados inseridos com sucesso!")
                        else:
                            print(f"Não foi encontrada prediction para a feature {feature_timestamp}")
                
            self.pg_conn.commit()
            print("\nTodos os dados foram processados com sucesso!")
            
        except Exception as e:
            self.pg_conn.rollback()
            print(f"Erro durante o processamento: {e}")
        
        finally:
            self.pg_cursor.close()
            self.pg_conn.close()

if __name__ == "__main__":
    migrator = MinioToPostgres()
    migrator.create_tables()
    migrator.process_mlflow_data()