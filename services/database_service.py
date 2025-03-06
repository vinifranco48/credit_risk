import json
from datetime import datetime
from minio import Minio
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import os

class MinioToPostgres:
    def __init__(self):
        try:
            # Configuração MinIO
            self.minio_client = Minio(
                "minio:9000",
                access_key="mlflow",
                secret_key="password",
                secure=False
            )
            
            # Configuração PostgreSQL
            database_url = os.getenv('DATABASE_URL', "postgresql://mlflow:secret@pgsql:5432/mlflow")
            self.conn = psycopg2.connect(database_url)
            self.cursor = self.conn.cursor()
            
            # Teste de conexão
            self.cursor.execute('SELECT 1')
            print("Conexão com PostgreSQL estabelecida com sucesso!")
            
        except Exception as e:
            print(f"Erro ao inicializar MinioToPostgres: {e}")
            raise

    def insert_feature(self, feature_data: Dict[str, Any]) -> int:
        try:
            print(f"Tentando inserir feature: {feature_data}")
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
                ) RETURNING id;
            """
            self.cursor.execute(sql, feature_data)
            feature_id = self.cursor.fetchone()[0]
            self.conn.commit()
            print(f"Feature inserida com sucesso! ID: {feature_id}")
            return feature_id
        except Exception as e:
            print(f"Erro ao inserir feature: {e}")
            self.conn.rollback()
            raise

    def insert_prediction(self, pred_data: Dict[str, Any]):
        """Insere dados de predição"""
        try:
            print(f"Tentando inserir predição: {pred_data}")
            sql = """
                INSERT INTO loan_predictions (
                    prediction, probability, credit_score,
                    prediction_timestamp, minio_storage_path, feature_id
                ) VALUES (
                    %(prediction)s, %(probability)s, %(credit_score)s,
                    %(prediction_timestamp)s, %(minio_storage_path)s, %(feature_id)s
                )
            """
            
            # Converte o timestamp para datetime se for string
            if isinstance(pred_data['prediction_timestamp'], str):
                pred_data['prediction_timestamp'] = datetime.fromisoformat(pred_data['prediction_timestamp'])
            
            self.cursor.execute(sql, pred_data)
            self.conn.commit()
            print("Predição inserida com sucesso!")
        
            
        except Exception as e:
            print(f"Erro ao inserir predição: {e}")
            self.conn.rollback()
            raise
    def create_tables(self):
        """Cria as tabelas necessárias no PostgreSQL se elas não existirem"""
        try:
            # Criação da tabela loan_features
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS loan_features (
                    id SERIAL PRIMARY KEY,
                    loan_amnt FLOAT,
                    term VARCHAR(20),
                    int_rate FLOAT,
                    grade VARCHAR(1),
                    sub_grade VARCHAR(2),
                    emp_length VARCHAR(20),
                    home_ownership VARCHAR(20),
                    annual_inc FLOAT,
                    verification_status VARCHAR(20),
                    purpose VARCHAR(50),
                    addr_state VARCHAR(2),
                    dti FLOAT,
                    inq_last_6mths INTEGER,
                    mths_since_last_delinq INTEGER,
                    open_acc INTEGER,
                    revol_bal FLOAT,
                    total_acc INTEGER,
                    initial_list_status VARCHAR(1),
                    tot_cur_bal FLOAT,
                    mths_since_earliest_cr_line INTEGER
                )
            """)

            # Criação da tabela loan_predictions
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS loan_predictions (
                    id SERIAL PRIMARY KEY,
                    prediction INTEGER,
                    probability FLOAT,
                    credit_score FLOAT,
                    prediction_timestamp TIMESTAMP,
                    minio_storage_path VARCHAR(255),
                    feature_id INTEGER REFERENCES loan_features(id)
                )
            """)

            self.conn.commit()
            print("Tabelas criadas com sucesso!")
            
        except Exception as e:
            print(f"Erro ao criar tabelas: {e}")
            self.conn.rollback()
            raise
if __name__ == "__main__":
    migrator = MinioToPostgres()
    migrator.create_tables()
    migrator.process_mlflow_data()