from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import json
import datetime
import logging
from services.model_services import ModelService
from services.storage_service import StorageService
from src.utils_modelling import compute_credit_scores
from services.database_service import MinioToPostgres


logger = logging.getLogger(__name__)  # noqa: F821
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

class PredictionService:
    FEATURE_ORDER = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length',
        'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'addr_state',
        'dti', 'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'revol_bal',
        'total_acc', 'initial_list_status', 'tot_cur_bal', 'mths_since_earliest_cr_line'
    ]

    # Lista das colunas esperadas pelos modelos EAD e LGD
    EAD_LGD_COLUMNS = [
        'loan_amnt', 'int_rate', 'emp_length', 'annual_inc', 'dti', 'inq_last_6mths',
        'mths_since_last_delinq', 'open_acc', 'revol_bal', 'total_acc', 'tot_cur_bal',
        'mths_since_earliest_cr_line', 'term_60', 'region_Northeast', 'region_South', 
        'region_West', 'home_ownership_OWN', 'home_ownership_RENT_NONE_OTHER',
        'purpose_debt_consolidation', 'purpose_home_improvement', 'purpose_house_car_medical',
        'purpose_major_purchase', 'purpose_other', 'purpose_small_business',
        'purpose_vacation_moving_wedding', 'verification_status_Source Verified',
        'verification_status_Verified', 'initial_list_status_w', 'grade', 'sub_grade'
    ]

    @staticmethod
    def reorder_features(df: pd.DataFrame) -> pd.DataFrame:
        return df[PredictionService.FEATURE_ORDER]

    @staticmethod
    def process_prediction(prediction: Any) -> Tuple[int, float]:
        if isinstance(prediction, (list, np.ndarray)):
            prediction = prediction[0]
        
        if isinstance(prediction, np.floating):
            prediction = float(prediction)

        probability = float(prediction)
        final_prediction = 1 if probability > 0.5 else 0

        return final_prediction, probability

    @staticmethod
    def calculate_expected_loss(pd: float, lgd: float, ead: float, loan_amount: float) -> float:
        """Calculate expected loss as PD * LGD * EAD"""
        return pd * lgd * ead * loan_amount

    @staticmethod
    def ensure_dataframe_format(data, column_names):
        """
        Garante que os dados estejam no formato DataFrame com as colunas corretas.
        Se for um array NumPy, converte para DataFrame com as colunas especificadas.
        """
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data, columns=column_names)
        return data

    @staticmethod
    async def predict(features: Dict[str, Any]) -> Dict[str, Any]:
        migrator = None
        try:
            migrator = MinioToPostgres() 
            migrator.create_tables()
            logger.info("Iniciando processo de predição...")
            
            # Add default values for missing required features
            features['grade'] = features.get('grade', 'C')  # Using 'C' as a middle-ground default grade
            features['sub_grade'] = features.get('sub_grade', 'C3')  # Using 'C3' as a middle sub-grade
            features['emp_length'] = features.get('emp_length', '5 years')  # Using '5 years' as a middle value
            
            # Handle optional features with default values
            if 'mths_since_earliest_cr_line' not in features or features['mths_since_earliest_cr_line'] is None:
                features['mths_since_earliest_cr_line'] = 60.0  # Default to 5 years of credit history
                
            if 'mths_since_last_delinq' not in features or features['mths_since_last_delinq'] is None:
                features['mths_since_last_delinq'] = 0.0  # Default to 0 if no delinquency history

            if 'addr_state' not in features or features['addr_state'] is None:
                features['addr_state'] = 'CA'  # Default to California if state is missing
            
            input_df = pd.DataFrame([features])
            input_df = PredictionService.reorder_features(input_df)

            # Store input features in MinIO
            timestamp = datetime.datetime.now().isoformat()
            features_path = f"features/{timestamp}.json"
            await StorageService.upload_json(features_path, features, bucket_name="mlflow")

            # Legacy model prediction with original pipeline
            pipeline = ModelService.load_preprocessing_pipeline()
            processed_data = pipeline.transform(input_df)
            
            # Make predictions for PD model - Store pd_prediction here
            model = ModelService.load_model()
            raw_prediction = model.predict(processed_data)
            prediction, probability = PredictionService.process_prediction(raw_prediction)
            pd_prediction = probability  # Store PD prediction

            # Calculate credit score from legacy model
            scorecard = ModelService.load_scorecard()
            credit_score = compute_credit_scores(
                X=processed_data,
                probas=probability,
                scorecard=scorecard
            )

            input_df['region'] = input_df['addr_state'].map(state_to_region)
            
            # Use different preprocessing for LGD and EAD models
            ead_lgd_pipeline = ModelService.load_preprocessing_ead_lgd()
            processed_data_ead_lgd = ead_lgd_pipeline.transform(input_df)
            
            # CORREÇÃO: Garantir que os dados estejam no formato DataFrame com as colunas nomeadas
            processed_data_ead_lgd_df = PredictionService.ensure_dataframe_format(
                processed_data_ead_lgd, 
                PredictionService.EAD_LGD_COLUMNS
            )
            
            # Verificação adicional para garantir que todas as colunas necessárias estejam presentes
            logger.debug(f"Colunas nos dados processados: {processed_data_ead_lgd_df.columns.tolist()}")
            
            # Make predictions with EAD and LGD models using the properly formatted data
            lgd_linear_model = ModelService.load_lgd_model()
            lgd_logistic_model = ModelService.load_lgd_logistic_model()
            ead_model = ModelService.load_ead_model()
            
            lgd_linear_prediction = float(lgd_linear_model.predict(processed_data_ead_lgd_df)[0])
            lgd_logistic_prediction = float(lgd_logistic_model.predict(processed_data_ead_lgd_df)[0])
            ead_prediction = float(ead_model.predict(processed_data_ead_lgd_df)[0])
            
            # Calculate expected loss
            loan_amount = float(features.get('loan_amnt', 0))
            expected_loss = PredictionService.calculate_expected_loss(
                pd=pd_prediction,  # Using the stored pd_prediction
                lgd=lgd_linear_prediction,
                ead=ead_prediction,
                loan_amount=loan_amount
            )

            # Prepare result dictionary
            timestamp = datetime.datetime.now().isoformat()
            minio_path = f"predictions/{timestamp}.json"

            result = {
                # Legacy model results
                "prediction": prediction,
                "probability": probability,
                "credit_score": credit_score,
                
                # New model results
                "pd_score": pd_prediction,
                "lgd_linear_score": lgd_linear_prediction,
                "lgd_logistic_score": lgd_logistic_prediction,
                "ead_score": ead_prediction,
                "expected_loss": expected_loss,
                
                # Metadata
                "prediction_timestamp": timestamp,
                "minio_storage_path": minio_path
            }

            # Save result to MinIO
            await StorageService.upload_json(minio_path, result)

            # Save to PostgreSQL
            logger.info("Iniciando salvamento no PostgreSQL...")
            
            # Save features first
            feature_id = migrator.insert_feature(features)
            logger.info(f"Features salvas com ID: {feature_id}")
            
            # Then save prediction
            prediction_data = {
                **result,
                "feature_id": feature_id
            }
            migrator.insert_prediction(prediction_data)
            logger.info("Predição salva com sucesso!")

            return result
    
        except Exception as e:
            logger.error(f"Erro durante a predição: {e}")
            if migrator and hasattr(migrator, 'pg_conn'):
                migrator.pg_conn.rollback()
            raise RuntimeError(f"Error predicting: {e}")
        
        finally:
            if migrator:
                if hasattr(migrator, 'pg_cursor') and migrator.pg_cursor:
                    migrator.pg_cursor.close()
                if hasattr(migrator, 'pg_conn') and migrator.pg_conn:
                    migrator.pg_conn.close()
                logger.info("Conexões fechadas com sucesso!")