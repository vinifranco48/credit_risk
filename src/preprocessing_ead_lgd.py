from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class EadLgdPreprocessor(BaseEstimator, TransformerMixin):
    def _init_(self):
        self.numeric_features = [
            'loan_amnt', 'int_rate', 'annual_inc', 'dti',
            'inq_last_6mths', 'open_acc', 'revol_bal',
            'total_acc', 'tot_cur_bal', 'mths_since_earliest_cr_line'
        ]
        
        self.categorical_features = [
            'term', 'grade', 'home_ownership', 
            'verification_status', 'purpose', 'addr_state'
        ]

    def fit(self, X, y=None):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), self.categorical_features)
            ])
        
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

def create_ead_lgd_pipeline():
    return Pipeline([
        ('preprocessor', EadLgdPreprocessor())
    ])