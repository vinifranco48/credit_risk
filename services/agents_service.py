from typing import Annotated, Sequence, TypedDict, Union
import operator
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from groq import Groq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langgraph.graph import END, Graph, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

DB_PARAMS = {
    'dbname': 'mlflow',
    'user': 'mlflow',
    'password': 'secret',
    'host': 'localhost',
    'port': '5434'
}

GROQ_API_KEY = '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'

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    loan_data: Union[pd.DataFrame, None]
    risk_analysis: Union[pd.DataFrame, None]
    report: Union[pd.DataFrame, None]
    insights: list[str]
    next_steps: str
    errors:list[str]

class DataTools:
    def __init__(self, engine):
        self.engine = create_engine(
            f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@"
            f"{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
        )
    
    
    def fetch_loan_data(self, id:int) -> pd.DataFrame:

        query ="""
          SELECT f.id, f.term, f.int_rate, f.home_ownership, f.annual_inc, f.verification_status,
            f.purpose, f.dti, f.inq_last_6mths, f.mths_since_last_delinq, f.open_acc, f.revol_bal, f.initial_list_status,
            f.tot_cur_bal, f.mths_since_earliest_cr_line, l.credit_score
            FROM loan_features AS f
            INNER JOIN loan_predictions AS l ON f.id = l.id      
        """
        return pd.read_sql(query, self.engine, params={'id':id})
class RiskAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze_risk(self, data : pd.DataFrame) -> dict:
        risk_factors = {
            'credit_score_risk':self._evaluate_credit_score(data['credit_score'].iloc[0]),
            'income_risk':self._evaluate_income(data['annual_inc']).iloc[0],
            'dti_risk':self._evaluate_dti(data['dti']).iloc[0],
        }

        return risk_factors

    def generate_risk_report(self, data:pd.DataFrame, risk_factors:dict) -> str:
        prompt = f"""
                    Analise os seguintes dados de emprestimo e fatores de risco:

                    Dados de emprestimo
                    {data.to_dict()}

                    fatores de risco:
                    {risk_factors}

                    Gere um relatório detalhado que inclua:
                    1. Resumo dos principais riscos identificados
                    2. Comparação com perfis similares
                    3. Recomendações específicas
                    4. Sinais de alerta, se houver

                    """
        response =self.llm.chat.completions.create(
                    messages=[{"role":"user", "content":prompt}],
                    temperature=0.7     
                    )
        return response.choices[0].messages.content
    
    def _evaluate_credit_score(self, score: float) -> str:
        if score >= 750: return "BAIXO"
        elif score >= 650: return "MÉDIO"
        return "ALTO"
    
    def _evaluate_income(self, income: float) -> str:
        if income >= 100000: return "BAIXO"
        elif income >= 50000: return "MÉDIO"
        return "ALTO"
    
    def _evaluate_dti(self, dti: float) -> str:
        if dti <= 30: return "BAIXO"
        elif dti <= 45: return "MÉDIO"
        return "ALTO"

        
    



