from typing import Annotated, Sequence, TypedDict, Union
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langgraph.graph import END, Graph, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Configurações do banco de dados e API
DB_CONFIG = {
    'dbname': 'riskdb',
    'user': 'risk_analyst',
    'password': 'secure_pass',
    'host': 'localhost',
    'port': '5432'
}

# Definição do estado do agente
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    loan_data: Union[pd.DataFrame, None]
    risk_analysis: Union[dict, None]
    report: str
    insights: list[str]
    next_steps: str
    errors: list[str]

# Classe para manipulação de dados
class DataManager:
    def __init__(self, db_config):
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
    
    def fetch_loan_data(self, loan_id: str) -> pd.DataFrame:
        query = """
        SELECT 
            l.loan_id,
            l.credit_score,
            l.annual_income,
            l.debt_to_income,
            l.employment_length,
            l.loan_amount,
            h.payment_history,
            h.delinquencies,
            h.collections
        FROM loan_applications l
        LEFT JOIN credit_history h ON l.loan_id = h.loan_id
        WHERE l.loan_id = :loan_id
        """
        return pd.read_sql(query, self.engine, params={'loan_id': loan_id})

# Classe para análise de risco
class RiskAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def analyze_risk_factors(self, loan_data: pd.DataFrame) -> dict:
        risk_factors = {
            'credit_score_risk': self._evaluate_credit_score(loan_data['credit_score'].iloc[0]),
            'income_risk': self._evaluate_income(loan_data['annual_income'].iloc[0]),
            'dti_risk': self._evaluate_dti(loan_data['debt_to_income'].iloc[0]),
            'history_risk': self._evaluate_history(loan_data)
        }
        return risk_factors
    
    def generate_risk_report(self, loan_data: pd.DataFrame, risk_factors: dict) -> str:
        prompt = f"""
        Analise os seguintes dados de empréstimo e fatores de risco:
        
        Dados do empréstimo:
        {loan_data.to_dict()}
        
        Fatores de risco:
        {risk_factors}
        
        Gere um relatório detalhado que inclua:
        1. Resumo dos principais riscos identificados
        2. Comparação com perfis similares
        3. Recomendações específicas
        4. Sinais de alerta, se houver
        """
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _evaluate_credit_score(self, score: float) -> str:
        if score >= 750: return "BAIXO"
        elif score >= 650: return "MÉDIO"
        return "ALTO"
    
    def _evaluate_income(self, income: float) -> str:
        if income >= 100000: return "BAIXO"
        elif income >= 50000: return "MÉDIO"
        return "ALTO"
    
    def _evaluate_dti(self, dti: float) -> str:
        if dti <= 0.36: return "BAIXO"
        elif dti <= 0.43: return "MÉDIO"
        return "ALTO"
    
    def _evaluate_history(self, loan_data: pd.DataFrame) -> str:
        delinquencies = loan_data['delinquencies'].iloc[0]
        collections = loan_data['collections'].iloc[0]
        
        if delinquencies == 0 and collections == 0: return "BAIXO"
        elif delinquencies <= 1 and collections == 0: return "MÉDIO"
        return "ALTO"

# Classe principal do sistema
class LoanRiskAnalysisSystem:
    def __init__(self, db_config, llm_client):
        self.data_manager = DataManager(db_config)
        self.risk_analyzer = RiskAnalyzer(llm_client)
    
    def process_loan_application(self, loan_id: str) -> AgentState:
        state = AgentState(
            messages=[],
            loan_data=None,
            risk_analysis=None,
            report="",
            insights=[],
            next_steps="",
            errors=[]
        )
        
        try:
            # Buscar dados do empréstimo
            state['loan_data'] = self.data_manager.fetch_loan_data(loan_id)
            
            # Realizar análise de risco
            state['risk_analysis'] = self.risk_analyzer.analyze_risk_factors(state['loan_data'])
            
            # Gerar relatório
            state['report'] = self.risk_analyzer.generate_risk_report(
                state['loan_data'],
                state['risk_analysis']
            )
            
            # Adicionar insights baseados na análise
            state['insights'] = self._generate_insights(state['risk_analysis'])
            
            # Definir próximos passos
            state['next_steps'] = self._determine_next_steps(state['risk_analysis'])
            
        except Exception as e:
            state['errors'].append(str(e))
        
        return state
    
    def _generate_insights(self, risk_analysis: dict) -> list[str]:
        insights = []
        risk_levels = list(risk_analysis.values())
        
        if all(risk == "BAIXO" for risk in risk_levels):
            insights.append("Perfil de risco excepcionalmente baixo")
        elif risk_analysis['credit_score_risk'] == "ALTO":
            insights.append("Histórico de crédito preocupante - requer atenção especial")
        elif risk_analysis['dti_risk'] == "ALTO":
            insights.append("Alta taxa de endividamento - considerar reestruturação")
            
        return insights
    
    def _determine_next_steps(self, risk_analysis: dict) -> str:
        high_risks = sum(1 for risk in risk_analysis.values() if risk == "ALTO")
        
        if high_risks == 0:
            return "Prosseguir com a aprovação do empréstimo"
        elif high_risks == 1:
            return "Solicitar documentação adicional para área de alto risco"
        else:
            return "Encaminhar para análise manual detalhada"