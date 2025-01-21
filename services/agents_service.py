from typing import Annotated, Sequence, TypedDict, Union
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from groq import Groq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langgraph.graph import END, Graph, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Configuration should be in a separate config file in practice
DB_PARAMS = {
    'dbname': 'mlflow',
    'user': 'mlflow',
    'password': 'secret',
    'host': 'localhost',
    'port': '5434'
}

GROQ_API_KEY = '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'
GROQ_MODEL = "mixtral-8x7b-32768"  # ou "llama2-70b-4096"

class AgentState(TypedDict):
    """Type definition for the agent's state"""
    messages: Sequence[BaseMessage]
    loan_data: Union[pd.DataFrame, None]
    risk_analysis: Union[dict, None]
    report: str        
    insights: list[str]
    next_steps: str
    errors: list[str]

class DataTools:
    """Class for handling data operations"""
    def __init__(self):
        self.engine = create_engine(
            f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@"
            f"{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
        )
    
    def fetch_loan_data(self, loan_id: int) -> pd.DataFrame:
        """Fetch loan data from database"""
        query = text("""
            SELECT f.id, f.term, f.int_rate, f.home_ownership, f.annual_inc, f.verification_status,
                f.purpose, f.dti, f.inq_last_6mths, f.mths_since_last_delinq, f.open_acc, 
                f.revol_bal, f.initial_list_status, f.tot_cur_bal, f.mths_since_earliest_cr_line, 
                l.credit_score
            FROM loan_features AS 
            INNER JOIN loan_predictions AS l ON f.id = l.id
            WHERE f.id = :id
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'id': loan_id})
            
        if df.empty:
            raise ValueError(f"No loan data found for ID: {loan_id}")
            
        return df

class RiskAnalyzer:
    """Class for analyzing loan risks"""
    def __init__(self, llm_client: Groq):
        self.llm = llm_client

    def analyze_risk(self, data: pd.DataFrame) -> dict:
        """Analyze various risk factors"""
        if data.empty:
            raise ValueError("No data provided for risk analysis")
            
        risk_factors = {
            'credit_score_risk': self._evaluate_credit_score(data['credit_score'].iloc[0]),
            'income_risk': self._evaluate_income(data['annual_inc'].iloc[0]),
            'dti_risk': self._evaluate_dti(data['dti'].iloc[0]),
        }
        return risk_factors

    def generate_risk_report(self, data: pd.DataFrame, risk_factors: dict) -> str:
        """Generate a detailed risk report"""
        if data.empty:
            raise ValueError("No data provided for report generation")
            
        prompt = f"""
            Analise os seguintes dados de empréstimo e fatores de risco:

            Dados de empréstimo:
            {data.to_dict('records')[0]}

            Fatores de risco:
            {risk_factors}

            Gere um relatório detalhado que inclua:
            1. Resumo dos principais riscos identificados
            2. Comparação com perfis similares
            3. Recomendações específicas
            4. Sinais de alerta, se houver
        """
        
        response = self.llm.chat.completions.create(
            model=GROQ_MODEL,  # Adicionando o parâmetro model obrigatório
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    @staticmethod
    def _evaluate_credit_score(score: float) -> str:
        """Evaluate credit score risk level"""
        if score >= 750:
            return "BAIXO"
        elif score >= 650:
            return "MÉDIO"
        return "ALTO"
    
    @staticmethod
    def _evaluate_income(income: float) -> str:
        """Evaluate income risk level"""
        if income >= 100000:
            return "BAIXO"
        elif income >= 50000:
            return "MÉDIO"
        return "ALTO"
    
    @staticmethod
    def _evaluate_dti(dti: float) -> str:
        """Evaluate debt-to-income ratio risk level"""
        if dti <= 30:
            return "BAIXO"
        elif dti <= 45:
            return "MÉDIO"
        return "ALTO"

class LoanRiskAnalysisSystem:
    """Main system class for loan risk analysis"""
    def __init__(self, llm_client: Groq):
        self.data_tools = DataTools()
        self.risk_analyzer = RiskAnalyzer(llm_client)
    
    def process_loan_application(self, loan_id: int) -> AgentState:
        """Process a loan application and generate analysis"""
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
            # Fetch loan data
            state['loan_data'] = self.data_tools.fetch_loan_data(loan_id)
            
            # Perform risk analysis
            state['risk_analysis'] = self.risk_analyzer.analyze_risk(state['loan_data'])
            
            # Generate report
            state['report'] = self.risk_analyzer.generate_risk_report(
                state['loan_data'],
                state['risk_analysis']
            )
            
            # Generate insights and next steps
            state['insights'] = self._generate_insights(state['risk_analysis'])
            state['next_steps'] = self._determine_next_steps(state['risk_analysis'])
            
        except Exception as e:
            state['errors'].append(str(e))
            
        return state
    
    @staticmethod
    def _generate_insights(risk_analysis: dict) -> list[str]:
        """Generate insights based on risk analysis"""
        insights = []
        risk_levels = list(risk_analysis.values())
        
        if all(risk == "BAIXO" for risk in risk_levels):
            insights.append("Perfil de risco excepcionalmente baixo")
        if risk_analysis['credit_score_risk'] == "ALTO":
            insights.append("Histórico de crédito preocupante - requer atenção especial")
        if risk_analysis['dti_risk'] == "ALTO":
            insights.append("Alta taxa de endividamento - considerar reestruturação")
            
        return insights
    
    @staticmethod
    def _determine_next_steps(risk_analysis: dict) -> str:
        """Determine next steps based on risk analysis"""
        high_risks = sum(1 for risk in risk_analysis.values() if risk == "ALTO")
        
        if high_risks == 0:
            return "Prosseguir com a aprovação do empréstimo"
        elif high_risks == 1:
            return "Solicitar documentação adicional para área de alto risco"
        return "Encaminhar para análise manual detalhada"

def main():
    try:
        # Initialize Groq client
        llm_client = Groq(api_key=GROQ_API_KEY)
        
        # Initialize the Loan Risk Analysis System
        loan_system = LoanRiskAnalysisSystem(llm_client)
        
        # Process a sample loan application
        loan_id = 1  # Example loan ID
        print(f"\nProcessing loan application ID: {loan_id}")
        print("-" * 50)
        
        result = loan_system.process_loan_application(loan_id)
        
        # Check for errors
        if result['errors']:
            print("\nErrors encountered:")
            for error in result['errors']:
                print(f"- {error}")
            return
        
        # Display loan data
        print("\nLoan Data:")
        print("-" * 50)
        print(result['loan_data'].to_string())
        
        # Display risk analysis
        print("\nRisk Analysis:")
        print("-" * 50)
        for factor, risk in result['risk_analysis'].items():
            print(f"{factor}: {risk}")
        
        # Display insights
        print("\nInsights:")
        print("-" * 50)
        for insight in result['insights']:
            print(f"- {insight}")
        
        # Display next steps
        print("\nNext Steps:")
        print("-" * 50)
        print(result['next_steps'])
        
        # Display detailed report
        print("\nDetailed Report:")
        print("-" * 50)
        print(result['report'])
        
    except Exception as e:
        print(f"Error running the system: {str(e)}")

if __name__ == "__main__":
    main()