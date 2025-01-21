from typing import Annotated, Sequence, TypedDict, Union
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from groq import Groq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Configuration
DB_PARAMS = {
    "dbname": "mlflow",
    "user": "mlflow",
    "password": "secret",
    "host": "pgsql",
    "port": "5432"
}

GROQ_API_KEY = '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'
GROQ_MODEL = "mixtral-8x7b-32768"

class AgentState(TypedDict):
    """Type definition for the agent's state"""
    messages: Sequence[BaseMessage]
    current_data: Union[pd.DataFrame, None]
    context: dict

class DataTools:
    """Enhanced class for handling database-wide operations"""
    def __init__(self):
        self.engine = create_engine(
            f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@"
            f"{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
        )
    
    def get_base_query(self, limit: int = 1000) -> str:
        """Get the base query for loan data"""
        return f"""
            SELECT 
                f.id, 
                f.term, 
                f.int_rate as interest_rate, 
                f.home_ownership, 
                f.annual_inc as annual_income,
                f.verification_status,
                f.purpose, 
                f.dti as debt_to_income_ratio,
                f.inq_last_6mths as inquiries_last_6months,
                f.mths_since_last_delinq as months_since_delinquency,
                f.open_acc as open_accounts,
                f.revol_bal as revolving_balance,
                f.initial_list_status,
                f.tot_cur_bal as total_current_balance,
                f.mths_since_earliest_cr_line as credit_history_months,
                l.credit_score
            FROM loan_features AS f
            INNER JOIN loan_predictions AS l ON f.id = l.id
            LIMIT {limit}
        """

    def fetch_loan_data(self, limit: int = 1000) -> pd.DataFrame:
        """Fetch loans with better column names"""
        query = self.get_base_query(limit)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            
        if df.empty:
            raise ValueError("No loan data found")
            
        return df
    
    def get_statistics(self, df: pd.DataFrame, column: str) -> dict:
        """Get statistical information about a column"""
        stats = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'min': df[column].min(),
            'max': df[column].max(),
            'std': df[column].std()
        }
        
        if df[column].dtype in ['int64', 'float64']:
            stats.update({
                'q1': df[column].quantile(0.25),
                'q3': df[column].quantile(0.75)
            })
            
        return stats

class ConversationalAnalyzer:
    """Enhanced class for database-wide interactive analysis"""
    def __init__(self, llm_client: Groq):
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL
        )
        self.data_tools = DataTools()

    def analyze_trends(self, data: pd.DataFrame, field: str) -> str:
        """Analyze trends in the data for a specific field"""
        stats = self.data_tools.get_statistics(data, field)
        
        prompt = f"""
        Analise as tendências do campo {field} na base de empréstimos:

        Estatísticas:
        {stats}
        
        Por favor:
        1. Faça uma análise detalhada da distribuição dos valores
        2. Identifique padrões ou tendências relevantes
        3. Forneça insights sobre o que esses números significam para análise de risco
        
        Lembre-se:
        - Seja conversacional e analítico
        - Use analogias quando apropriado
        - Explique os termos técnicos
        - Indique quando um valor parecer atípico ou preocupante
        """
        
        response = self.llm.invoke(prompt)
        return response.content

    def answer_query(self, query: str, data: pd.DataFrame) -> str:
        """Generate response to user query about the loan data"""
        numeric_summary = data.describe()
        categorical_summary = {col: data[col].value_counts().to_dict() 
                             for col in data.select_dtypes(include=['object']).columns}
        
        prompt = f"""
        Você é um analista de crédito experiente analisando uma base de empréstimos.
        
        Resumo dos dados numéricos:
        {numeric_summary.to_dict()}
        
        Resumo dos dados categóricos:
        {categorical_summary}
        
        Pergunta do usuário: {query}
        
        Por favor:
        1. Responda à pergunta de forma direta e clara
        2. Use os dados disponíveis para fundamentar sua resposta
        3. Forneça insights relevantes
        4. Sugira pontos adicionais para investigação quando apropriado
        
        Seja conversacional mas mantenha o rigor analítico.
        Use analogias e exemplos quando ajudar a explicar conceitos complexos.
        """
        
        response = self.llm.invoke(prompt)
        return response.content

def process_user_input(state: AgentState, user_input: str) -> AgentState:
    """Process user input and update state"""
    analyzer = ConversationalAnalyzer(Groq(api_key=GROQ_API_KEY))
    
    # Add user message to history
    state['messages'].append(HumanMessage(content=user_input))
    
    # Fetch data if not already present
    if state['current_data'] is None:
        state['current_data'] = analyzer.data_tools.fetch_loan_data()
    
    # Generate response based on user input
    if "tendência" in user_input.lower() or "tendencias" in user_input.lower():
        field = next((f for f in state['current_data'].columns 
                     if f in user_input.lower()), None)
        if field:
            response = analyzer.analyze_trends(state['current_data'], field)
        else:
            response = "Qual campo você gostaria de analisar as tendências?"
    else:
        response = analyzer.answer_query(user_input, state['current_data'])
    
    # Add AI response to history
    state['messages'].append(AIMessage(content=response))
    
    return state

def main():
    try:
        # Initialize system
        analyzer = ConversationalAnalyzer(Groq(api_key=GROQ_API_KEY))
        
        # Initialize state
        state = AgentState(
            messages=[],
            current_data=None,
            context={}
        )
        
        print("\nSistema de Análise de Empréstimos Iniciado")
        print("-" * 50)
        
        # Interactive loop
        while True:
            user_input = input("\nO que você gostaria de analisar? (ou 'sair' para encerrar): ")
            if user_input.lower() == 'sair':
                break
                
            state = process_user_input(state, user_input)
            print("\nResposta:")
            print("-" * 50)
            print(state['messages'][-1].content)
            
    except Exception as e:
        print(f"Erro ao executar o sistema: {str(e)}")

if __name__ == "__main__":
    main()