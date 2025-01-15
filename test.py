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

# Configuration
DB_PARAMS = {
    'dbname': 'mlflow',
    'user': 'mlflow',
    'password': 'secret',
    'host': 'localhost',
    'port': '5434'
}

GROQ_API_KEY = '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'

# State Management
class AgentState(TypedDict):
    """Represents the state of our credit risk analysis workflow"""
    messages: Sequence[BaseMessage]
    loan_data: Union[pd.DataFrame, None]
    risk_analysis: Union[pd.DataFrame, None]
    report: Union[pd.DataFrame, None]
    insights: list[str]
    next_step: str
    errors: list[str]

# Tools
class DataTools:
    def __init__(self, engine):
        self.engine = engine
    
    def fetch_loan_data(self) -> pd.DataFrame:
        """Fetches loan data from the database"""
        query = """
        SELECT loan_amnt, term, int_rate, grade, sub_grade, emp_length, home_ownership, 
               annual_inc, verification_status, purpose, addr_state, dti, inq_last_6mths,
               mths_since_last_delinq, open_acc, revol_bal, total_acc, initial_list_status,
               tot_cur_bal, mths_since_earliest_cr_line
        FROM loan_features
        """
        return pd.read_sql(query, self.engine)

    def analyze_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyzes credit risk based on loan data"""
        data = data.copy()
        
        numerical_features = ['int_rate', 'dti', 'inq_last_6mths', 'annual_inc']
        for feature in numerical_features:
            data[f'{feature}_normalized'] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
        
        data['risk_score'] = (
            data['int_rate_normalized'] * 0.35 +
            data['dti_normalized'] * 0.25 +
            data['inq_last_6mths_normalized'] * 0.20 +
            (1 - data['annual_inc_normalized']) * 0.20
        )
        
        data['risk_category'] = pd.qcut(
            data['risk_score'], 
            q=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        return data

    def generate_report(self, analyzed_data: pd.DataFrame) -> pd.DataFrame:
        """Generates a comprehensive report from analyzed data"""
        report = analyzed_data.groupby(['grade', 'risk_category']).agg({
            'risk_score': ['mean', 'std', 'count'],
            'loan_amnt': ['sum', 'mean', 'std'],
            'int_rate': ['mean', 'min', 'max'],
            'dti': ['mean', 'median']
        }).round(3)
        
        report.columns = [f"{col[0]}_{col[1]}" for col in report.columns]
        return report.reset_index()

class InsightGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=200
        )
    
    def generate_insights(self, report: pd.DataFrame) -> list[str]:
        """Generates AI-powered insights from the report"""
        insights = []
        
        for _, row in report.iterrows():
            prompt = f"""Analyze the following lending data for grade {row['grade']} and risk category {row['risk_category']}:
            
            Risk Metrics:
            - Average risk score: {row['risk_score_mean']:.3f}
            - Risk score std dev: {row['risk_score_std']:.3f}
            - Number of loans: {row['risk_score_count']}
            
            Loan Metrics:
            - Total amount: ${row['loan_amnt_sum']:,.2f}
            - Average amount: ${row['loan_amnt_mean']:,.2f}
            - Amount std dev: ${row['loan_amnt_std']:,.2f}
            
            Interest and DTI:
            - Avg interest rate: {row['int_rate_mean']:.2f}%
            - Interest range: {row['int_rate_min']:.2f}% - {row['int_rate_max']:.2f}%
            - Median DTI: {row['dti_median']:.2f}
            
            Provide a concise, data-driven analysis of risks and opportunities for this loan segment.
            Focus on actionable insights and specific recommendations."""

            response = self.llm.invoke(prompt)
            insights.append(f"Grade {row['grade']} ({row['risk_category']}): {response.content}")
        
        return insights

def initialize_state() -> AgentState:
    """Initialize the agent's state"""
    return AgentState(
        messages=[],
        loan_data=None,
        risk_analysis=None,
        report=None,
        insights=[],
        next_step="fetch_data",
        errors=[]
    )

def handle_errors(state: AgentState, error: Exception) -> AgentState:
    """Handle errors in the workflow"""
    state['errors'].append(f"{datetime.now()}: {str(error)}")
    state['next_step'] = "end"
    return state

class CreditRiskAgent:
    def __init__(self):
        # Initialize database connection
        self.engine = create_engine(
            f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@"
            f"{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
        )
        
        # Initialize tool instances
        self.data_tools = DataTools(self.engine)
        self.insight_generator = InsightGenerator()
        
        # Create proper Tool instances
        self.tools = [
            Tool(
                name="fetch_loan_data",
                func=self.data_tools.fetch_loan_data,
                description="Fetches loan data from the database"
            ),
            Tool(
                name="analyze_risk",
                func=self.data_tools.analyze_risk,
                description="Analyzes credit risk based on loan data"
            ),
            Tool(
                name="generate_report",
                func=self.data_tools.generate_report,
                description="Generates a comprehensive report from analyzed data"
            ),
            Tool(
                name="generate_insights",
                func=self.insight_generator.generate_insights,
                description="Generates AI-powered insights from the report"
            )
        ]
        
        # Build workflow graph
        self.workflow = self.create_workflow()

    def create_node(self, tool_name: str):
        """Creates a node function for the given tool"""
        def node_func(state: AgentState) -> AgentState:
            try:
                if tool_name == "fetch_loan_data":
                    state['loan_data'] = self.data_tools.fetch_loan_data()
                    state['next_step'] = "analyze_risk"
                elif tool_name == "analyze_risk" and state['loan_data'] is not None:
                    state['risk_analysis'] = self.data_tools.analyze_risk(state['loan_data'])
                    state['next_step'] = "generate_report"
                elif tool_name == "generate_report" and state['risk_analysis'] is not None:
                    state['report'] = self.data_tools.generate_report(state['risk_analysis'])
                    state['next_step'] = "generate_insights"
                elif tool_name == "generate_insights" and state['report'] is not None:
                    state['insights'] = self.insight_generator.generate_insights(state['report'])
                    state['next_step'] = "end"
            except Exception as e:
                return handle_errors(state, e)
            return state
        return node_func

    def create_workflow(self) -> Graph:
        """Creates the workflow graph"""
        # Create workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("fetch_data", self.create_node("fetch_loan_data"))
        workflow.add_node("analyze_risk", self.create_node("analyze_risk"))
        workflow.add_node("generate_report", self.create_node("generate_report"))
        workflow.add_node("generate_insights", self.create_node("generate_insights"))
        
        # Define end condition
        def end_condition(state: AgentState) -> bool:
            return state["next_step"] == "end"
        
        # Add edges with conditional routing
        workflow.add_edge("fetch_data", "analyze_risk", condition=lambda x: not end_condition(x))
        workflow.add_edge("analyze_risk", "generate_report", condition=lambda x: not end_condition(x))
        workflow.add_edge("generate_report", "generate_insights", condition=lambda x: not end_condition(x))
        
        # Add edges to END
        workflow.add_edge("fetch_data", END, condition=end_condition)
        workflow.add_edge("analyze_risk", END, condition=end_condition)
        workflow.add_edge("generate_report", END, condition=end_condition)
        workflow.add_edge("generate_insights", END, condition=end_condition)
        
        # Set entry point
        workflow.set_entry_point("fetch_data")
        
        return workflow.compile()
    
    def run(self) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Execute the complete credit risk analysis workflow"""
        try:
            # Initialize and run workflow
            state = initialize_state()
            final_state = self.workflow.invoke(state)
            
            # Export report if available
            if final_state['report'] is not None:
                final_state['report'].to_csv('credit_risk_report.csv', index=False)
                print("Report exported to 'credit_risk_report.csv'")
            
            return final_state['report'], final_state['insights'], final_state['errors']
            
        finally:
            self.engine.dispose()

def main():
    print("Starting Credit Risk Analysis...")
    agent = CreditRiskAgent()
    report, insights, errors = agent.run()
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"- {error}")
    
    if report is not None:
        print("\nCredit Risk Analysis Report:")
        print(report)
        
        print("\nAI-Generated Insights:")
        for insight in insights:
            print(f"\n{insight}")
    else:
        print("Failed to generate report.")

if __name__ == "__main__":
    main()