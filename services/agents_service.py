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
        self.engine = engine
    
    def fetch_loan_data(self) -> pd.DataFrame:

        query ="""
            SELECT f.id, f.loan_amount, f.term, f.int_rate, f.home_ownership, f.annual_inc, f.verification_status,
                    f.purpose, f.dti, f.inq_last_6mths, f.mths_since_last_delinq, f.open_acc, f.revol_bal, f.initial_list_status,
                    f.tot_cur_bal, f.mths_since_earliest_cr_line
            FROM loan_features f




        
        """
        
