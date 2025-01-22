import streamlit as st
from groq import Groq
import pandas as pd
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from services.agents_service import ConversationalAnalyzer, GROQ_API_KEY, GROQ_MODEL, DataTools

# Page configuration
st.set_page_config(
    page_title="Loan Analysis System",
    page_icon="💰",
    layout="wide"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = ConversationalAnalyzer(Groq(api_key=GROQ_API_KEY))
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

def load_initial_data():
    """Load initial data if not already loaded"""
    if st.session_state.current_data is None:
        try:
            # Corrigido: fetch_loans -> fetch_loan_data
            st.session_state.current_data = st.session_state.analyzer.data_tools.fetch_loan_data()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

def process_user_input(user_input: str):
    """Process user input and generate response"""
    if st.session_state.current_data is None:
        load_initial_data()
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Generate response based on user input
    if "tendência" in user_input.lower() or "tendencias" in user_input.lower():
        field = next((f for f in st.session_state.current_data.columns 
                     if f in user_input.lower()), None)
        if field:
            response = st.session_state.analyzer.analyze_trends(
                st.session_state.current_data, field
            )
        else:
            campos = "\n".join([f"- {col}" for col in st.session_state.current_data.columns])
            response = f"🤔 Qual campo você gostaria de analisar as tendências? Você pode perguntar sobre:\n{campos}"
    else:
        response = st.session_state.analyzer.answer_query(
            user_input, st.session_state.current_data
        )
    
    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

def main():
    # Header
    st.title("💬 Análise Conversacional de Empréstimos")
    
    # Load initial data
    load_initial_data()
    
    # Main chat area
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if not st.session_state.messages:
            st.info("""
            Olá! Sou seu assistente de análise de empréstimos.
            
            Você pode me perguntar sobre:
            - Tendências nos dados
            - Análise de campos específicos
            - Padrões e correlações
            - Insights sobre a carteira
            
            Por exemplo:
            - "Como está a distribuição de credit_score?"
            - "Qual o perfil típico dos empréstimos?"
            - "Existe relação entre renda e inadimplência?"
            """)
    
    # Input area
    user_input = st.chat_input("Pergunte sobre a base de empréstimos...")
    if user_input:
        process_user_input(user_input)
        st.rerun()

if __name__ == "__main__":
    main()