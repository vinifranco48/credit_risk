import streamlit as st
from groq import Groq
import pandas as pd
from datetime import datetime
import time

# Import the loan analysis system (assuming it's in loan_system.py)
from services.agents_service import LoanRiskAnalysisSystem, GROQ_API_KEY, GROQ_MODEL

# Page configuration
st.set_page_config(
    page_title="Loan Risk Analysis System",
    page_icon="💰",
    layout="wide"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'loan_system' not in st.session_state:
    llm_client = Groq(api_key=GROQ_API_KEY)
    st.session_state.loan_system = LoanRiskAnalysisSystem(llm_client)

def analyze_loan(loan_id: int):
    """Process loan analysis and format results for display"""
    with st.spinner('Analyzing loan data...'):
        result = st.session_state.loan_system.process_loan_application(loan_id)
        
        if result['errors']:
            return {
                'type': 'error',
                'content': '\n'.join(result['errors'])
            }
        
        # Format loan data
        loan_data = result['loan_data'].to_dict('records')[0]
        loan_info = '\n'.join([f"**{k}**: {v}" for k, v in loan_data.items()])
        
        # Format risk analysis
        risk_analysis = '\n'.join([f"**{k}**: {v}" for k, v in result['risk_analysis'].items()])
        
        # Format insights
        insights = '\n'.join([f"• {insight}" for insight in result['insights']])
        
        return {
            'type': 'analysis',
            'loan_data': loan_info,
            'risk_analysis': risk_analysis,
            'insights': insights,
            'next_steps': result['next_steps'],
            'report': result['report']
        }

def display_analysis_result(result):
    """Display analysis results in an organized way"""
    if result['type'] == 'error':
        st.error(result['content'])
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📊 Loan Data", expanded=True):
            st.markdown(result['loan_data'])
        
        with st.expander("🎯 Risk Analysis", expanded=True):
            st.markdown(result['risk_analysis'])
    
    with col2:
        with st.expander("💡 Insights", expanded=True):
            st.markdown(result['insights'])
        
        with st.expander("➡️ Next Steps", expanded=True):
            st.markdown(result['next_steps'])
    
    with st.expander("📝 Detailed Report", expanded=True):
        st.markdown(result['report'])

def main():
    # Header
    st.title("🏦 Interactive Loan Risk Analysis System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Enter a loan ID in the input field
        2. Click 'Analyze Loan' to process the application
        3. View the analysis results in the expandable sections
        4. Previous analyses will be saved in the chat history
        """)
        
        st.header("About")
        st.markdown("""
        This system analyzes loan applications and provides:
        - Risk assessment
        - Financial insights
        - Recommended next steps
        - Detailed analysis report
        """)
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "analysis":
                display_analysis_result(message["content"])
    
    # Input area
    loan_id = st.number_input("Enter Loan ID:", min_value=1, step=1)
    
    if st.button("Analyze Loan"):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": f"Requesting analysis for Loan ID: {loan_id}"
        })
        
        # Process the loan
        result = analyze_loan(loan_id)
        
        # Add assistant message with analysis
        st.session_state.messages.append({
            "role": "assistant",
            "type": "analysis",
            "content": result
        })
        
        # Rerun to update the chat
        st.rerun()

if __name__ == "__main__":
    main()