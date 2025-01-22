import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Credit Risk Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Função para criar o gráfico de medidor (gauge)
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Credit Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 580], 'color': 'red'},
                {'range': [580, 670], 'color': 'yellow'},
                {'range': [670, 740], 'color': 'lightgreen'},
                {'range': [740, 850], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "black", 'family': "Arial"}
    )
    return fig


st.title("📊 Credit Risk Prediction App")
st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
        color: #4F4F4F;
    }
    </style>
    <p class="big-font">Enter customer information to predict credit risk.</p>
    """, unsafe_allow_html=True)

# Formulário de entrada
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0, help="Enter the loan amount requested by the customer.")
        term = st.selectbox("Term (months)", [36, 60], help="Select the loan term in months.")
        int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, help="Enter the interest rate for the loan.")
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], help="Select the customer's home ownership status.")
        annual_inc = st.number_input("Annual Income", min_value=0.0, value=50000.0, help="Enter the customer's annual income.")
        verification_status = st.selectbox("Verification Status", ["Verified", "Not Verified", "Source Verified"], help="Select the verification status of the customer's income.")
        purpose = st.selectbox("Purpose", [
            "debt_consolidation", "credit_card", "home_improvement",
            "other", "major_purchase", "small_business", "car"
        ], help="Select the purpose of the loan.")

    with col2:
        addr_state = st.selectbox("State", [
            "CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "MI",
            "NC", "VA", "MD", "AZ", "MA", "WA", "CO", "MN", "IN", "TN"
        ], help="Select the customer's state of residence.")
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=20.0, help="Enter the customer's debt-to-income ratio.")
        inq_last_6mths = st.number_input("Inquiries Last 6 Months", min_value=0, value=1, help="Enter the number of credit inquiries in the last 6 months.")
        mths_since_last_delinq = st.number_input("Months Since Last Delinquency", min_value=0.0, value=36.0, help="Enter the number of months since the last delinquency.")
        open_acc = st.number_input("Open Accounts", min_value=0, value=10, help="Enter the number of open credit accounts.")
        revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=15000.0, help="Enter the total revolving balance.")
        total_acc = st.number_input("Total Accounts", min_value=0.0, value=20.0, help="Enter the total number of credit accounts.")
        initial_list_status = st.selectbox("Initial List Status", ["w", "f"], help="Select the initial list status of the loan.")
        tot_cur_bal = st.number_input("Total Current Balance", min_value=0.0, value=50000.0, help="Enter the total current balance of all accounts.")
        mths_since_earliest_cr_line = st.number_input("Months Since Earliest Credit Line", min_value=0.0, value=120.0, help="Enter the number of months since the earliest credit line was opened.")

    submitted = st.form_submit_button("🚀 Predict Credit Risk")

# Processamento da previsão
if submitted:
    data = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "addr_state": addr_state,
        "dti": dti,
        "inq_last_6mths": inq_last_6mths,
        "mths_since_last_delinq": mths_since_last_delinq,
        "open_acc": open_acc,
        "revol_bal": revol_bal,
        "total_acc": total_acc,
        "initial_list_status": initial_list_status,
        "tot_cur_bal": tot_cur_bal,
        "mths_since_earliest_cr_line": mths_since_earliest_cr_line
    }

    try:
        response = requests.post("http://api_model:8000/predict", json=data)
        response.raise_for_status()
        result = response.json()

        # Exibir resultados
        st.success("✅ Prediction completed successfully!")
        st.header("📈 Prediction Results")

        # Métricas em cards
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Credit Risk", value="High Risk" if result["prediction"] == 1 else "Low Risk")
        with col2:
            st.metric(label="Risk Probability", value=f"{result['probability']:.1%}")

        # Gráfico de medidor
        st.plotly_chart(create_gauge(result['credit_score']), use_container_width=True)

        # Interpretação do score
        score = result['credit_score']
        if score >= 740:
            score_text = "Excellent Credit Score"
            score_color = "green"
        elif score >= 670:
            score_text = "Good Credit Score"
            score_color = "lightgreen"
        elif score >= 580:
            score_text = "Fair Credit Score"
            score_color = "#FFA500"  # Orange
        else:
            score_text = "Poor Credit Score"
            score_color = "red"
        st.markdown(f"<h3 style='text-align: center; color: {score_color};'>{score_text}</h3>", unsafe_allow_html=True)

        # Informações adicionais
        st.subheader("📋 Additional Information")
        st.write(f"Prediction Timestamp: {result['prediction_timestamp']}")
        st.write(f"Storage Path: {result['minio_storage_path']}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error(f"Response content: {e.response.content if e.response else 'No response'}")

# Sidebar aprimorada


