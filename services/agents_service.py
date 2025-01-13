import pandas as pd
from sqlalchemy import create_engine
from groq import Groq

# Configurações do banco de dados (Docker)
db_params = {
    'dbname': 'mlflow',
    'user': 'mlflow',
    'password': 'secret',
    'host': 'localhost',
    'port': '5434'
}

# Configuração da API da Groq
groq_api_key = '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'
client = Groq(api_key=groq_api_key)

class CreditRiskAgent:
    def __init__(self, engine):
        self.engine = engine

    def fetch_data(self):
        """
        Consulta o banco de dados para obter os dados dos empréstimos.
        """
        query = """
        SELECT loan_amnt, term, int_rate, grade, sub_grade, emp_length, home_ownership, annual_inc, 
               verification_status, purpose, addr_state, dti, inq_last_6mths, mths_since_last_delinq, 
               open_acc, revol_bal, total_acc, initial_list_status, tot_cur_bal, mths_since_earliest_cr_line
        FROM loan_features
        """
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"Erro ao buscar dados: {str(e)}")
            return None

    def analyze_risk(self, data):
        """
        Analisa o risco de crédito com base nos dados.
        """
        if data is None or data.empty:
            return None
        
        try:
            # Cálculo mais robusto do score de risco
            data['risk_score'] = (
                data['int_rate'] * 0.4 +  # Taxa de juros tem peso maior
                data['dti'] * 0.3 +       # DTI (Debt-to-Income) tem peso médio
                (data['inq_last_6mths'] / data['inq_last_6mths'].max()) * 0.3  # Consultas normalizadas
            )
            return data
        except Exception as e:
            print(f"Erro na análise de risco: {str(e)}")
            return None

    def generate_report(self, analyzed_data):
        """
        Gera um relatório agregado com base nos dados analisados.
        """
        if analyzed_data is None or analyzed_data.empty:
            return None

        try:
            report = analyzed_data.groupby('grade').agg({
                'risk_score': ['mean', 'std', 'count'],
                'loan_amnt': ['sum', 'mean']
            }).round(2)
            
            # Renomeia as colunas para melhor legibilidade
            report.columns = [
                'risco_medio', 'risco_desvio_padrao', 'quantidade_emprestimos',
                'valor_total_emprestado', 'valor_medio_emprestimo'
            ]
            report = report.reset_index()
            return report
        except Exception as e:
            print(f"Erro na geração do relatório: {str(e)}")
            return None

    def generate_insights(self, report):
        """
        Usa a API da Groq para gerar insights a partir do relatório.
        """
        if report is None or report.empty:
            return []

        insights = []
        try:
            for _, row in report.iterrows():
                prompt = f"""Analise os seguintes dados de empréstimos para a grade {row['grade']}:
                - Score médio de risco: {row['risco_medio']:.2f}
                - Desvio padrão do risco: {row['risco_desvio_padrao']:.2f}
                - Número de empréstimos: {row['quantidade_emprestimos']}
                - Valor total emprestado: R$ {row['valor_total_emprestado']:,.2f}
                - Valor médio por empréstimo: R$ {row['valor_medio_emprestimo']:,.2f}
                
                Forneça uma análise concisa dos riscos e oportunidades para esta grade de empréstimo."""

                chat_completion = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    model="mixtral-8x7b-32768",  # Modelo Mixtral da Groq
                    temperature=0.7,
                    max_tokens=150
                )
                
                insight = chat_completion.choices[0].message.content
                insights.append(f"Grade {row['grade']}: {insight}")
                
        except Exception as e:
            print(f"Erro na geração de insights: {str(e)}")
        
        return insights

    def run(self):
        """
        Executa o fluxo completo do agente.
        """
        # Passo 1: Busca os dados do banco de dados
        print("Buscando dados...")
        data = self.fetch_data()
        if data is None:
            return None, []
        
        # Passo 2: Analisa o risco de crédito
        print("Analisando riscos...")
        analyzed_data = self.analyze_risk(data)
        if analyzed_data is None:
            return None, []
        
        # Passo 3: Gera o relatório
        print("Gerando relatório...")
        report = self.generate_report(analyzed_data)
        if report is None:
            return None, []
        
        # Passo 4: Gera insights com a Groq
        print("Gerando insights...")
        insights = self.generate_insights(report)
        
        # Passo 5: Exporta o relatório
        try:
            report.to_csv('credit_risk_report.csv', index=False)
            print("Relatório exportado com sucesso para 'credit_risk_report.csv'")
        except Exception as e:
            print(f"Erro ao exportar relatório: {str(e)}")
        
        return report, insights

def main():
    try:
        # Cria a conexão com o banco de dados
        engine = create_engine(
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@"
            f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
        # Cria e executa o agente
        agent = CreditRiskAgent(engine)
        report, insights = agent.run()
        
        if report is not None:
            print("\nRelatório de Risco de Crédito:")
            print(report)
            
            print("\nInsights Gerados:")
            for insight in insights:
                print(f"\n{insight}")
        else:
            print("Não foi possível gerar o relatório.")
            
    except Exception as e:
        print(f"Erro na execução do programa: {str(e)}")
    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == "__main__":
    main()