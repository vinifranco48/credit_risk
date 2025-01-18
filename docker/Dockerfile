FROM python:3.11

WORKDIR /app

# Copiar requirements e instalar dependências primeiro
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copiar todo o código fonte
COPY . .

# Garantir que temos __init__.py em todos os diretórios necessários
RUN touch src/__init__.py

# Configurar PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Modificar o comando para executar via python ao invés de uvicorn diretamente
CMD ["python", "/app/app/main.py"]