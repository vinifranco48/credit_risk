# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copiar o arquivo de requisitos
COPY ./requirements.txt /app/requirements.txt

# Instalar dependências
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Instalar pytest e httpx
RUN pip install pytest httpx

# Copiar a pasta src para o contêiner
COPY ./src /app/src

# Definir o comando padrão
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
