# gen_DFN_3D — Modelagem 3D de Redes de Fraturas Discretas
FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
# (libgl1 substitui libgl1-mesa-glx, que não existe mais nas imagens Debian atuais;
#  curl é necessário para o HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código da aplicação (func_tools.py e assets/ são usados pelo app.py)
COPY app.py func_tools.py ./
COPY modules/ ./modules/
COPY data/ ./data/
COPY assets/ ./assets/

# Configuração do Streamlit
RUN mkdir -p /root/.streamlit && printf '%s\n' \
    '[server]' \
    'headless = true' \
    'port = 8501' \
    'enableCORS = false' \
    '' \
    '[theme]' \
    "base = 'dark'" \
    "primaryColor = '#2c7be5'" \
    > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
