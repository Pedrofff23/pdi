FROM python:3.11-slim

# Instala dependências do sistema necessárias para o OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia os arquivos de dependência e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script de segmentação
COPY segmentation.py .

# Define o ponto de entrada para rodar a segmentação
ENTRYPOINT ["python", "segmentation.py"]
