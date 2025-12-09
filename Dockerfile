FROM python:3.10-slim

WORKDIR /app

# Install dependencies required for PyTorch + spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "modules/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
