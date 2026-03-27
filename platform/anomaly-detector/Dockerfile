FROM python:3.11-slim

WORKDIR /app

# Install system deps for numpy/torch compilation (if needed by wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Copy pre-trained model weights into the image.
# The Makefile `copy-models` target populates this directory from ml/models/
# before building the Docker image.
COPY models/ /app/models/

EXPOSE 8091

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
