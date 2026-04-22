FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8092

# Single worker by design: cooldown state, active_action_ids, and recovery
# task tracking rely on shared in-memory state within one process.  Scaling
# is done via Kubernetes replicas with leader election for recovery actions.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]
