# ── Stage 1: Incremental build on top of existing image ─────────────
# The existing image already has torch, scikit-learn, numpy, pandas.
# We only install the NEW dependencies (xgboost ~20MB, joblib ~300KB).
FROM localhost:5111/skam/anomaly-detector:latest AS base

WORKDIR /app

# Only install what's new — avoids re-downloading torch (906MB)
RUN pip install --no-cache-dir --timeout=120 --retries=3 xgboost==2.1.0 joblib==1.4.2

# Overwrite app code with updated detectors
COPY app/ ./app/

# Overwrite models with new ensemble weights
COPY models/ /app/models/

# Model paths for all 6 detectors
ENV IF_MODEL_PATH=/app/models/isolation_forest.pkl
ENV LSTM_MODEL_PATH=/app/models/lstm_autoencoder.pt
ENV XGB_MODEL_PATH=/app/models/xgboost_lstm.pkl
ENV ATTENTION_MODEL_PATH=/app/models/attention_net.pt
ENV XGB_ATTENTION_MODEL_PATH=/app/models/xgboost_attention.pkl
ENV XGB_META_MODEL_PATH=/app/models/xgboost_lstm_ensemble.pkl

EXPOSE 8091

# Single worker by design: the detector holds in-memory model state, sliding
# windows, and history buffers that are not safe to share across processes.
# Scale via horizontal pod replicas with service-level partitioning instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
