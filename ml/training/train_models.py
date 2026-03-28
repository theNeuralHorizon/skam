#!/usr/bin/env python3
"""Pre-train anomaly detection models on the RCAEval Online Boutique dataset.

Models trained:
  1. Isolation Forest        (unsupervised, point-wise)
  2. LSTM Autoencoder        (unsupervised, temporal sequences)
  3. XGBoost + LSTM          (supervised, point-wise XGB + LSTM reconstruction)
  4. XGBoost + Attention     (supervised, point-wise XGB + self-attention temporal)
  5. One-Class SVM           (unsupervised, point-wise)

Usage::

    python -m ml.training.train_models                 # from repo root
    python ml/training/train_models.py                  # direct invocation
    python ml/training/train_models.py --data-dir ml/data/rcaeval/online-boutique
    python ml/training/train_models.py --epochs 30 --seq-length 20

Outputs (saved to ``ml/models/``)::

    isolation_forest.pkl    – scikit-learn IsolationForest + scaler
    lstm_autoencoder.pt     – PyTorch LSTM Autoencoder state dict
    xgboost_lstm.pkl        – XGBoost classifier + scaler (for XGB+LSTM ensemble)
    xgboost_attention.pkl   – XGBoost classifier + scaler (for XGB+Attention ensemble)
    attention_net.pt        – PyTorch Self-Attention scorer state dict
    ocsvm.pkl               – One-Class SVM + scaler
    training_stats.json     – feature statistics for normalisation at inference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is importable when run as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.training.data_loader import (
    FEATURE_ORDER,
    build_sequence_arrays,
    build_training_arrays,
)


# ═══════════════════════════════════════════════════════════════════════
# Isolation Forest
# ═══════════════════════════════════════════════════════════════════════

def train_isolation_forest(
    X_normal: np.ndarray,
    output_dir: Path,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """Train an IsolationForest on **normal-only** data and persist it.

    The model is trained only on normal samples so that it learns the
    boundary of "healthy" behaviour.  Anomalous samples are used only
    for validation / threshold tuning.
    """
    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    print("\n══════════════════════════════════════")
    print("  Training Isolation Forest")
    print("══════════════════════════════════════")

    # ── Robust scaling (median / IQR — less sensitive to outliers) ────
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_normal)

    print(f"  Training samples : {len(X_scaled):,}")
    print(f"  Features         : {X_scaled.shape[1]}")
    print(f"  Contamination    : {contamination}")
    print(f"  Estimators       : {n_estimators}")

    # ── Fit ───────────────────────────────────────────────────────────
    t0 = time.time()
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    elapsed = time.time() - t0
    print(f"  Training time    : {elapsed:.1f}s")

    # ── Quick self-check on training data ─────────────────────────────
    scores = model.score_samples(X_scaled)
    preds = model.predict(X_scaled)
    n_anomaly = (preds == -1).sum()
    print(f"  Self-check       : {n_anomaly}/{len(preds)} flagged ({n_anomaly/len(preds)*100:.1f}%)")

    # ── Persist ───────────────────────────────────────────────────────
    artefact = {
        "model": model,
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "training_samples": len(X_scaled),
    }
    out_path = output_dir / "isolation_forest.pkl"
    joblib.dump(artefact, out_path, compress=3)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved            : {out_path} ({size_mb:.1f} MB)")

    return {
        "training_samples": len(X_scaled),
        "training_time_s": round(elapsed, 2),
        "self_check_anomaly_pct": round(n_anomaly / len(preds) * 100, 2),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }


# ═══════════════════════════════════════════════════════════════════════
# LSTM Autoencoder
# ═══════════════════════════════════════════════════════════════════════

def train_lstm_autoencoder(
    X_normal_seq: np.ndarray,
    X_anomalous_seq: np.ndarray,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    num_layers: int = 2,
) -> dict:
    """Train an LSTM Autoencoder on normal sequences and persist it.

    Architecture
    ------------
    Encoder:  LSTM(input=5, hidden=32, layers=2)
    Decoder:  LSTM(input=5, hidden=32, layers=2) + Linear(32 → 5)

    The model learns to reconstruct normal time-series windows.
    Anomalous windows produce higher reconstruction error.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("\n══════════════════════════════════════")
    print("  Training LSTM Autoencoder")
    print("══════════════════════════════════════")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device           : {device}")

    input_dim = X_normal_seq.shape[2]  # 5 features
    seq_length = X_normal_seq.shape[1]

    # ── Normalisation (per-feature min-max over training set) ─────────
    flat = X_normal_seq.reshape(-1, input_dim)
    feat_min = flat.min(axis=0)
    feat_max = flat.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0  # avoid div-by-zero

    def normalise(arr: np.ndarray) -> np.ndarray:
        return (arr - feat_min) / feat_range

    X_train = normalise(X_normal_seq)
    X_val_anom = normalise(X_anomalous_seq) if len(X_anomalous_seq) > 0 else None

    # ── Model definition ──────────────────────────────────────────────
    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
            super().__init__()
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )
            self.decoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )
            self.output_layer = nn.Linear(hidden_dim, input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encode
            _, (hidden, cell) = self.encoder(x)

            # Decode: use encoder's final state, feed original sequence
            decoder_out, _ = self.decoder(x, (hidden, cell))

            # Project back to input dimension
            reconstruction = self.output_layer(decoder_out)
            return reconstruction

    model = LSTMAutoencoder(input_dim, hidden_dim, num_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters       : {total_params:,}")
    print(f"  Sequence length  : {seq_length}")
    print(f"  Hidden dim       : {hidden_dim}")
    print(f"  Normal sequences : {len(X_train):,}")
    if X_val_anom is not None:
        print(f"  Anomaly sequences: {len(X_val_anom):,}")

    # ── Training ──────────────────────────────────────────────────────
    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_anom_loss": []}
    best_loss = float("inf")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)

        # Validation on anomalous sequences (should have HIGHER loss)
        anom_loss = 0.0
        if X_val_anom is not None and len(X_val_anom) > 0:
            model.eval()
            with torch.no_grad():
                anom_tensor = torch.FloatTensor(X_val_anom).to(device)
                anom_recon = model(anom_tensor)
                anom_loss = criterion(anom_recon, anom_tensor).item()
            history["val_anom_loss"].append(anom_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 5 == 0 or epoch == 1:
            sep_ratio = anom_loss / avg_loss if avg_loss > 0 else 0
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={avg_loss:.6f}  "
                f"anom_loss={anom_loss:.6f}  "
                f"separation={sep_ratio:.2f}x"
            )

    elapsed = time.time() - t0
    print(f"  Training time    : {elapsed:.1f}s")
    print(f"  Best train loss  : {best_loss:.6f}")

    # ── Compute reconstruction error threshold ────────────────────────
    model.eval()
    with torch.no_grad():
        train_tensor = torch.FloatTensor(X_train).to(device)
        train_recon = model(train_tensor)
        per_sample_mse = torch.mean((train_tensor - train_recon) ** 2, dim=(1, 2))
        threshold_mean = per_sample_mse.mean().item()
        threshold_std = per_sample_mse.std().item()
        threshold = threshold_mean + 3 * threshold_std  # 3-sigma

    print(f"  Recon error mean : {threshold_mean:.6f}")
    print(f"  Recon error std  : {threshold_std:.6f}")
    print(f"  Anomaly threshold: {threshold:.6f} (mean + 3σ)")

    # ── Persist ───────────────────────────────────────────────────────
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "seq_length": seq_length,
        "feat_min": feat_min.tolist(),
        "feat_max": feat_max.tolist(),
        "threshold_mean": threshold_mean,
        "threshold_std": threshold_std,
        "threshold": threshold,
        "training_samples": len(X_train),
        "epochs": epochs,
        "best_loss": best_loss,
    }

    out_path = output_dir / "lstm_autoencoder.pt"
    torch.save(checkpoint, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved            : {out_path} ({size_mb:.1f} MB)")

    return {
        "training_samples": len(X_train),
        "training_time_s": round(elapsed, 2),
        "best_train_loss": round(best_loss, 6),
        "threshold": round(threshold, 6),
        "separation_ratio": round(
            history["val_anom_loss"][-1] / history["train_loss"][-1], 2
        ) if history["val_anom_loss"] else 0,
        "total_params": total_params,
    }


# ═══════════════════════════════════════════════════════════════════════
# Feature statistics (for inference-time normalisation)
# ═══════════════════════════════════════════════════════════════════════

def save_training_stats(
    X_normal: np.ndarray,
    output_dir: Path,
) -> None:
    """Persist per-feature statistics so the live detector can normalise
    incoming Prometheus metrics the same way."""
    stats = {}
    for i, name in enumerate(FEATURE_ORDER):
        col = X_normal[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "p25": float(np.percentile(col, 25)),
            "p50": float(np.percentile(col, 50)),
            "p75": float(np.percentile(col, 75)),
            "p99": float(np.percentile(col, 99)),
        }

    out_path = output_dir / "training_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Feature stats saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# XGBoost + LSTM Ensemble (supervised)
# ═══════════════════════════════════════════════════════════════════════

def train_xgboost_lstm(
    X_normal: np.ndarray,
    X_anomalous: np.ndarray,
    output_dir: Path,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Train an XGBoost classifier on labeled point-wise data.

    The XGBoost component scores point anomalies (supervised).  At
    inference time, its probability output is combined 50/50 with the
    LSTM autoencoder reconstruction error to form the XGBoost+LSTM
    ensemble.
    """
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBClassifier

    print("\n══════════════════════════════════════")
    print("  Training XGBoost (for XGB+LSTM)")
    print("══════════════════════════════════════")

    # Build labeled dataset: normal=0, anomaly=1
    X = np.vstack([X_normal, X_anomalous])
    y = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomalous))])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=random_state,
    )

    print(f"  Total samples    : {len(X):,} ({len(X_normal):,} normal + {len(X_anomalous):,} anomaly)")
    print(f"  Train / Val      : {len(X_train):,} / {len(X_val):,}")
    print(f"  Estimators       : {n_estimators}")
    print(f"  Max depth        : {max_depth}")

    t0 = time.time()
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  Training time    : {elapsed:.1f}s")

    # Validation accuracy
    val_proba = model.predict_proba(X_val)[:, 1]
    val_preds = (val_proba >= 0.5).astype(int)
    accuracy = (val_preds == y_val).mean()
    print(f"  Val accuracy     : {accuracy:.4f}")

    # Persist
    artefact = {
        "model": model,
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "xgb_weight": 0.5,
        "lstm_weight": 0.5,
        "training_samples": len(X_train),
    }
    out_path = output_dir / "xgboost_lstm.pkl"
    joblib.dump(artefact, out_path, compress=3)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved            : {out_path} ({size_mb:.1f} MB)")

    return {
        "training_samples": len(X_train),
        "training_time_s": round(elapsed, 2),
        "val_accuracy": round(accuracy, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# XGBoost + Attention Ensemble (supervised)
# ═══════════════════════════════════════════════════════════════════════

def train_xgboost_attention(
    X_normal: np.ndarray,
    X_anomalous: np.ndarray,
    X_normal_seq: np.ndarray,
    X_anomalous_seq: np.ndarray,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> dict:
    """Train XGBoost + Self-Attention temporal scorer.

    XGBoost: same as XGB+LSTM (supervised point-wise classifier).
    Attention: ``SelfAttentionScorer`` trained as binary classifier on
    labeled temporal sequences.
    """
    import joblib
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from torch.utils.data import DataLoader, TensorDataset
    from xgboost import XGBClassifier

    from ml.training.attention_model import SelfAttentionScorer

    print("\n══════════════════════════════════════")
    print("  Training XGBoost + Self-Attention")
    print("══════════════════════════════════════")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── XGBoost component (point-wise, supervised) ─────────────────────
    X_pw = np.vstack([X_normal, X_anomalous])
    y_pw = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomalous))])

    scaler = RobustScaler()
    X_pw_scaled = scaler.fit_transform(X_pw)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_pw_scaled, y_pw, test_size=0.2, stratify=y_pw, random_state=random_state,
    )

    print(f"  XGBoost samples  : {len(X_pw):,}")

    t0 = time.time()
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss",
        random_state=random_state, n_jobs=-1, use_label_encoder=False,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_time = time.time() - t0
    print(f"  XGBoost time     : {xgb_time:.1f}s")

    # Persist XGBoost component
    xgb_artefact = {
        "model": xgb_model,
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "xgb_weight": 0.5,
        "attention_weight": 0.5,
        "training_samples": len(X_tr),
    }
    xgb_path = output_dir / "xgboost_attention.pkl"
    joblib.dump(xgb_artefact, xgb_path, compress=3)
    print(f"  XGBoost saved    : {xgb_path}")

    # ── Attention component (temporal, supervised) ─────────────────────
    input_dim = X_normal_seq.shape[2]
    seq_length = X_normal_seq.shape[1]

    # Min-max normalise sequences
    flat = X_normal_seq.reshape(-1, input_dim)
    feat_min = flat.min(axis=0)
    feat_max = flat.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0

    def norm_seq(arr: np.ndarray) -> np.ndarray:
        return (arr - feat_min) / feat_range

    X_seq = np.vstack([norm_seq(X_normal_seq), norm_seq(X_anomalous_seq)])
    y_seq = np.concatenate([
        np.zeros(len(X_normal_seq)),
        np.ones(len(X_anomalous_seq)),
    ])

    # Stratified split
    idx = np.arange(len(X_seq))
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = TensorDataset(
        torch.FloatTensor(X_seq[train_idx]),
        torch.FloatTensor(y_seq[train_idx]).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_seq[val_idx]),
        torch.FloatTensor(y_seq[val_idx]).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"  Attention seqs   : {len(X_seq):,} ({len(train_idx)} train / {len(val_idx)} val)")
    print(f"  Device           : {device}")

    attn_model = SelfAttentionScorer(input_dim=input_dim, d_model=32, num_heads=4).to(device)
    total_params = sum(p.numel() for p in attn_model.parameters())
    print(f"  Attention params : {total_params:,}")

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(attn_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    t0 = time.time()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        attn_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = attn_model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attn_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Validation
        attn_model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = attn_model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                val_batches += 1
        avg_val = val_loss / max(val_batches, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in attn_model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train={avg_loss:.4f}  val={avg_val:.4f}")

    attn_time = time.time() - t0
    print(f"  Attention time   : {attn_time:.1f}s")
    print(f"  Best val loss    : {best_val_loss:.4f}")

    # Persist attention model
    checkpoint = {
        "model_state_dict": best_state or attn_model.state_dict(),
        "input_dim": input_dim,
        "d_model": 32,
        "num_heads": 4,
        "seq_length": seq_length,
        "feat_min": feat_min.tolist(),
        "feat_max": feat_max.tolist(),
        "total_params": total_params,
    }
    attn_path = output_dir / "attention_net.pt"
    torch.save(checkpoint, attn_path)
    size_kb = attn_path.stat().st_size / 1024
    print(f"  Attention saved  : {attn_path} ({size_kb:.0f} KB)")

    return {
        "xgb_training_samples": len(X_tr),
        "xgb_training_time_s": round(xgb_time, 2),
        "attention_sequences": len(X_seq),
        "attention_training_time_s": round(attn_time, 2),
        "attention_best_val_loss": round(best_val_loss, 4),
        "attention_params": total_params,
    }


# ═══════════════════════════════════════════════════════════════════════
# One-Class SVM (unsupervised)
# ═══════════════════════════════════════════════════════════════════════

def train_ocsvm(
    X_normal: np.ndarray,
    output_dir: Path,
    nu: float = 0.05,
    max_samples: int = 2000,
    random_state: int = 42,
) -> dict:
    """Train a One-Class SVM on normal-only data.

    Uses RBF kernel; subsamples to ``max_samples`` since OCSVM is
    O(n^2) in memory.
    """
    import joblib
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import OneClassSVM

    print("\n══════════════════════════════════════")
    print("  Training One-Class SVM")
    print("══════════════════════════════════════")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_normal)

    # Subsample if needed
    if len(X_scaled) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_scaled), max_samples, replace=False)
        X_train = X_scaled[idx]
        print(f"  Subsampled       : {len(X_normal):,} → {max_samples:,}")
    else:
        X_train = X_scaled

    print(f"  Training samples : {len(X_train):,}")
    print(f"  Kernel           : rbf")
    print(f"  Nu               : {nu}")

    t0 = time.time()
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=nu)
    model.fit(X_train)
    elapsed = time.time() - t0
    print(f"  Training time    : {elapsed:.1f}s")

    # Quick self-check
    preds = model.predict(X_train)
    n_anomaly = (preds == -1).sum()
    print(f"  Self-check       : {n_anomaly}/{len(preds)} flagged ({n_anomaly/len(preds)*100:.1f}%)")

    # Persist
    artefact = {
        "model": model,
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "nu": nu,
        "training_samples": len(X_train),
    }
    out_path = output_dir / "ocsvm.pkl"
    joblib.dump(artefact, out_path, compress=3)
    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved            : {out_path} ({size_kb:.0f} KB)")

    return {
        "training_samples": len(X_train),
        "training_time_s": round(elapsed, 2),
        "self_check_anomaly_pct": round(n_anomaly / len(preds) * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-train anomaly detection models on RCAEval data",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "ml" / "data" / "rcaeval" / "online-boutique"),
        help="Path to the extracted online-boutique dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "ml" / "models"),
        help="Directory for saved model artefacts",
    )
    parser.add_argument("--epochs", type=int, default=50, help="LSTM training epochs")
    parser.add_argument("--seq-length", type=int, default=20, help="LSTM sequence window")
    parser.add_argument("--batch-size", type=int, default=64, help="LSTM batch size")
    parser.add_argument("--interval", type=int, default=15, help="Downsample interval (seconds)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════╗")
    print("║  SKAM — Pre-training Anomaly Detection Models   ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\n  Dataset : {data_dir}")
    print(f"  Output  : {output_dir}")

    # ── Phase 1: Load point-wise data (for Isolation Forest) ──────────
    print("\n── Loading point-wise training data ──")
    X_normal, X_anomalous, labels = build_training_arrays(
        data_dir,
        interval_seconds=args.interval,
    )
    print(f"  Normal samples   : {len(X_normal):,}")
    print(f"  Anomalous samples: {len(X_anomalous):,}")

    # ── Phase 2: Train Isolation Forest ───────────────────────────────
    if_stats = train_isolation_forest(X_normal, output_dir)

    # ── Phase 3: Save feature statistics ──────────────────────────────
    save_training_stats(X_normal, output_dir)

    # ── Phase 4: Load sequence data (for LSTM) ────────────────────────
    print("\n── Loading sequence training data ──")
    X_normal_seq, X_anomalous_seq, seq_labels = build_sequence_arrays(
        data_dir,
        seq_length=args.seq_length,
        interval_seconds=args.interval,
    )
    print(f"  Normal sequences : {len(X_normal_seq):,}")
    print(f"  Anomaly sequences: {len(X_anomalous_seq):,}")

    # ── Phase 5: Train LSTM Autoencoder ───────────────────────────────
    lstm_stats = train_lstm_autoencoder(
        X_normal_seq,
        X_anomalous_seq,
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ── Phase 6: Train XGBoost + LSTM ──────────────────────────────
    xgb_lstm_stats = train_xgboost_lstm(X_normal, X_anomalous, output_dir)

    # ── Phase 7: Train XGBoost + Attention ─────────────────────────
    xgb_attn_stats = train_xgboost_attention(
        X_normal, X_anomalous,
        X_normal_seq, X_anomalous_seq,
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ── Phase 8: Train One-Class SVM ───────────────────────────────
    ocsvm_stats = train_ocsvm(X_normal, output_dir)

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "dataset": str(data_dir),
        "interval_seconds": args.interval,
        "isolation_forest": if_stats,
        "lstm_autoencoder": lstm_stats,
        "xgboost_lstm": xgb_lstm_stats,
        "xgboost_attention": xgb_attn_stats,
        "ocsvm": ocsvm_stats,
        "feature_order": FEATURE_ORDER,
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Training Complete                               ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  IF samples       : {if_stats['training_samples']:>8,}               ║")
    print(f"║  IF train time    : {if_stats['training_time_s']:>7.1f}s               ║")
    print(f"║  LSTM sequences   : {lstm_stats['training_samples']:>8,}               ║")
    print(f"║  LSTM train time  : {lstm_stats['training_time_s']:>7.1f}s               ║")
    print(f"║  LSTM separation  : {lstm_stats['separation_ratio']:>7.2f}x               ║")
    print(f"║  XGB+LSTM acc     : {xgb_lstm_stats['val_accuracy']:>7.4f}                ║")
    print(f"║  XGB+LSTM time    : {xgb_lstm_stats['training_time_s']:>7.1f}s               ║")
    print(f"║  XGB+Attn val     : {xgb_attn_stats['attention_best_val_loss']:>7.4f}                ║")
    print(f"║  OCSVM samples    : {ocsvm_stats['training_samples']:>8,}               ║")
    print(f"║  OCSVM time       : {ocsvm_stats['training_time_s']:>7.1f}s               ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\n  Artefacts in: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
