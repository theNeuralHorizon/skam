#!/usr/bin/env python3
"""Retrain XGBoost+LSTM, Self-Attention, and OCSVM on own cluster data.

Reads the exported Prometheus training data from ml/data/own-logs/training_data.csv
and retrains all three ensemble models, saving updated weights to ml/models/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FEATURES = ["request_rate", "error_rate", "p99_latency", "cpu_usage", "memory_usage"]
OUTPUT_DIR = REPO_ROOT / "ml" / "models"
DATA_PATH = REPO_ROOT / "ml" / "data" / "own-logs" / "training_data.csv"
SEQ_LEN = 20


def train_xgboost(X, y, X_normal):
    """Train XGBoost classifier on labeled point-wise data."""
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    print("\n" + "=" * 60)
    print("  Phase 1: XGBoost (supervised, point-wise)")
    print("=" * 60)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42,
    )

    # Handle 19:1 imbalance
    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"  scale_pos_weight: {pos_weight:.1f}")

    t0 = time.time()
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=pos_weight,
        random_state=42, n_jobs=-1, use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_time = time.time() - t0

    val_proba = model.predict_proba(X_val)[:, 1]
    val_preds = (val_proba >= 0.5).astype(int)
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds, zero_division=0)
    f1 = f1_score(y_val, val_preds, zero_division=0)
    auc = roc_auc_score(y_val, val_proba)

    print(f"  Training time  : {xgb_time:.1f}s")
    print(f"  Val accuracy   : {acc:.4f}")
    print(f"  Val precision  : {prec:.4f}")
    print(f"  Val recall     : {rec:.4f}")
    print(f"  Val F1         : {f1:.4f}")
    print(f"  Val AUC-ROC    : {auc:.4f}")

    print("\n  Feature importance:")
    for i, col in enumerate(FEATURES):
        imp = model.feature_importances_[i]
        bar = "#" * int(imp * 40)
        print(f"    {col:20s}: {imp:.3f} {bar}")

    # Save for XGB+LSTM ensemble
    artefact = {
        "model": model, "scaler": scaler, "feature_order": FEATURES,
        "xgb_weight": 0.5, "lstm_weight": 0.5,
        "training_samples": len(X_train),
        "trained_on": "own-cluster-data", "val_auc_roc": round(auc, 4),
    }
    joblib.dump(artefact, OUTPUT_DIR / "xgboost_lstm.pkl", compress=3)
    print(f"\n  Saved: xgboost_lstm.pkl")

    # Save for XGB+Attention ensemble
    artefact_attn = {
        "model": model, "scaler": scaler, "feature_order": FEATURES,
        "xgb_weight": 0.5, "attention_weight": 0.5,
        "training_samples": len(X_train),
        "trained_on": "own-cluster-data", "val_auc_roc": round(auc, 4),
    }
    joblib.dump(artefact_attn, OUTPUT_DIR / "xgboost_attention.pkl", compress=3)
    print(f"  Saved: xgboost_attention.pkl")

    return {"auc_roc": round(auc, 4), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4)}


def train_attention(df):
    """Train SelfAttentionScorer on labeled temporal sequences."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from ml.training.attention_model import SelfAttentionScorer

    print("\n" + "=" * 60)
    print("  Phase 2: Self-Attention (supervised, temporal)")
    print("=" * 60)

    # Build sequences per service
    seq_X, seq_y = [], []
    for svc in df.service.unique():
        svc_df = df[df.service == svc].sort_values("timestamp")
        svc_features = svc_df[FEATURES].values.astype(np.float32)
        svc_labels = svc_df["label"].values
        for i in range(len(svc_features) - SEQ_LEN):
            seq_X.append(svc_features[i:i + SEQ_LEN])
            seq_y.append(svc_labels[i + SEQ_LEN - 1])

    seq_X = np.array(seq_X)
    seq_y = np.array(seq_y)
    print(f"  Sequences: {len(seq_X)} (normal={(seq_y == 0).sum()}, anomaly={(seq_y == 1).sum()})")

    # Normalize
    flat = seq_X.reshape(-1, len(FEATURES))
    feat_min = flat.min(axis=0)
    feat_max = flat.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    seq_X_norm = (seq_X - feat_min) / feat_range

    # Split
    idx = np.arange(len(seq_X_norm))
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))

    train_ds = TensorDataset(
        torch.FloatTensor(seq_X_norm[idx[:split]]),
        torch.FloatTensor(seq_y[idx[:split]]).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(seq_X_norm[idx[split:]]),
        torch.FloatTensor(seq_y[idx[split:]]).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = SelfAttentionScorer(input_dim=len(FEATURES), d_model=32, num_heads=4)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    t0 = time.time()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 51):
        model.train()
        epoch_loss, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg_train = epoch_loss / n

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/50  train={avg_train:.4f}  val={avg_val:.4f}")

    attn_time = time.time() - t0
    print(f"  Training time: {attn_time:.1f}s")
    print(f"  Best val loss: {best_val_loss:.4f}")

    checkpoint = {
        "model_state_dict": best_state or model.state_dict(),
        "input_dim": len(FEATURES), "d_model": 32, "num_heads": 4,
        "seq_length": SEQ_LEN,
        "feat_min": feat_min.tolist(), "feat_max": feat_max.tolist(),
        "trained_on": "own-cluster-data",
    }
    torch.save(checkpoint, OUTPUT_DIR / "attention_net.pt")
    print(f"  Saved: attention_net.pt")

    return {"best_val_loss": round(best_val_loss, 4), "sequences": len(seq_X)}


def train_ocsvm(X_normal):
    """Train One-Class SVM on normal-only data."""
    import joblib
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import OneClassSVM

    print("\n" + "=" * 60)
    print("  Phase 3: One-Class SVM (unsupervised, normal-only)")
    print("=" * 60)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_normal)

    if len(X_scaled) > 2000:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(X_scaled), 2000, replace=False)
        X_train = X_scaled[sub_idx]
    else:
        X_train = X_scaled

    t0 = time.time()
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    model.fit(X_train)
    elapsed = time.time() - t0

    preds = model.predict(X_train)
    flagged = (preds == -1).sum()
    print(f"  Training samples: {len(X_train)}")
    print(f"  Training time   : {elapsed:.1f}s")
    print(f"  Self-check      : {flagged}/{len(preds)} flagged ({flagged / len(preds) * 100:.1f}%)")

    artefact = {
        "model": model, "scaler": scaler, "feature_order": FEATURES,
        "nu": 0.05, "training_samples": len(X_train),
        "trained_on": "own-cluster-data",
    }
    joblib.dump(artefact, OUTPUT_DIR / "ocsvm.pkl", compress=3)
    print(f"  Saved: ocsvm.pkl")

    return {"training_samples": len(X_train),
            "self_check_pct": round(flagged / len(preds) * 100, 2)}


def main():
    print("=" * 60)
    print("  RETRAINING ON OWN CLUSTER DATA")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values.astype(np.float64)
    y = df["label"].values.astype(np.int32)
    X_normal = X[y == 0]
    X_anomaly = X[y == 1]

    print(f"\n  Dataset: {len(X)} samples ({len(X_normal)} normal, {len(X_anomaly)} anomaly)")
    print(f"  Source : {DATA_PATH}")

    xgb_stats = train_xgboost(X, y, X_normal)
    attn_stats = train_attention(df)
    ocsvm_stats = train_ocsvm(X_normal)

    # Update training summary
    summary_path = OUTPUT_DIR / "training_summary.json"
    summary = json.load(open(summary_path))
    summary["retrained_on"] = "own-cluster-data"
    summary["retrain_dataset"] = str(DATA_PATH)
    summary["retrain_stats"] = {
        "xgboost": xgb_stats,
        "attention": attn_stats,
        "ocsvm": ocsvm_stats,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  RETRAINING COMPLETE")
    print("=" * 60)
    print(f"  XGBoost AUC-ROC  : {xgb_stats['auc_roc']}")
    print(f"  XGBoost F1       : {xgb_stats['f1']}")
    print(f"  Attention val    : {attn_stats['best_val_loss']}")
    print(f"  OCSVM flagged    : {ocsvm_stats['self_check_pct']}%")
    print(f"  Data source      : own-cluster-data")
    print("=" * 60)

    print(f"\n  Models in: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
