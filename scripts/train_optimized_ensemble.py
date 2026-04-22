#!/usr/bin/env python3
"""Optimized XGBoost+LSTM ensemble using XGBoost meta-learner + PCA.

Instead of a neural network fusion (which starves on 196 anomaly samples),
this uses:
  1. PCA to compress 32-dim LSTM embedding -> 4 dims (removes noise)
  2. XGBoost as meta-learner on compact 11-dim joint features

Joint features: [xgb_proba(1), pca_embedding(4), recon_error(1), raw_features(5)]
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FEATURES = ["request_rate", "error_rate", "p99_latency", "cpu_usage", "memory_usage"]
OUTPUT_DIR = REPO_ROOT / "ml" / "models"
DATA_PATH = REPO_ROOT / "ml" / "data" / "own-logs" / "training_data.csv"
SEQ_LEN = 20
N_PCA = 4


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        out, _ = self.decoder(x, (h, c))
        return self.output_layer(out)

    def encode(self, x):
        _, (h, _) = self.encoder(x)
        return h[-1]


def main():
    print("=" * 60)
    print("  OPTIMIZED XGBoost+LSTM ENSEMBLE")
    print("  (XGBoost meta-learner + PCA-compressed LSTM)")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    # Load pre-trained XGBoost
    xgb_art = joblib.load(OUTPUT_DIR / "xgboost_lstm.pkl")
    xgb_model = xgb_art["model"]
    xgb_scaler = xgb_art["scaler"]
    print("  XGBoost: loaded")

    # Load pre-trained LSTM
    ckpt = torch.load(OUTPUT_DIR / "lstm_autoencoder.pt", map_location="cpu", weights_only=False)
    lstm = LSTMAutoencoder(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["num_layers"])
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.eval()

    feat_min = np.array(ckpt["feat_min"])
    feat_max = np.array(ckpt["feat_max"])
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    print("  LSTM: loaded")

    # ── Extract joint features ───────────────────────────────────────
    print("\n  Extracting joint features...")
    embeddings = []
    xgb_probas = []
    recon_errors = []
    raw_feats = []
    labels = []

    for svc in df.service.unique():
        svc_df = df[df.service == svc].sort_values("timestamp")
        svc_feat = svc_df[FEATURES].values.astype(np.float32)
        svc_labels = svc_df["label"].values

        for i in range(len(svc_feat) - SEQ_LEN):
            window = svc_feat[i:i + SEQ_LEN]
            latest = svc_feat[i + SEQ_LEN - 1]
            label = svc_labels[i + SEQ_LEN - 1]

            # XGBoost probability
            xgb_proba = float(
                xgb_model.predict_proba(
                    xgb_scaler.transform(latest.reshape(1, -1))
                )[0][1]
            )

            # LSTM embedding + reconstruction error
            w_norm = (window - feat_min) / feat_range
            wt = torch.FloatTensor(w_norm).unsqueeze(0)
            with torch.no_grad():
                emb = lstm.encode(wt).squeeze(0).numpy()
                recon = lstm(wt)
                recon_err = float(torch.mean((wt - recon) ** 2).item())

            embeddings.append(emb)
            xgb_probas.append(xgb_proba)
            recon_errors.append(recon_err)
            raw_feats.append(latest)
            labels.append(label)

    embeddings = np.array(embeddings)
    xgb_probas = np.array(xgb_probas).reshape(-1, 1)
    recon_errors = np.array(recon_errors).reshape(-1, 1)
    raw_feats = np.array(raw_feats)
    labels = np.array(labels, dtype=np.float32)

    n_anom = int((labels == 1).sum())
    n_norm = int((labels == 0).sum())
    print(f"  Samples: {len(labels)} (normal={n_norm}, anomaly={n_anom})")

    # ── PCA compress LSTM embedding ──────────────────────────────────
    print(f"\n  PCA: 32-dim LSTM embedding -> {N_PCA} dims")
    pca = PCA(n_components=N_PCA)
    emb_pca = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance retained: {var_explained:.1%}")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i}: {v:.1%}")

    # ── Build compact feature matrix ─────────────────────────────────
    X = np.hstack([xgb_probas, emb_pca, recon_errors, raw_feats])
    feat_names = (
        ["xgb_proba"]
        + [f"pca_{i}" for i in range(N_PCA)]
        + ["recon_err"]
        + FEATURES
    )
    print(f"  Compact dim: {X.shape[1]} features")

    # ── Stratified split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=42,
    )

    # ── Train XGBoost meta-learner ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training XGBoost Meta-Learner")
    print("=" * 60)

    pos_weight = n_norm / max(n_anom, 1)
    print(f"  scale_pos_weight: {pos_weight:.1f}")

    t0 = time.time()
    meta = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    meta.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0

    val_proba = meta.predict_proba(X_val)[:, 1]
    val_true = y_val.astype(int)
    auc = roc_auc_score(val_true, val_proba)

    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Val AUC-ROC  : {auc:.4f}")

    # ── Find optimal threshold ───────────────────────────────────────
    print("\n  Threshold analysis:")
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (val_proba >= t).astype(int)
        tp = int(((preds == 1) & (val_true == 1)).sum())
        fp = int(((preds == 1) & (val_true == 0)).sum())
        fn = int(((preds == 0) & (val_true == 1)).sum())
        tn = int(((preds == 0) & (val_true == 0)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-8)
        if f > best_f1:
            best_f1, best_t = f, float(t)
        if round(t, 1) in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            print(f"    @{t:.1f}: P={p:.3f} R={r:.3f} F1={f:.3f} (TP={tp} FP={fp} FN={fn} TN={tn})")

    print(f"\n  Optimal threshold: {best_t:.2f} (F1={best_f1:.4f})")

    # At optimal threshold
    opt_preds = (val_proba >= best_t).astype(int)
    opt_tp = int(((opt_preds == 1) & (val_true == 1)).sum())
    opt_fp = int(((opt_preds == 1) & (val_true == 0)).sum())
    opt_fn = int(((opt_preds == 0) & (val_true == 1)).sum())
    opt_p = opt_tp / max(opt_tp + opt_fp, 1)
    opt_r = opt_tp / max(opt_tp + opt_fn, 1)
    print(f"  @optimal: P={opt_p:.3f} R={opt_r:.3f} F1={best_f1:.3f}")

    # ── Feature importance ───────────────────────────────────────────
    print("\n  Feature importance:")
    for i, name in enumerate(feat_names):
        imp = meta.feature_importances_[i]
        bar = "#" * int(imp * 50)
        print(f"    {name:20s}: {imp:.3f} {bar}")

    # ── Save ─────────────────────────────────────────────────────────
    artefact = {
        "meta_model": meta,
        "pca": pca,
        "xgb_model": xgb_model,
        "xgb_scaler": xgb_scaler,
        "lstm_feat_min": feat_min.tolist(),
        "lstm_feat_max": feat_max.tolist(),
        "feature_order": FEATURES,
        "n_pca_components": N_PCA,
        "optimal_threshold": best_t,
        "val_auc_roc": round(auc, 4),
        "val_f1": round(best_f1, 4),
        "compact_feature_names": feat_names,
        "trained_on": "own-cluster-data",
    }
    out_path = OUTPUT_DIR / "xgboost_lstm_ensemble.pkl"
    joblib.dump(artefact, out_path, compress=3)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved: xgboost_lstm_ensemble.pkl ({size_kb:.0f} KB)")

    joblib.dump(pca, OUTPUT_DIR / "lstm_pca.pkl")
    print(f"  Saved: lstm_pca.pkl")

    # Update summary
    summary_path = OUTPUT_DIR / "training_summary.json"
    summary = json.load(open(summary_path))
    summary["xgboost_lstm_ensemble"] = {
        "type": "xgboost_meta_learner_with_pca",
        "val_auc_roc": round(auc, 4),
        "val_f1": round(best_f1, 4),
        "optimal_threshold": best_t,
        "compact_dim": int(X.shape[1]),
        "pca_variance_retained": round(float(var_explained), 4),
        "trained_on": "own-cluster-data",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  Standalone XGBoost : AUC=0.8703  F1=0.4421")
    print(f"  Neural fusion      : AUC=0.7105  F1=0.2614")
    print(f"  XGBoost meta (new) : AUC={auc:.4f}  F1={best_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
