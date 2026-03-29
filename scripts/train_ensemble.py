#!/usr/bin/env python3
"""End-to-end XGBoost+LSTM ensemble training.

Combines XGBoost probability + LSTM encoder embeddings + LSTM reconstruction
error + raw features into a joint 39-dim feature vector, then trains a fusion
neural network that learns the optimal combination.

Architecture:
    XGBoost proba (1) + LSTM embedding (32) + recon error (1) + raw features (5)
    = 39 dims -> Linear(64) -> ReLU -> BN -> Dropout
              -> Linear(32) -> ReLU -> BN -> Dropout
              -> Linear(16) -> ReLU -> Linear(1) -> Sigmoid
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
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FEATURES = ["request_rate", "error_rate", "p99_latency", "cpu_usage", "memory_usage"]
OUTPUT_DIR = REPO_ROOT / "ml" / "models"
DATA_PATH = REPO_ROOT / "ml" / "data" / "own-logs" / "training_data.csv"
SEQ_LEN = 20


# ── LSTM Autoencoder (must match trained architecture) ───────────────

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
        _, (hidden, cell) = self.encoder(x)
        decoder_out, _ = self.decoder(x, (hidden, cell))
        return self.output_layer(decoder_out)

    def encode(self, x):
        """Return encoder final hidden state as embedding."""
        _, (hidden, _) = self.encoder(x)
        return hidden[-1]  # (batch, hidden_dim)


# ── Fusion Network ───────────────────────────────────────────────────

class XGBoostLSTMFusionNet(nn.Module):
    """End-to-end fusion of XGBoost + LSTM signals.

    Input:  [xgb_proba(1), lstm_embedding(32), recon_error(1), raw_features(5)]
    Output: anomaly probability in [0, 1]
    """

    def __init__(self, input_dim=39):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def build_joint_features(df, xgb_model, xgb_scaler, lstm_model, feat_min, feat_range):
    """Extract joint features from both models for each sliding window."""
    joint_features = []
    joint_labels = []

    for svc in df.service.unique():
        svc_df = df[df.service == svc].sort_values("timestamp")
        svc_features = svc_df[FEATURES].values.astype(np.float32)
        svc_labels = svc_df["label"].values

        for i in range(len(svc_features) - SEQ_LEN):
            window = svc_features[i:i + SEQ_LEN]
            latest_point = svc_features[i + SEQ_LEN - 1]
            label = svc_labels[i + SEQ_LEN - 1]

            # XGBoost: anomaly probability
            xgb_scaled = xgb_scaler.transform(latest_point.reshape(1, -1))
            xgb_proba = float(xgb_model.predict_proba(xgb_scaled)[0][1])

            # LSTM: encoder embedding + reconstruction error
            window_norm = (window - feat_min) / feat_range
            window_tensor = torch.FloatTensor(window_norm).unsqueeze(0)

            with torch.no_grad():
                lstm_embedding = lstm_model.encode(window_tensor).squeeze(0).numpy()
                recon = lstm_model(window_tensor)
                recon_error = float(torch.mean((window_tensor - recon) ** 2).item())

            # Joint vector: 1 + 32 + 1 + 5 = 39 dims
            joint_vec = np.concatenate([
                [xgb_proba],
                lstm_embedding,
                [recon_error],
                latest_point,
            ])
            joint_features.append(joint_vec)
            joint_labels.append(label)

    return np.array(joint_features, dtype=np.float32), np.array(joint_labels, dtype=np.float32)


def main():
    print("=" * 60)
    print("  END-TO-END XGBoost+LSTM ENSEMBLE TRAINING")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\n  Dataset: {len(df)} rows from {DATA_PATH.name}")

    # ── Load pre-trained components ──────────────────────────────────
    print("\n  Loading pre-trained components...")

    xgb_artefact = joblib.load(OUTPUT_DIR / "xgboost_lstm.pkl")
    xgb_model = xgb_artefact["model"]
    xgb_scaler = xgb_artefact["scaler"]
    print(f"    XGBoost: loaded")

    lstm_ckpt = torch.load(OUTPUT_DIR / "lstm_autoencoder.pt", map_location="cpu", weights_only=False)
    lstm_model = LSTMAutoencoder(
        lstm_ckpt["input_dim"], lstm_ckpt["hidden_dim"], lstm_ckpt["num_layers"],
    )
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
    lstm_model.eval()

    feat_min = np.array(lstm_ckpt["feat_min"])
    feat_max = np.array(lstm_ckpt["feat_max"])
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    print(f"    LSTM: loaded (hidden={lstm_ckpt['hidden_dim']})")

    # ── Build joint features ─────────────────────────────────────────
    print("\n  Building joint feature vectors...")
    X_joint, y_joint = build_joint_features(
        df, xgb_model, xgb_scaler, lstm_model, feat_min, feat_range,
    )

    n_normal = int((y_joint == 0).sum())
    n_anomaly = int((y_joint == 1).sum())
    print(f"  Joint feature dim: {X_joint.shape[1]}")
    print(f"  Samples: {len(X_joint)} (normal={n_normal}, anomaly={n_anomaly})")

    # ── Stratified split ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
    anom_idx = np.where(y_joint == 1)[0]
    norm_idx = np.where(y_joint == 0)[0]
    rng.shuffle(anom_idx)
    rng.shuffle(norm_idx)

    anom_split = int(0.8 * len(anom_idx))
    norm_split = int(0.8 * len(norm_idx))

    train_idx = np.concatenate([anom_idx[:anom_split], norm_idx[:norm_split]])
    val_idx = np.concatenate([anom_idx[anom_split:], norm_idx[norm_split:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    pos_weight = n_normal / max(n_anomaly, 1)
    print(f"  Train/Val: {len(train_idx)}/{len(val_idx)}")
    print(f"  pos_weight: {pos_weight:.1f}")

    train_ds = TensorDataset(
        torch.FloatTensor(X_joint[train_idx]),
        torch.FloatTensor(y_joint[train_idx]).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_joint[val_idx]),
        torch.FloatTensor(y_joint[val_idx]).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # ── Train fusion network ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Fusion Network")
    print("=" * 60)

    fusion = XGBoostLSTMFusionNet(input_dim=X_joint.shape[1])
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"  Params: {total_params:,}")
    print(f"  Architecture: {X_joint.shape[1]} -> 64 -> 32 -> 16 -> 1")

    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7,
    )

    t0 = time.time()
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, 101):
        fusion.train()
        epoch_loss, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = fusion(xb)
            loss = criterion(pred, yb)
            weights = torch.where(yb == 1, pos_weight, 1.0)
            loss = (loss * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg_train = epoch_loss / n

        # Validation
        fusion.eval()
        val_preds_all, val_labels_all = [], []
        val_loss_sum, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = fusion(xb)
                loss = criterion(pred, yb)
                w = torch.where(yb == 1, pos_weight, 1.0)
                val_loss_sum += (loss * w).mean().item()
                vn += 1
                val_preds_all.extend(pred.squeeze().numpy())
                val_labels_all.extend(yb.squeeze().numpy())

        avg_val = val_loss_sum / max(vn, 1)
        val_preds_arr = np.array(val_preds_all)
        val_labels_arr = np.array(val_labels_all, dtype=int)

        # F1 at 0.5
        val_binary = (val_preds_arr >= 0.5).astype(int)
        tp = ((val_binary == 1) & (val_labels_arr == 1)).sum()
        fp = ((val_binary == 1) & (val_labels_arr == 0)).sum()
        fn = ((val_binary == 0) & (val_labels_arr == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(val_labels_arr, val_preds_arr)
        except ValueError:
            auc = 0.0

        scheduler.step(f1)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_auc = auc
            best_state = {k: v.cpu().clone() for k, v in fusion.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/100  train={avg_train:.4f}  val={avg_val:.4f}  "
                f"F1={f1:.4f}  P={prec:.3f}  R={rec:.3f}  AUC={auc:.4f}"
            )

    elapsed = time.time() - t0
    print(f"\n  Training time  : {elapsed:.1f}s")
    print(f"  Best val F1    : {best_val_f1:.4f}")
    print(f"  Best val AUC   : {best_val_auc:.4f}")

    # ── Threshold analysis ───────────────────────────────────────────
    fusion.load_state_dict(best_state)
    fusion.eval()
    with torch.no_grad():
        val_proba = fusion(torch.FloatTensor(X_joint[val_idx])).squeeze().numpy()
    val_true = y_joint[val_idx].astype(int)

    print("\n  Threshold analysis:")
    best_thresh, best_thresh_f1 = 0.5, 0.0
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (val_proba >= thresh).astype(int)
        tp = ((preds == 1) & (val_true == 1)).sum()
        fp = ((preds == 1) & (val_true == 0)).sum()
        fn = ((preds == 0) & (val_true == 1)).sum()
        tn = ((preds == 0) & (val_true == 0)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-8)
        if f > best_thresh_f1:
            best_thresh_f1 = f
            best_thresh = thresh
        print(f"    @{thresh:.1f}: P={p:.3f} R={r:.3f} F1={f:.3f} (TP={tp} FP={fp} FN={fn} TN={tn})")

    print(f"  Optimal threshold: {best_thresh} (F1={best_thresh_f1:.4f})")

    # ── Save ─────────────────────────────────────────────────────────
    fusion_checkpoint = {
        "model_state_dict": best_state,
        "input_dim": X_joint.shape[1],
        "architecture": f"{X_joint.shape[1]}->64->32->16->1",
        "xgb_model_path": "xgboost_lstm.pkl",
        "lstm_model_path": "lstm_autoencoder.pt",
        "best_val_f1": round(best_val_f1, 4),
        "best_val_auc_roc": round(best_val_auc, 4),
        "optimal_threshold": best_thresh,
        "trained_on": "own-cluster-data",
        "lstm_feat_min": lstm_ckpt["feat_min"],
        "lstm_feat_max": lstm_ckpt["feat_max"],
        "joint_feature_order": (
            ["xgb_proba"]
            + [f"lstm_emb_{i}" for i in range(lstm_ckpt["hidden_dim"])]
            + ["recon_error"]
            + FEATURES
        ),
    }
    out_path = OUTPUT_DIR / "xgboost_lstm_fusion.pt"
    torch.save(fusion_checkpoint, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved: xgboost_lstm_fusion.pt ({size_kb:.0f} KB)")

    # Update training summary
    summary_path = OUTPUT_DIR / "training_summary.json"
    summary = json.load(open(summary_path))
    summary["xgboost_lstm_fusion"] = {
        "val_f1": round(best_val_f1, 4),
        "val_auc_roc": round(best_val_auc, 4),
        "optimal_threshold": best_thresh,
        "joint_dim": X_joint.shape[1],
        "fusion_params": total_params,
        "trained_on": "own-cluster-data",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  END-TO-END ENSEMBLE COMPLETE")
    print("=" * 60)
    print(f"  XGBoost features  -> {1} dim (anomaly probability)")
    print(f"  LSTM embedding    -> {lstm_ckpt['hidden_dim']} dim (temporal representation)")
    print(f"  LSTM recon error  -> {1} dim (reconstruction anomaly)")
    print(f"  Raw features      -> {len(FEATURES)} dim (current metrics)")
    print(f"  Fusion input      -> {X_joint.shape[1]} dim total")
    print(f"  Fusion output     -> anomaly probability [0, 1]")
    print(f"  Val AUC-ROC       : {best_val_auc:.4f}")
    print(f"  Val F1            : {best_val_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
