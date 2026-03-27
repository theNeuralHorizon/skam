#!/usr/bin/env python3
"""Pre-train Isolation Forest and LSTM Autoencoder on the RCAEval Online
Boutique dataset.

Usage::

    python -m ml.training.train_models                 # from repo root
    python ml/training/train_models.py                  # direct invocation
    python ml/training/train_models.py --data-dir ml/data/rcaeval/online-boutique
    python ml/training/train_models.py --epochs 30 --seq-length 20

Outputs (saved to ``ml/models/``)::

    isolation_forest.pkl    – scikit-learn IsolationForest + scaler
    lstm_autoencoder.pt     – PyTorch LSTM Autoencoder state dict
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

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "dataset": str(data_dir),
        "interval_seconds": args.interval,
        "isolation_forest": if_stats,
        "lstm_autoencoder": lstm_stats,
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
    print(f"║  LSTM threshold   : {lstm_stats['threshold']:>10.6f}            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\n  Artefacts in: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
