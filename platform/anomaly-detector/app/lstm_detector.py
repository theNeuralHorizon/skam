"""Stage-2 anomaly detector -- LSTM Autoencoder for temporal patterns."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import structlog
import torch
import torch.nn as nn

from app.metrics import model_training_samples

logger = structlog.get_logger(__name__)

_NUM_FEATURES = 5
_WINDOW_SIZE = 20
_HIDDEN_SIZE = 32
_NUM_LAYERS = 2
_TRAIN_EPOCHS = 30
_LEARNING_RATE = 1e-3
_MIN_SEQUENCES = 30


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        _, hidden = self.lstm(x)
        return hidden


class _Decoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        seq_len: int,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Repeat the last hidden output across the time dimension
        batch_size = hidden[0].size(1)
        decoder_input = hidden[0][-1].unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, H)
        output, _ = self.lstm(decoder_input, hidden)
        return self.fc(output)  # (B, T, output_size)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int = _NUM_FEATURES,
        hidden_size: int = _HIDDEN_SIZE,
        num_layers: int = _NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.encoder = _Encoder(input_size, hidden_size, num_layers)
        self.decoder = _Decoder(hidden_size, input_size, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        return self.decoder(x.size(1), hidden)


# ---------------------------------------------------------------------------
# High-level detector
# ---------------------------------------------------------------------------

class LSTMDetector:
    """Temporal anomaly detection via an LSTM Autoencoder.

    Maintains a sliding window of ``_WINDOW_SIZE`` time-steps for each
    service.  The reconstruction error (MSE) is normalised to ``[0, 1]``
    using an exponential moving average of the historical maximum error.
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = LSTMAutoencoder().to(self._device)
        self._trained = False
        self._lock = threading.Lock()

        # Per-service sliding windows: service -> deque of feature vectors
        self._windows: dict[str, deque[np.ndarray]] = {}

        # Running normalisation for reconstruction error
        self._ema_max_error: float = 1.0
        self._ema_alpha: float = 0.05

        # Buffer of completed sequences for training
        self._train_buffer: list[np.ndarray] = []

        # Feature normalisation (simple min-max from training data)
        self._feat_min: np.ndarray | None = None
        self._feat_max: np.ndarray | None = None

    # -- public API ----------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    def window_ready(self, service: str) -> bool:
        """Return True if the sliding window for *service* has enough data."""
        return len(self._windows.get(service, [])) >= _WINDOW_SIZE

    def push(self, service: str, features: np.ndarray) -> None:
        """Append a feature vector to the sliding window for *service*."""
        features = np.asarray(features, dtype=np.float32).ravel()
        if service not in self._windows:
            self._windows[service] = deque(maxlen=_WINDOW_SIZE)
        self._windows[service].append(features)

        # When a window fills, snapshot it for training
        if len(self._windows[service]) == _WINDOW_SIZE:
            seq = np.stack(list(self._windows[service]))
            self._train_buffer.append(seq)
            model_training_samples.labels(detector="lstm").set(len(self._train_buffer))

    def train(self, sequences: list[np.ndarray] | None = None) -> None:
        """Train the LSTM Autoencoder on a list of (T, F) numpy arrays."""
        seqs = sequences if sequences is not None else self._train_buffer
        if len(seqs) < _MIN_SEQUENCES:
            logger.info("lstm_detector.skip_train", sequences=len(seqs), required=_MIN_SEQUENCES)
            return

        data = np.stack(seqs).astype(np.float32)  # (N, T, F)

        # Compute feature-wise min/max for normalisation
        flat = data.reshape(-1, _NUM_FEATURES)
        self._feat_min = flat.min(axis=0)
        self._feat_max = flat.max(axis=0)
        denom = self._feat_max - self._feat_min
        denom[denom < 1e-8] = 1.0
        data_norm = (data - self._feat_min) / denom

        tensor = torch.tensor(data_norm, dtype=torch.float32, device=self._device)

        self._model.train()
        optimiser = torch.optim.Adam(self._model.parameters(), lr=_LEARNING_RATE)
        criterion = nn.MSELoss()

        for epoch in range(_TRAIN_EPOCHS):
            optimiser.zero_grad()
            reconstructed = self._model(tensor)
            loss = criterion(reconstructed, tensor)
            loss.backward()
            optimiser.step()

            if (epoch + 1) % 10 == 0:
                logger.debug("lstm_detector.train_epoch", epoch=epoch + 1, loss=loss.item())

        with self._lock:
            self._trained = True
            # Keep recent half of buffer
            self._train_buffer = self._train_buffer[-(len(self._train_buffer) // 2):]

        logger.info("lstm_detector.trained", sequences=len(seqs), final_loss=loss.item())

    def predict(self, sequence: np.ndarray) -> float:
        """Return a reconstruction-error-based anomaly score in ``[0, 1]``.

        *sequence* should have shape ``(_WINDOW_SIZE, _NUM_FEATURES)``.
        """
        if not self._trained:
            return 0.0

        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]  # (1, T, F)

        # Normalise using training statistics
        if self._feat_min is not None and self._feat_max is not None:
            denom = self._feat_max - self._feat_min
            denom[denom < 1e-8] = 1.0
            sequence = (sequence - self._feat_min) / denom

        tensor = torch.tensor(sequence, dtype=torch.float32, device=self._device)

        self._model.eval()
        with self._lock, torch.no_grad():
            reconstructed = self._model(tensor)

        mse = float(torch.mean((tensor - reconstructed) ** 2).item())

        # Update EMA of max error for normalisation
        if mse > self._ema_max_error:
            self._ema_max_error = mse
        else:
            self._ema_max_error = (
                self._ema_alpha * mse + (1 - self._ema_alpha) * self._ema_max_error
            )

        # Normalise to [0, 1]
        score = mse / max(self._ema_max_error, 1e-8)
        return float(np.clip(score, 0.0, 1.0))

    def predict_for_service(self, service: str) -> float:
        """Convenience: run prediction on the current sliding window."""
        if not self.window_ready(service):
            return 0.0
        seq = np.stack(list(self._windows[service]))
        return self.predict(seq)

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._trained = True
        logger.info("lstm_detector.weights_loaded", path=path)

    def save_weights(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)
        logger.info("lstm_detector.weights_saved", path=path)
