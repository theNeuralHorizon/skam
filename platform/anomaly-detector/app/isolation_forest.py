"""Stage-1 anomaly detector based on scikit-learn Isolation Forest."""

from __future__ import annotations

import threading

import numpy as np
import structlog
from sklearn.ensemble import IsolationForest

from app.metrics import model_training_samples

logger = structlog.get_logger(__name__)

_NUM_FEATURES = 5
_RETRAIN_THRESHOLD = 1000
_MIN_TRAIN_SAMPLES = 50


class IsolationForestDetector:
    """Unsupervised anomaly scoring via Isolation Forest.

    Scores are normalised to the ``[0, 1]`` range where higher means more
    anomalous.  The detector maintains a running Z-score normaliser and
    automatically retrains when its internal buffer exceeds
    ``_RETRAIN_THRESHOLD`` samples.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state

        self._model: IsolationForest | None = None
        self._trained = False
        self._lock = threading.Lock()

        # Running Z-score statistics
        self._count: int = 0
        self._mean = np.zeros(_NUM_FEATURES, dtype=np.float64)
        self._m2 = np.zeros(_NUM_FEATURES, dtype=np.float64)  # sum of squared diffs

        # Buffer for (re-)training
        self._buffer: list[np.ndarray] = []

    # -- public API ----------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def sample_count(self) -> int:
        return self._count

    def add_sample(self, features: np.ndarray) -> None:
        """Ingest a raw feature vector, update stats, and buffer for training.

        If the buffer reaches ``_RETRAIN_THRESHOLD`` the model is retrained
        automatically.
        """
        features = np.asarray(features, dtype=np.float64).ravel()
        assert features.shape == (_NUM_FEATURES,), f"Expected {_NUM_FEATURES} features, got {features.shape}"

        self._update_running_stats(features)
        self._buffer.append(features.copy())
        model_training_samples.labels(detector="isolation_forest").set(len(self._buffer))

        if len(self._buffer) >= _RETRAIN_THRESHOLD:
            self.train()

    def train(self, data: np.ndarray | None = None) -> None:
        """Fit (or re-fit) the Isolation Forest on supplied or buffered data."""
        if data is None:
            if len(self._buffer) < _MIN_TRAIN_SAMPLES:
                logger.info(
                    "isolation_forest.skip_train",
                    samples=len(self._buffer),
                    required=_MIN_TRAIN_SAMPLES,
                )
                return
            data = np.stack(self._buffer)

        normalised = self._normalise(data)

        model = IsolationForest(
            contamination=self._contamination,
            n_estimators=self._n_estimators,
            random_state=self._random_state,
            n_jobs=-1,
        )
        model.fit(normalised)

        with self._lock:
            self._model = model
            self._trained = True
            # Keep only the most recent half to avoid unbounded growth
            self._buffer = self._buffer[-(len(self._buffer) // 2):]

        logger.info("isolation_forest.trained", samples=len(data))

    def predict(self, features: np.ndarray) -> float:
        """Return an anomaly score in ``[0, 1]``.

        The raw Isolation Forest ``decision_function`` output is negative for
        anomalies.  We map it via a sigmoid-like transform so that higher
        values indicate stronger anomalies.
        """
        if not self._trained or self._model is None:
            return 0.0

        features = np.asarray(features, dtype=np.float64).reshape(1, _NUM_FEATURES)
        normalised = self._normalise(features)

        with self._lock:
            raw_score = self._model.decision_function(normalised)[0]

        # decision_function: large negative -> anomaly, near zero / positive -> normal
        # Map to [0, 1] with sigmoid centred around 0
        score = 1.0 / (1.0 + np.exp(5.0 * raw_score))
        return float(np.clip(score, 0.0, 1.0))

    # -- internals -----------------------------------------------------------

    def _update_running_stats(self, x: np.ndarray) -> None:
        """Welford's online algorithm for running mean / variance."""
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def _running_std(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(_NUM_FEATURES, dtype=np.float64)
        variance = self._m2 / (self._count - 1)
        std = np.sqrt(variance)
        # Avoid division by zero: replace near-zero std with 1
        std[std < 1e-8] = 1.0
        return std

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalisation using the running mean and std."""
        return (data - self._mean) / self._running_std()
