# SKAM Predictive Capabilities: From Reactive to Predictive

> Research document -- March 2026
> Status: RESEARCH ONLY -- no implementation changes

---

## Table of Contents

1. [Current State](#1-current-state-reactive-detect--heal)
2. [The Prediction Problem](#2-the-prediction-problem)
3. [Feasible Predictive Features](#3-feasible-predictive-features)
4. [What Data We'd Need](#4-what-data-wed-need)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Honest Assessment](#6-honest-assessment)

---

## 1. Current State: Reactive (Detect -> Heal)

SKAM currently operates as a **reactive** system. It detects anomalies *after* they
manifest in live Prometheus telemetry, then triggers autonomous healing. The pipeline
works as follows:

### Architecture

```
Prometheus  -->  MetricsCollector  -->  AnomalyDetector  -->  DecisionEngine  -->  K8s API
(scrape 15s)     (5 PromQL queries)    (IF + LSTM ensemble)  (policy match)       (heal)
```

### Detection Pipeline

1. **MetricsCollector** (`collector.py`) queries Prometheus every ~15 seconds for 5
   features per service:
   - `request_rate` -- `rate(http_requests_total{app="..."}[1m])`
   - `error_rate` -- ratio of 5xx to total requests
   - `p99_latency` -- `histogram_quantile(0.99, ...)`
   - `cpu_usage` -- `rate(container_cpu_usage_seconds_total{...}[1m])`
   - `memory_usage` -- `container_memory_usage_bytes{...}`

2. **Isolation Forest** (`isolation_forest.py`) scores each 5-feature snapshot as a
   point anomaly. Uses running Z-score normalization with Welford's algorithm. Score
   mapped via sigmoid to [0,1].

3. **LSTM Autoencoder** (`lstm_detector.py`) scores temporal patterns across a sliding
   window of 20 time-steps (5 minutes at 15s intervals). Reconstruction error = anomaly
   score.

4. **Ensemble** (`detector.py`) combines scores with weights (IF=0.4, LSTM=0.6). An
   anomaly is declared only after 2 consecutive windows exceed the 0.7 threshold.

5. **Severity Classifier** (`severity.py`) maps scores to NORMAL/LOW/MEDIUM/HIGH/CRITICAL
   with escalation rules for sustained anomalies and rapid score velocity.

6. **Policy Engine** (`policies.py`) maps anomaly type + severity to healing actions
   (restart, scale up, remove network policy, etc.) with cooldown windows.

### What It Does Well

- Detects ongoing faults (CPU spike, latency spike, pod crash) within 30--60 seconds
- Classifies anomaly types using feature-level z-score attribution
- Autonomous healing with cooldown-based deduplication
- Pre-trained models from RCAEval eliminate cold-start delay

### What It Cannot Do

- **Predict** a failure before it happens
- **Forecast** when a metric will cross a threshold
- **Detect** slow-burn issues (memory leaks, disk fill) that are technically "normal"
  at each point but trending toward exhaustion
- **Anticipate** cascade failures from upstream service degradation
- **Adapt** to changing traffic patterns (concept drift)

---

## 2. The Prediction Problem

### Why Predicting Is Harder Than Detecting

Detection answers: "Is this data point abnormal *right now*?"
Prediction answers: "Will something go wrong *in the next N minutes*?"

This is fundamentally harder for several reasons:

**1. The signal-to-noise ratio drops dramatically.**
Current anomalies produce clear statistical deviations. Precursor signals -- the
subtle changes that happen *before* a crash -- are much weaker. A memory increase
of 2% per minute is indistinguishable from normal variance until you've observed it
for 20+ minutes.

**2. The class imbalance problem gets worse.**
Failures are rare events. In the RCAEval dataset, ~50% of each trace is pre-injection
(normal) and ~50% is post-injection (anomalous), which is unrealistically balanced.
In production, failures might represent <0.1% of all time windows.

**3. Labeled training data is scarce.**
Detection can be unsupervised (Isolation Forest) or semi-supervised (LSTM autoencoder
trained on normal data). Prediction requires *labeled pairs*: "here is what the metrics
looked like 5/10/30 minutes BEFORE this specific failure type." Our RCAEval dataset has
inject_time markers, but the pre-injection period is just normal operation -- it does
not contain the gradual degradation patterns you'd see in real production failures.

**4. Time horizon introduces uncertainty.**
Predicting 30 seconds ahead is feasible with trend extrapolation. Predicting 30 minutes
ahead requires understanding causality (service dependencies, load patterns, resource
contention). Each doubling of prediction horizon roughly halves accuracy.

**5. The counterfactual problem.**
If a prediction triggers preemptive healing and the failure never happens, was the
prediction correct? This makes measuring prediction accuracy difficult and complicates
training feedback loops.

### The Research Landscape (2024--2026)

Recent academic work confirms both the promise and difficulty:

- A Frontiers paper (2025) demonstrated Prophet + LSTM hybrid models for Kubernetes
  autoscaling, achieving 12% lower RMSE than single models for workload prediction.
- An MDPI paper (2024) proposed a "Night's Watch" algorithm for cloud-native anomaly
  detection combining multiple ML models.
- Research on failure diagnosis in microservices (ACM TOSEM 2024) cataloged 98 papers
  and found that most work focuses on *root cause analysis after failure*, not
  *prediction before failure*.
- A Wiley paper (2024) on cascading failure explanation from logs showed that failures
  propagate through known dependency paths, which makes cascade prediction tractable
  if the dependency graph is known.

---

## 3. Feasible Predictive Features

Ranked from easiest to hardest to implement, with specific integration points for SKAM.

---

### Feature 1: Trend Forecasting on Key Metrics

**Difficulty: LOW (1--2 weeks)**
**Impact: MEDIUM**
**Confidence: HIGH that it works**

#### What It Does

Fit short-horizon forecasting models (ARIMA, Prophet, or simple linear regression) to
each of the 5 metric streams per service. Forecast the value N minutes into the future.
If the forecast crosses a severity threshold, emit a *predictive alert* before the
actual threshold is breached.

#### How It Works

```python
# Pseudocode for trend forecasting integration
from pmdarima import auto_arima
# OR
from prophet import Prophet

class TrendForecaster:
    def __init__(self, horizon_minutes=10, history_minutes=60):
        self.horizon = horizon_minutes
        self.history = history_minutes
        self._models = {}  # (service, feature) -> fitted model

    def update(self, service: str, feature_name: str, timestamp, value):
        """Append to history buffer, refit periodically."""
        ...

    def forecast(self, service: str, feature_name: str) -> ForecastResult:
        """Return predicted value + confidence interval at t+horizon."""
        ...

    def will_breach_threshold(self, service, feature, threshold) -> float | None:
        """Return estimated minutes until threshold breach, or None."""
        # Linear extrapolation for simple cases:
        # slope = (y[-1] - y[-N]) / (t[-1] - t[-N])
        # time_to_threshold = (threshold - y[-1]) / slope
        ...
```

#### Library Choices

| Library | Pros | Cons | Fit for SKAM |
|---------|------|------|-------------|
| **pmdarima** (`auto_arima`) | Auto-selects ARIMA order, handles seasonality | Slow fitting (~1--5s per model), no GPU | Good for per-service batch updates every 5--10 min |
| **Prophet** | Handles daily/weekly seasonality natively, robust to missing data | Heavy dependency (cmdstanpy), slow for many models | Good if we have 14+ days of history |
| **statsmodels** (Holt-Winters) | Lightweight, fast, handles trends | Manual parameter tuning | Good for simple trend extrapolation |
| **Simple linear regression** (numpy) | Near-zero latency, trivial to implement | Only catches linear trends | Best for first iteration |

#### Prometheus Queries

The existing 5 queries in `collector.py` are sufficient. For longer history, use
range queries:

```promql
# 1-hour history at 15s resolution (240 data points)
rate(http_requests_total{app="api-gateway"}[1m])[1h:15s]

# For Prophet/ARIMA, we'd want 24h-7d of history:
rate(http_requests_total{app="api-gateway"}[1m])[7d:1m]
```

#### Integration with Decision Engine

Add a new alert type to `AnomalyAlert`:

```python
class PredictiveAlert(BaseModel):
    service: str
    metric: str              # which metric is forecasted to breach
    current_value: float
    predicted_value: float
    predicted_breach_time: datetime
    confidence: float        # model's prediction confidence
    horizon_minutes: int
```

The policy engine would need new policies:

```python
HealingPolicy(
    name="predicted-cpu-saturation",
    anomaly_type="predicted_resource",    # new type
    severity_threshold=0.6,
    action_type="scale_up",
    parameters={"replicas_add": 1},       # gentler than reactive
    cooldown_seconds=300,
)
```

---

### Feature 2: Precursor Pattern Mining

**Difficulty: MEDIUM (2--3 weeks)**
**Impact: HIGH**
**Confidence: MEDIUM (depends on data quality)**

#### What It Does

Analyze historical metric windows that *preceded* known failures and extract
discriminative patterns. Use these patterns to recognize when current metrics resemble
a pre-failure state.

#### How It Works

1. **Build a precursor dataset** from RCAEval data:
   - For each scenario, take the 5--10 minutes *before* inject_time as "pre-failure"
   - Take random normal windows as "normal"
   - Label them as (pre-failure=1, normal=0)

2. **Train a classifier** (Random Forest, XGBoost, or a small neural net) on these
   windowed sequences.

3. **At runtime**, continuously classify the most recent window and emit alerts when
   the pre-failure probability exceeds a threshold.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class PrecursorDetector:
    """Detects metric patterns that historically preceded failures."""

    def __init__(self, lookback_minutes=10, features_per_step=5):
        self.lookback = lookback_minutes
        self.window_size = lookback_minutes * 4  # at 15s intervals
        self.model = GradientBoostingClassifier(n_estimators=100)

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract statistical features from a (T, 5) window.

        Features include:
        - Mean, std, min, max of each metric
        - Slope (linear trend) of each metric
        - Rate of change (first derivative)
        - Coefficient of variation
        - Cross-correlations between metrics
        """
        stats = []
        for col in range(window.shape[1]):
            series = window[:, col]
            stats.extend([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.polyfit(range(len(series)), series, 1)[0],  # slope
                np.mean(np.abs(np.diff(series))),              # mean abs change
                np.std(series) / (np.mean(series) + 1e-8),     # coeff of variation
            ])
        return np.array(stats)
```

#### RCAEval Data Suitability

The RCAEval dataset has a **critical limitation** for precursor detection: faults are
injected *instantaneously* at inject_time. In reality, failures often have gradual
onset (memory leak over 30 min, cascading latency over 5 min). The pre-injection
window in RCAEval is just normal data -- it does not contain the gradual degradation
you'd want to learn from.

**Mitigation**: Use the *first few minutes after injection* as the "precursor" period,
since the full failure may take time to develop. Alternatively, generate synthetic
precursor data by interpolating between normal and anomalous distributions.

---

### Feature 3: Capacity Exhaustion Prediction

**Difficulty: LOW-MEDIUM (1--2 weeks)**
**Impact: HIGH**
**Confidence: HIGH (well-understood problem)**

#### What It Does

Detect slow-burn resource exhaustion: memory leaks, disk fill, connection pool drain,
thread count growth. These are "normal" at each point in time but will eventually
cause an outage if unchecked.

#### How It Works

For each resource metric, fit a simple trend model and estimate time-to-exhaustion:

```python
import numpy as np

class CapacityPredictor:
    """Predicts when a resource will hit its limit."""

    def __init__(self, min_history_minutes=30):
        self.min_history = min_history_minutes

    def predict_exhaustion(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        capacity_limit: float,
    ) -> dict:
        """Estimate time until resource exhaustion.

        Uses robust linear regression on the recent trend.
        Returns minutes until capacity is reached, or None if trend is flat/negative.
        """
        if len(values) < self.min_history * 4:  # need 30+ min at 15s intervals
            return {"exhaustion_minutes": None, "confidence": 0.0}

        # Use last 30 minutes for trend estimation
        recent = values[-self.min_history * 4:]
        t = np.arange(len(recent), dtype=np.float64)

        # Robust linear fit (use median of slopes for outlier resistance)
        slopes = []
        for i in range(0, len(recent) - 10, 5):
            chunk = recent[i:i+10]
            chunk_t = t[i:i+10]
            if len(chunk) >= 2:
                slope = np.polyfit(chunk_t, chunk, 1)[0]
                slopes.append(slope)

        median_slope = np.median(slopes)  # per-step slope

        if median_slope <= 0:
            return {"exhaustion_minutes": None, "confidence": 0.0}

        current = values[-1]
        remaining = capacity_limit - current
        steps_to_exhaustion = remaining / median_slope
        minutes = steps_to_exhaustion * 0.25  # 15s per step

        # Confidence based on R-squared of the trend
        predicted = current + median_slope * np.arange(len(recent))
        ss_res = np.sum((recent - predicted[-len(recent):])**2)
        ss_tot = np.sum((recent - np.mean(recent))**2)
        r_squared = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "exhaustion_minutes": max(0, minutes),
            "slope_per_minute": median_slope * 4,
            "current_value": float(current),
            "capacity_limit": capacity_limit,
            "confidence": float(r_squared),
        }
```

#### Prometheus Queries for Capacity Limits

```promql
# Memory limit for a pod
kube_pod_container_resource_limits{resource="memory", pod=~"api-gateway.*"}

# Current usage vs limit ratio
container_memory_usage_bytes{pod=~"api-gateway.*"}
  / on(pod) kube_pod_container_resource_limits{resource="memory"}

# Disk usage (if applicable)
kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes

# JVM heap (for Java services)
jvm_memory_used_bytes{area="heap"} / jvm_memory_max_bytes{area="heap"}
```

#### Memory Leak Detection Heuristic

A memory leak has a specific signature: monotonically increasing memory with
periodic partial drops (GC cycles that reclaim less each time). Detect this by:

1. Compute the rolling minimum over 10-minute windows
2. If the rolling minimum is itself increasing, it's likely a leak
3. Estimate time until the rolling minimum reaches the memory limit

---

### Feature 4: Anomaly Trajectory Prediction

**Difficulty: LOW (1 week)**
**Impact: MEDIUM**
**Confidence: HIGH**

#### What It Does

SKAM already computes score velocity in `severity.py`. Extend this to predict *when*
the anomaly score will cross severity thresholds.

#### How It Works

The `SeverityClassifier` already tracks score history and computes velocity. Extend it:

```python
def predict_threshold_crossing(
    self,
    service: str,
    target_threshold: float = 0.7,  # anomaly threshold
) -> float | None:
    """Predict seconds until anomaly score crosses target_threshold.

    Returns estimated seconds, or None if score is stable/decreasing.
    Uses the last 5 score observations for velocity estimation.
    """
    history = self._score_history.get(service)
    if not history or len(history) < 3:
        return None

    recent = list(history)[-5:]
    current_score = recent[-1][1]

    if current_score >= target_threshold:
        return 0.0  # already above threshold

    velocity = self._compute_velocity(service)  # already implemented
    if velocity <= 0:
        return None  # score is stable or decreasing

    remaining = target_threshold - current_score
    seconds_to_threshold = remaining / velocity

    # Only return if prediction is within a reasonable horizon (30 min)
    if seconds_to_threshold > 1800:
        return None

    return seconds_to_threshold
```

#### Integration

This is the simplest predictive feature to add because it builds directly on existing
infrastructure in `severity.py`. The decision engine could use it to issue "early
warning" alerts with lower severity:

```python
# In the detection loop:
seconds = severity_classifier.predict_threshold_crossing(service)
if seconds is not None and seconds < 300:  # threshold breach in < 5 min
    emit_predictive_alert(
        service=service,
        alert_type="trajectory_warning",
        estimated_breach_seconds=seconds,
    )
```

---

### Feature 5: Dependency Cascade Prediction

**Difficulty: HIGH (3--4 weeks)**
**Impact: VERY HIGH**
**Confidence: MEDIUM (requires service mesh telemetry)**

#### What It Does

If service A shows anomalous behavior, predict which downstream services (B, C, D)
will be affected and how quickly. This enables preemptive scaling or circuit-breaking
*before* the cascade reaches critical services.

#### How It Works

1. **Build a dependency graph** from Kubernetes service topology and/or distributed
   tracing data (Jaeger/Zipkin spans).

2. **Learn propagation patterns**: For each edge in the graph, learn the typical
   delay between anomaly in upstream and impact on downstream.

3. **At detection time**: When service A goes anomalous, walk the dependency graph
   and emit predictive alerts for downstream services.

```python
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DependencyEdge:
    upstream: str
    downstream: str
    avg_propagation_delay_s: float
    impact_probability: float  # how often upstream failures affect downstream
    impact_severity_ratio: float  # how much severity is amplified/dampened

class CascadePredictor:
    def __init__(self):
        self.graph: dict[str, list[DependencyEdge]] = defaultdict(list)

    def load_from_traces(self, trace_data):
        """Build dependency graph from distributed tracing spans."""
        ...

    def predict_cascade(
        self,
        failed_service: str,
        failure_severity: float,
    ) -> list[CascadePrediction]:
        """Predict downstream impact of a failure in failed_service."""
        predictions = []
        visited = set()
        queue = [(failed_service, failure_severity, 0.0)]  # (service, severity, delay)

        while queue:
            svc, sev, delay = queue.pop(0)
            if svc in visited:
                continue
            visited.add(svc)

            for edge in self.graph.get(svc, []):
                downstream_sev = sev * edge.impact_severity_ratio
                downstream_delay = delay + edge.avg_propagation_delay_s

                if downstream_sev > 0.3 and edge.impact_probability > 0.5:
                    predictions.append(CascadePrediction(
                        service=edge.downstream,
                        predicted_severity=downstream_sev,
                        estimated_impact_delay_s=downstream_delay,
                        probability=edge.impact_probability,
                    ))
                    queue.append((edge.downstream, downstream_sev, downstream_delay))

        return sorted(predictions, key=lambda p: p.estimated_impact_delay_s)
```

#### Data Requirements

This feature needs one or more of:
- **Distributed tracing** (Jaeger/Zipkin) to discover service dependencies
- **Istio/Linkerd service mesh** metrics for inter-service call rates
- **Kubernetes service topology** from the API
- **Historical incident data** mapping which services were co-affected

#### Prometheus Queries for Dependency Discovery

```promql
# Inter-service call rates (requires Istio)
istio_requests_total{
  source_app="api-gateway",
  destination_app="user-service"
}

# Or from application-level metrics if instrumented:
http_client_requests_total{target_service="user-service", source="api-gateway"}
```

---

### Feature 6: Seasonal Baseline Learning

**Difficulty: MEDIUM (2--3 weeks)**
**Impact: MEDIUM-HIGH**
**Confidence: HIGH**

#### What It Does

Learn that "high CPU at 9am Monday is normal" vs "high CPU at 3am Sunday is anomalous."
Current SKAM uses a single global baseline per service. Seasonal baselines would
drastically reduce false positives during known traffic peaks and increase sensitivity
during quiet periods.

#### How It Works

Replace the single running mean/variance in `IsolationForestDetector` with
time-bucketed baselines:

```python
from collections import defaultdict
import numpy as np

class SeasonalBaseline:
    """Maintains per-time-bucket baselines for seasonal normalization.

    Buckets: 168 (24 hours x 7 days) for hourly resolution with day-of-week.
    """

    def __init__(self, n_features=5, n_buckets=168):
        self.n_features = n_features
        self.n_buckets = n_buckets
        # Per-bucket Welford statistics
        self._count = np.zeros(n_buckets, dtype=np.int64)
        self._mean = np.zeros((n_buckets, n_features), dtype=np.float64)
        self._m2 = np.zeros((n_buckets, n_features), dtype=np.float64)

    def _get_bucket(self, timestamp) -> int:
        """Map a datetime to its hourly-weekly bucket (0-167)."""
        hour = timestamp.hour
        day = timestamp.weekday()  # 0=Monday
        return day * 24 + hour

    def update(self, timestamp, features: np.ndarray):
        """Update the baseline for the appropriate time bucket."""
        bucket = self._get_bucket(timestamp)
        self._count[bucket] += 1
        delta = features - self._mean[bucket]
        self._mean[bucket] += delta / self._count[bucket]
        delta2 = features - self._mean[bucket]
        self._m2[bucket] += delta * delta2

    def normalize(self, timestamp, features: np.ndarray) -> np.ndarray:
        """Normalize features relative to seasonal baseline."""
        bucket = self._get_bucket(timestamp)
        if self._count[bucket] < 50:  # not enough data for this bucket
            # Fall back to global mean
            global_mean = np.mean(self._mean[self._count > 0], axis=0)
            global_std = np.ones(self.n_features)
            return (features - global_mean) / global_std

        std = np.sqrt(self._m2[bucket] / (self._count[bucket] - 1))
        std[std < 1e-8] = 1.0
        return (features - self._mean[bucket]) / std

    def is_calibrated(self, min_samples_per_bucket=50) -> bool:
        """Check if we have enough history for seasonal baselines."""
        # Need at least 70% of buckets to have sufficient data
        calibrated = np.sum(self._count >= min_samples_per_bucket)
        return calibrated >= self.n_buckets * 0.7
```

#### Data Requirements

Seasonal baselines need **at least 2 weeks** (ideally 4 weeks) of continuous metric
history to populate all 168 hourly buckets with statistically meaningful samples. At
15-second scrape intervals, each hour contributes 240 samples. After 2 weeks, each
bucket has ~480 samples.

#### Prometheus Queries for Historical Baseline

```promql
# Pull 4 weeks of hourly averages for baseline initialization
avg_over_time(
  rate(http_requests_total{app="api-gateway"}[5m])[4w:1h]
)

# Seasonal comparison: current vs same hour last week
rate(http_requests_total{app="api-gateway"}[5m])
  / on() avg_over_time(
      rate(http_requests_total{app="api-gateway"}[5m])[1h] offset 7d
  )
```

---

## 4. What Data We'd Need

### Current Data (Available Now)

| Data Source | What We Have | Limitation |
|-------------|-------------|------------|
| RCAEval Online Boutique | 10 services, 5 fault types, ~70 min traces | Faults are instantaneous injection, not gradual degradation |
| Prometheus (live) | 5 metrics per service at 15s intervals | Only available during live cluster operation |
| Pre-trained models | IF + LSTM trained on RCAEval normal data | Trained for detection, not prediction |

### Additional Data Needed for Prediction

| Data Need | Purpose | How to Obtain | Priority |
|-----------|---------|---------------|----------|
| **Longer metric history (7-30 days)** | Seasonal baselines, trend forecasting | Configure Prometheus retention or use Thanos/Mimir for long-term storage | P0 |
| **Labeled incident data** | Precursor pattern training | Instrument SKAM to log every detected anomaly with timestamps and outcomes | P0 |
| **Resource limits per pod** | Capacity exhaustion prediction | Query `kube_pod_container_resource_limits` from Prometheus | P1 |
| **Service dependency graph** | Cascade prediction | Parse Kubernetes Services/Endpoints or use Istio/Jaeger | P1 |
| **Distributed traces** | Dependency latency learning | Deploy Jaeger/Zipkin or use OpenTelemetry collector | P2 |
| **Synthetic degradation scenarios** | Gradual failure training data | Create chaos experiments with slow ramp-up (not instant injection) | P2 |
| **Change events** (deploys, config changes) | Correlate failures with causes | Webhook from CI/CD to SKAM | P3 |

### Is the RCAEval Dataset Sufficient for Predictive Training?

**Short answer: Partially, with caveats.**

The RCAEval dataset provides:
- 10 services x 5 fault types x multiple runs = diverse fault scenarios
- Pre-injection and post-injection segments clearly separated by `inject_time`
- 5 metric features matching our detector's feature vector

However, it **lacks**:
- **Gradual degradation patterns**: Faults are injected instantaneously. Real memory
  leaks develop over hours. Real cascading failures propagate over seconds-to-minutes.
- **Seasonal variation**: All traces are ~70 minutes. No day/night or weekday/weekend
  patterns.
- **Multi-failure scenarios**: Each scenario has exactly one injected fault. Production
  systems experience correlated failures.
- **Recovery data**: What do metrics look like as a system recovers? This matters for
  predicting "will this incident resolve on its own?"

**Recommendation**: Use RCAEval for initial model validation, but collect real
production telemetry (or generate synthetic gradual-degradation scenarios) for
training production-grade predictive models.

---

## 5. Implementation Roadmap

### Phase 1: Quick Wins (Weeks 1--2)

**Goal**: Add predictive *awareness* without changing the core pipeline.

1. **Anomaly Trajectory Prediction** (Feature 4)
   - Extend `SeverityClassifier` with `predict_threshold_crossing()`
   - Add trajectory warnings to the dashboard
   - *No new dependencies, minimal code change*

2. **Simple Trend Forecasting** (Feature 1, linear only)
   - Add linear regression on 30-minute windows for each metric
   - Predict minutes-to-threshold for CPU and memory
   - *Dependency: numpy only (already installed)*

3. **Capacity Exhaustion for Memory** (Feature 3, simplified)
   - Track rolling minimum of memory usage
   - Alert if rolling minimum has a positive slope (possible leak)
   - *Dependency: numpy only*

### Phase 2: Statistical Forecasting (Weeks 3--4)

**Goal**: Real forecasting models with confidence intervals.

4. **ARIMA/Prophet Integration** (Feature 1, full)
   - Install `pmdarima` for auto-ARIMA on per-service metrics
   - Forecast 10--30 minutes ahead with confidence intervals
   - *New dependency: pmdarima (~5MB)*

5. **Seasonal Baseline Learning** (Feature 6)
   - Implement `SeasonalBaseline` class
   - Requires 2+ weeks of metric history to become effective
   - Integrate with anomaly normalization
   - *No new dependencies*

### Phase 3: Pattern-Based Prediction (Weeks 5--8)

**Goal**: Learn from past incidents.

6. **Precursor Pattern Mining** (Feature 2)
   - Build precursor dataset from RCAEval + any production incidents
   - Train gradient boosting classifier on windowed features
   - *New dependency: scikit-learn (already installed), optionally xgboost*

7. **Concept Drift Detection**
   - Monitor the Isolation Forest's internal statistics for distribution shift
   - Trigger model retraining when drift exceeds threshold
   - *New dependency: `river` or `alibi-detect` for drift detection*

### Phase 4: Graph-Based Prediction (Weeks 9--12+)

**Goal**: Understand service interdependencies.

8. **Dependency Graph Construction** (Feature 5, prerequisite)
   - Parse Kubernetes service topology
   - Optionally ingest Jaeger traces for weighted edges
   - *New dependency: networkx for graph operations*

9. **Cascade Prediction** (Feature 5)
   - Implement `CascadePredictor` with BFS over dependency graph
   - Learn propagation delays from historical data
   - Emit pre-emptive alerts for downstream services

### Architecture for Prediction Integration

```
                          +-------------------+
                          | Trend Forecaster  |  (Feature 1)
                          | (Prophet/ARIMA)   |
                          +--------+----------+
                                   |
Prometheus --> MetricsCollector ---+---> AnomalyDetector ---> DecisionEngine --> K8s
                |                  |         |                     |
                |                  |    +----+-----+         +----+------+
                |                  |    | Severity |         | Predictive|
                |                  |    | + Traject.|        | Policy    |
                |                  |    +----------+         | Engine    |
                |                  |                         +-----------+
                |           +------+--------+
                |           | Capacity      |  (Feature 3)
                |           | Predictor     |
                |           +---------------+
                |
                |           +---------------+
                +---------->| Seasonal      |  (Feature 6)
                |           | Baseline      |
                |           +---------------+
                |
                |           +---------------+
                +---------->| Precursor     |  (Feature 2)
                            | Detector      |
                            +---------------+
```

The decision engine would need a new `PredictivePolicyEngine` that handles predictive
alerts with different semantics:
- Lower confidence thresholds (prediction is inherently uncertain)
- Gentler actions (scale by +1, not +2)
- Longer cooldowns (avoid thrashing on noisy predictions)
- Confirmation windows (wait for a second prediction before acting)

---

## 6. Honest Assessment

### What's Practical in 2--4 Weeks

| Feature | Feasibility | Value | Notes |
|---------|-------------|-------|-------|
| Anomaly trajectory prediction | Very High | Medium | 1-2 days of work, extends existing code |
| Linear trend forecasting | Very High | Medium | ~3 days, numpy only |
| Memory leak detection | High | High | ~1 week, clear heuristic |
| CPU/latency trend alerts | High | Medium | ~1 week with pmdarima |
| Seasonal baselines | Medium | High | Code is easy (~1 week) but needs 2+ weeks of data to calibrate |

### What's a Semester Project (2--4 Months)

| Feature | Challenge | Notes |
|---------|-----------|-------|
| Precursor pattern mining | Need labeled incident data we don't have | RCAEval is insufficient; need production data or synthetic gradual-failure generation |
| Dependency cascade prediction | Need service mesh or tracing infrastructure | Graph construction is the bottleneck, not the prediction algorithm |
| Concept drift adaptation | Need to define "drift" for our specific models | Research shows DNN+Autoencoder approaches, but tuning is domain-specific |

### What's a PhD Thesis

| Feature | Why It's Hard |
|---------|---------------|
| General failure prediction (arbitrary failure, arbitrary horizon) | Unsolved problem; requires causal understanding of the system |
| Counterfactual evaluation of predictions | If we prevent a failure, we can never confirm the prediction was right |
| Transfer learning across clusters | Different deployments have different baselines, dependencies, and failure modes |
| Multi-modal prediction (logs + metrics + traces) | Data fusion across heterogeneous telemetry streams is an active research area |

### Key Risks

1. **False positive fatigue**: Predictive alerts that don't materialize erode trust.
   Mitigation: Start with high-confidence predictions only (capacity exhaustion, clear
   linear trends) and require 2+ confirmations before acting.

2. **Prediction-induced oscillation**: If predictions trigger scaling, the scaling
   changes the metrics, which changes the prediction. This feedback loop can cause
   thrashing. Mitigation: Cooldown windows and damping factors.

3. **Stale models**: Predictive models degrade faster than detection models because
   they depend on future behavior, which changes with every deployment. Mitigation:
   Continuous retraining and concept drift monitoring.

4. **Computational cost**: Running Prophet or ARIMA for 6 services x 5 metrics = 30
   models, each re-fit every 5--10 minutes, adds non-trivial CPU overhead.
   Mitigation: Use simple linear extrapolation as default, Prophet only for services
   with known seasonal patterns.

### Recommended Starting Point

**Start with Feature 4 (anomaly trajectory prediction) and Feature 3 (capacity
exhaustion prediction).** These are:
- Trivial to implement (days, not weeks)
- Use only numpy (no new dependencies)
- Build on existing infrastructure (severity classifier, metric collector)
- Provide immediate, tangible value
- Low false-positive risk (extrapolating a clear trend is reliable)

Then iterate toward Feature 1 (ARIMA forecasting) and Feature 6 (seasonal baselines)
once there is sufficient metric history.

---

## References

### Academic Papers

- "AI-Driven Anomaly Detection in Cloud-Native Microservices: The Night's Watch
  Algorithm" -- MDPI Applied Sciences, 2024
- "Time Series Forecasting-based Kubernetes Autoscaling Using Facebook Prophet and
  Long Short-Term Memory" -- Frontiers in Computer Science, 2025
- "Failure Diagnosis in Microservice Systems: A Comprehensive Survey and Analysis"
  -- ACM Transactions on Software Engineering and Methodology, 2024
- "Explaining Microservices' Cascading Failures From Their Logs" -- Wiley Software:
  Practice and Experience, 2024
- "Concept Drift Adaptation for Time Series Anomaly Detection via Transformer" --
  Springer Neural Processing Letters, 2022
- "A Comprehensive Survey on Root Cause Analysis in (Micro) Services" -- arXiv, 2024
- "Resilient Microservices: A Systematic Review of Recovery Patterns, Strategies,
  and Evaluation Frameworks" -- arXiv, 2025
- "Telemetry-based Software Failure Prediction by Concept Drift Detection" -- HAL, 2023
- "Deep Learning for Time Series Anomaly Detection: A Survey" -- ACM Computing
  Surveys, 2024

### Tools and Libraries

- [pmdarima](https://alkaline-ml.com/pmdarima/) -- Auto-ARIMA for Python
- [Prophet](https://facebook.github.io/prophet/) -- Facebook/Meta time-series forecasting
- [statsmodels](https://www.statsmodels.org/) -- Statistical models (Holt-Winters, ARIMA)
- [river](https://riverml.xyz/) -- Online/incremental learning for concept drift
- [alibi-detect](https://github.com/SeldonIO/alibi-detect) -- Drift detection library
- [KEDA](https://keda.sh/) -- Kubernetes event-driven autoscaling
- [Grafana Anomaly Detection](https://grafana.com/blog/2024/10/03/how-to-use-prometheus-to-efficiently-detect-anomalies-at-scale/) -- PromQL-based anomaly detection

### Industry Resources

- "AI-Powered Predictive Scaling in Kubernetes" -- DEV Community, 2025
- "Predictive Autoscaling in Kubernetes with KEDA and Prophet" -- Medium, 2024
- "How to Use Prometheus to Efficiently Detect Anomalies at Scale" -- Grafana Labs, 2024
