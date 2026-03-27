"""Prometheus metrics for the SKAM Chaos Engine."""

from prometheus_client import Counter, Gauge, Histogram

# Total experiments executed, labelled by fault type and terminal status.
chaos_experiments_total = Counter(
    "chaos_experiments_total",
    "Total number of chaos experiments executed",
    ["fault_type", "status"],
)

# Duration of each experiment in seconds.
chaos_experiment_duration_seconds = Histogram(
    "chaos_experiment_duration_seconds",
    "Duration of chaos experiments in seconds",
    ["fault_type"],
    buckets=[5, 10, 30, 60, 120, 300, 600, 1800, 3600],
)

# Number of experiments currently in the 'running' state.
chaos_active_experiments = Gauge(
    "chaos_active_experiments",
    "Number of currently active chaos experiments",
)

# Total faults injected, labelled by fault type (incremented on successful
# injection regardless of whether the experiment later fails or is rolled back).
chaos_faults_injected_total = Counter(
    "chaos_faults_injected_total",
    "Total number of individual faults injected",
    ["fault_type"],
)
