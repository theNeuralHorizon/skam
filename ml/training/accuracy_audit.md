# ML Pipeline Accuracy Audit

**Date:** 2026-03-28
**Dataset:** RCAEval Online Boutique (125 scenarios, 5 fault types, 5 services)
**Balanced:** 600 samples per fault type (3000 normal, 3000 anomalous)

## 1. Data Quality

- **Zero NaN/Inf** across all fault types
- **No data leakage**: 0/100 sampled test points match training data (min distance = 0.44)
- **Balance**: cpu/mem have 3500 raw samples, delay/disk/loss have 600 each. We subsample to 600 per fault type for fairness.

## 2. Feature Separability (Cohen's d per fault type)

| Fault | request_rate | error_rate | latency | cpu_usage | memory |
|-------|-------------|-----------|---------|-----------|--------|
| cpu | 0.03 | 0.00 | 0.11 | **9.93** | 0.20 |
| mem | 0.22 | 0.00 | **2.16** | **16.66** | **2.75** |
| delay | 0.36 | 0.00 | 0.90 | 0.17 | 0.01 |
| disk | 0.03 | 0.00 | **1.16** | **3.39** | 0.10 |
| loss | 0.03 | 0.00 | 0.26 | 0.13 | 0.02 |

**Finding:** cpu and mem faults are trivially separable via cpu_usage (d > 9). Loss faults have near-zero separability on all features — our 5-feature vector doesn't capture network packet loss.

## 3. Cross-Validation (5-Fold Stratified)

| Fold | Train AUC | Test AUC | Gap |
|------|-----------|----------|-----|
| 1 | 0.9155 | 0.9245 | -0.009 |
| 2 | 0.9145 | 0.9040 | +0.010 |
| 3 | 0.9190 | 0.9109 | +0.008 |
| 4 | 0.9175 | 0.9109 | +0.007 |
| 5 | 0.9191 | 0.9334 | -0.014 |
| **Mean** | **0.9171 ± 0.002** | **0.9167 ± 0.011** | **0.0004** |

**Finding:** No overfitting. Train-test gap is 0.0004 (negligible). The model generalizes well.

## 4. Per-Fault AUC-ROC

| Fault Type | AUC-ROC | Difficulty |
|-----------|---------|------------|
| mem | 1.000 | Trivial |
| disk | 0.994 | Easy |
| delay | 0.992 | Easy |
| cpu | 0.986 | Easy |
| **loss** | **0.696** | **Hard** |

**Finding:** Aggregate AUC (0.917) is inflated by easy faults. Loss faults are barely above random (0.5). This is a feature limitation — network packet loss doesn't strongly affect CPU, memory, latency, or request rate in the RCAEval dataset.

## 5. Score Distribution

| Metric | Normal | Anomalous |
|--------|--------|-----------|
| Mean score | 0.477 | 0.917 |
| Std dev | 0.286 | 0.166 |
| p5 | 0.046 | 0.469 |
| p50 | 0.460 | 0.999 |
| p95 | 0.940 | 1.000 |

- **Separation:** 0.440 (good)
- **Overlap:** 73.5% of anomaly scores fall within normal score range
- **At threshold 0.7:** 26.7% false positive rate, 89.4% recall

## 6. Honest Assessment

### What's Accurate
- 0.917 AUC-ROC is reproducible across 5 folds (± 0.011)
- No overfitting, no data leakage
- cpu/mem/disk/delay detection is genuinely excellent (>0.98 AUC each)

### What's Inflated
- Aggregate AUC is pulled up by trivially-separable faults (cpu_usage d=9.93)
- Loss faults at 0.696 AUC are barely above random
- 26.7% false positive rate at 0.7 threshold is not great for production

### What Would Improve It
1. Add network-level features (packet loss rate, connection errors, TCP retransmissions) for loss fault detection
2. Lower threshold to 0.6 for better recall at cost of more false positives, or use severity-based thresholds
3. Use per-fault-type thresholds instead of one global 0.7
4. Add more loss fault training data (only 600 samples vs 3500 for cpu/mem)
