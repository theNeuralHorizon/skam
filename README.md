<div align="center">

```
   ███████╗██╗  ██╗ █████╗ ███╗   ███╗
   ██╔════╝██║ ██╔╝██╔══██╗████╗ ████║
   ███████╗█████╔╝ ███████║██╔████╔██║
   ╚════██║██╔═██╗ ██╔══██║██║╚██╔╝██║
   ███████║██║  ██╗██║  ██║██║ ╚═╝ ██║
   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
```

### **Self-Healing Kubernetes • Predictive Outage Detection • Chaos Engineering at Scale**

*An autonomous platform that breaks itself, detects the breakage with a four-model ML ensemble, and heals itself — all before a human even notices something went wrong.*

<br />

#### **Kshitij Betwal**

[![Email](https://img.shields.io/badge/kshitij.betwal@gmail.com-0A66C2?style=flat-square&logo=gmail&logoColor=white)](mailto:kshitij.betwal@gmail.com)
[![GitHub](https://img.shields.io/badge/theNeuralHorizon-0969DA?style=flat-square&logo=github&logoColor=white)](https://github.com/theNeuralHorizon)
[![Profile](https://img.shields.io/badge/Portfolio-0284C7?style=flat-square&logo=vercel&logoColor=white)](https://github.com/theNeuralHorizon)

<br />

[![Go](https://img.shields.io/badge/Go-1.22-0EA5E9?style=for-the-badge&logo=go&logoColor=white)](https://go.dev/)
[![Python](https://img.shields.io/badge/Python-3.11-0369A1?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-k3d-1D4ED8?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-0891B2?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-MetaLearner-0EA5E9?style=for-the-badge)](https://xgboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-0284C7?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-0369A1?style=for-the-badge&logo=react&logoColor=white)](https://react.dev/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-1E40AF?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0EA5E9?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />

**`AUC-ROC 0.9919`**  `·`  **`F1 0.9832`**  `·`  **`<15s MTTD`**  `·`  **`<30s MTTR`**  `·`  **`Zero Human Intervention`**

</div>

---

## The Thesis

> **Modern distributed systems fail. Constantly. In ways no runbook can predict.**
>
> SKAM's bet is simple: stop writing runbooks. Train a model. Let the cluster heal itself.

SKAM (Self-healing Kubernetes Anomaly Mitigation) is a production-grade reference platform that **continuously injects failures into its own microservices**, **detects those failures using a four-model ML ensemble on live Prometheus telemetry**, and **autonomously recovers** — by restarting pods, scaling deployments, and removing faulty network policies through the Kubernetes API.

No rule-based alerts. No PagerDuty. No humans. Just telemetry, tensors, and the control plane.

---

## The Closed Loop

<div align="center">

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │    🔥 INJECT  ────►  🧠 DETECT  ────►  ⚖️  DECIDE  ────►  💚 HEAL│
    │      ▲                                                     │    │
    │      │                                                     │    │
    │      └─────────────────────────────────────────────────────┘    │
    │                    (all in < 30 seconds)                        │
    └─────────────────────────────────────────────────────────────────┘
```

</div>

| Stage | Component | What Happens |
|-------|-----------|--------------|
| 🔥 **Inject** | `chaos-engine` | Kills pods, starves memory, blocks networks, injects latency — on a schedule |
| 🧠 **Detect** | `anomaly-detector` | Scrapes Prometheus, runs 4-model ensemble, emits confidence-scored alerts |
| ⚖️ **Decide** | `decision-engine` | Maps alert → policy → K8s action. Validates recovery via PromQL |
| 💚 **Heal** | Kubernetes API | Pods restart. HPAs scale. NetworkPolicies are removed. Life goes on. |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              k3d Cluster                                   │
│                                                                            │
│  ┌──────────────── MICROSERVICES (passive targets) ──────────────────┐    │
│  │                                                                    │    │
│  │   api-gateway ─┐                                                   │    │
│  │   user-svc ────┤                                                   │    │
│  │   product-svc ─┼─── /metrics (15s scrape) ────► Prometheus         │    │
│  │   order-svc ───┤                                    │              │    │
│  │   payment-svc ─┤                                    │              │    │
│  │   notif-svc ───┘                                    │              │    │
│  │                                                     │              │    │
│  │   (Go 1.22 · chi · pgx · go-redis · Postgres · Redis)              │    │
│  └────────────────────────────────────────────────────┼──────────────┘    │
│                                                       │                    │
│                                                       │  PromQL            │
│                                                       ▼                    │
│  ┌────────────────── PLATFORM (active operators) ────────────────────┐    │
│  │                                                                    │    │
│  │    ┌───────────────┐        ┌──────────────────────────────────┐  │    │
│  │    │ Chaos Engine  │        │   Anomaly Detector (ML Core)     │  │    │
│  │    │               │        │                                  │  │    │
│  │    │ • pod-kill    │        │  ┌────────────────────────────┐  │  │    │
│  │    │ • mem-limit   │        │  │ XGBoost Meta-Learner       │  │  │    │
│  │    │ • net-policy  │        │  │  ├─ Isolation Forest ──┐   │  │  │    │
│  │    │ • latency     │        │  │  ├─ LSTM Autoencoder ──┤   │  │  │    │
│  │    │ • cache-kill  │        │  │  ├─ XGBoost+LSTM ──────┤   │  │  │    │
│  │    └───────┬───────┘        │  │  └─ XGBoost+Attention ─┘   │  │  │    │
│  │            │                │  └────────────┬───────────────┘  │  │    │
│  │            │ K8s API        │               │ score > 0.7      │  │    │
│  │            ▼                │               ▼                  │  │    │
│  │      [ microservices ]      │      POST /alerts                │  │    │
│  │                             └──────────────┬───────────────────┘  │    │
│  │                                            │                      │    │
│  │                                            ▼                      │    │
│  │                             ┌─────────────────────────┐           │    │
│  │                             │  Decision Engine        │           │    │
│  │                             │   policy → action       │           │    │
│  │                             │   validate → confirm    │           │    │
│  │                             └───────────┬─────────────┘           │    │
│  │                                         │ K8s API                 │    │
│  │                                         ▼                         │    │
│  │                                 [ microservices (healed) ]        │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐       │
│   │Prometheus│  │   Loki   │  │ Grafana  │  │  React Dashboard    │       │
│   │ (scrape) │  │  (logs)  │  │  (viz)   │  │  (live WebSocket)   │       │
│   └──────────┘  └──────────┘  └──────────┘  └─────────────────────┘       │
└────────────────────────────────────────────────────────────────────────────┘
```

**Microservices and platform components never talk to each other directly.** They communicate through two intermediaries:
1. **Prometheus** — the observability bus (pull-based metric scraping + PromQL reads)
2. **Kubernetes API** — the control plane (one-way mutations: delete, patch, scale)

This decoupling is the whole point: microservices have **zero knowledge** of the ML pipeline or the healing logic. Any service that exposes Prometheus metrics can be protected.

---

## The ML Core — Why a Four-Model Ensemble

Anomaly detection on noisy telemetry is a losing game for any single model. Isolation Forests catch point outliers but miss temporal patterns. LSTMs learn sequence behavior but cold-start poorly. XGBoost is fast but struggles without time-aware features.

**Solution:** stack them. Train a meta-learner on top.

<div align="center">

### Benchmark Leaderboard *(on RCAEval Online Boutique, 5 fault types, balanced test set)*

| Rank | Model | AUC-ROC | AUC-PR | F1 | Precision@0.7 | Throughput |
|:----:|:------|:-------:|:------:|:--:|:-------------:|:----------:|
| 🥇 | **XGBoost + LSTM Ensemble** | **`0.9919`** | **`0.9947`** | **`0.9832`** | `1.000` | 2,361 /s |
| 🥈 | **XGBoost + Attention** | `0.9883` | `0.9913` | `0.9736` | `0.9965` | 2,405 /s |
| 🥉 | **XGBoost Meta-Learner (PCA)** | `0.9815` | — | `0.9041` | — | — |
| 4 | Isolation Forest + LSTM Fusion | `0.8798` | `0.9027` | `0.8151` | `1.000` | 160 /s |
| 5 | Isolation Forest (standalone) | `0.9073` | `0.9225` | `0.8286` | `0.7342` | 160 /s |
| 6 | IQR (statistical) | `0.8411` | `0.7940` | `0.8364` | `0.8670` | 107,516 /s |
| 7 | One-Class SVM | `0.8185` | `0.8825` | `0.8177` | `0.9867` | 11,046 /s |
| 8 | Z-Score (statistical) | `0.7696` | `0.8472` | `0.7653` | `0.9503` | 278,784 /s |
| 9 | EWMA (baseline) | `0.4943` | `0.5130` | `0.6764` | `0.6396` | 132,026 /s |

*The top-3 models are all stacked ensembles. Every single-model approach is dominated.*

</div>

### Model Architectures

<details>
<summary><b>1. Isolation Forest</b> — point outlier detection on 5-D feature space</summary>

- Trained on 8,800 samples of `(request_rate, error_rate, latency, cpu_usage, memory_usage)`
- `score_mean = -0.47`, `score_std = 0.044` — tight normal distribution
- Training time: 0.42s. Inference: 160 scores/sec.
- **Strength:** zero-parameter, works on unseen faults.
- **Weakness:** no temporal awareness.

</details>

<details>
<summary><b>2. LSTM Autoencoder</b> — temporal reconstruction error</summary>

- PyTorch. 27K parameters. 64-hidden, 2-layer bidirectional encoder/decoder.
- Trained on 6,425 windows of 30-timestep sequences.
- Final train loss: `5.9e-5`. Threshold: `4.57e-4`. **Separation ratio: 53,199×**.
- **Strength:** catches slow-burning degradations (latency creep, memory leaks).
- **Weakness:** 30-sample cold start.

</details>

<details>
<summary><b>3. XGBoost + LSTM Fusion</b> — tree-based boosted classifier</summary>

- 14,090 training samples. Validation accuracy: `99.43%`.
- Training time: 0.39s (yes, really).
- **Strength:** fast, robust, handles non-linear feature interactions.

</details>

<details>
<summary><b>4. XGBoost + Attention Network</b> — tree + transformer</summary>

- XGBoost trained on 14,090 tabular samples (0.17s).
- Attention net: 5,025 params, trained on 12,863 sequences of 15 steps (31.89s).
- Best validation loss: `0.0175`.
- **Strength:** attention weights highlight *which timestep* triggered the anomaly — great for RCA.

</details>

<details>
<summary><b>5. XGBoost Meta-Learner with PCA</b> — the ensemble head</summary>

- Takes the 4 base-model outputs + engineered statistics → 39-D feature vector.
- PCA compresses to 11-D while retaining `99.9%` of variance.
- Meta-XGBoost trained on own-cluster data.
- **Val AUC-ROC: 0.9815 · F1: 0.9041 · Optimal threshold: 0.45**.
- **Strength:** calibrated confidence scores across heterogeneous base models.

</details>

### Predictive Outage Detection *(new in `feat/predictive-outage-detection`)*

Rather than reacting to anomalies, SKAM now **forecasts them**. A dedicated predictor runs continuously on metric windows, and when it emits a "likely failure within N seconds" signal, the decision engine **pre-emptively scales or drains** the affected service — healing the system *before* the fault manifests.

This flips the traditional MTTD/MTTR curves: the observable recovery time becomes effectively zero, because the recovery started before the outage did.

---

## Tech Stack

<div align="center">

| Layer | Technology | Why |
|:------|:-----------|:----|
| **Orchestration** | `k3d` (K3s in Docker) | Reproducible, single-machine K8s — no cloud dependency |
| **Microservices** | `Go 1.22` · `chi` · `pgx` · `go-redis` | Fast cold-start, low memory, easy Prometheus instrumentation |
| **Datastores** | `PostgreSQL 15` · `Redis 7` | The stateful realism that makes chaos scenarios interesting |
| **Chaos Engine** | `Python` · `kubernetes-client` | Direct control-plane mutations — no Chaos Mesh dependency |
| **Anomaly Detection** | `PyTorch 2.5` · `XGBoost` · `scikit-learn` | Best-in-class tabular + temporal model stack |
| **Self-Healing** | `FastAPI` · `kubernetes-client` | Async-friendly policy engine with PromQL validation |
| **Metrics** | `Prometheus` · `kube-prometheus-stack` | The de-facto standard; SKAM reads it, writes it, lives on it |
| **Logs** | `Loki` · `promtail` | Grafana-native log aggregation |
| **Visualization** | `Grafana` · `React 18` · `Vite` · `TailwindCSS` · `Recharts` | Two dashboards: operator (Grafana) and demo (React) |
| **CI/CD** | `Docker` · `Makefile` · `Helm` | One-command deploys, reproducible images |

</div>

---

## Quick Start

```bash
# 1.  Install prerequisites (k3d, helm, kubectl, docker) — one-shot
bash setup.sh

# 2.  Build all 9 container images (6 Go services + 3 Python platform services)
make build

# 3.  Spin up k3d cluster + deploy infrastructure + deploy everything
make deploy

# 4.  Kick off baseline traffic (1000 req/s across all services)
make load-test

# 5.  Run the full chaos demo (5 failure scenarios, each with auto-recovery)
make chaos-demo

# 6.  Open the real-time React dashboard
make dashboard
# → http://localhost:5173
```

After `make deploy`, you get:
- **Grafana** at `http://localhost:3000` (admin / prom-operator)
- **Prometheus** at `http://localhost:9090`
- **Dashboard** at `http://localhost:5173` — live WebSocket feed of anomalies, chaos events, and healing actions

---

## Chaos Demo Scenarios

<div align="center">

| # | Scenario | Fault Injected | Detection Signal | Recovery Action |
|:-:|:--------|:---------------|:-----------------|:----------------|
| 1 | **Pod Kill Recovery** | `kubectl delete pod order-service-xxx` | error_rate spike + latency → ∞ | Auto-restart via ReplicaSet controller |
| 2 | **Memory Pressure** | Patch container to `limits.memory: 64Mi` | OOMKilled loop + request_rate crash | Increase limit → rolling restart |
| 3 | **Network Partition** | `NetworkPolicy` blocks payment-service | error_rate = 100% on payment edge | Remove NetworkPolicy |
| 4 | **Cascading Failure** | 500ms artificial latency via tc | Queue depth growth + HPA misfire | HPA scale-up + circuit-break |
| 5 | **Cache Failure** | `kubectl delete pod redis-0` | cache_hit_ratio → 0, DB load spike | Restart Redis + cache warm-up |

*Every scenario recovers in under 30 seconds, end-to-end, with no human in the loop.*

</div>

---

## Project Structure

```
skam/
├── services/                    # 6 Go microservices (the things that break)
│   ├── api-gateway/             # Public entry point, auth, rate limiting
│   ├── user-service/            # User CRUD + sessions (Postgres)
│   ├── product-service/         # Catalog + inventory (Postgres + Redis)
│   ├── order-service/           # Order state machine (Postgres)
│   ├── payment-service/         # Payment processing (Postgres)
│   └── notification-service/    # Async fanout (Redis pub/sub)
│
├── platform/                    # 3 Python platform operators (the things that heal)
│   ├── chaos-engine/            # Scheduled fault injection via K8s API
│   ├── anomaly-detector/        # 4-model ML ensemble + FastAPI alert emitter
│   │   └── app/
│   │       ├── isolation_forest.py
│   │       ├── lstm_detector.py
│   │       ├── xgboost_detector.py
│   │       ├── attention_detector.py
│   │       ├── xgboost_meta_detector.py   # ← the meta-learner
│   │       └── predictor.py               # ← predictive outage detection
│   └── decision-engine/         # Policy engine + K8s action executor + recovery validator
│
├── ml/                          # Offline training + benchmarking
│   ├── training/                # Model training scripts (reproducible)
│   ├── models/                  # Pickled/torch-saved model weights (shipped into images)
│   ├── data/                    # RCAEval Online Boutique traces + own-cluster logs
│   ├── benchmark_results/       # JSON metrics + 10 comparison charts (PNG)
│   └── generate_report.py       # Regenerates the leaderboard + charts
│
├── dashboard/                   # React 18 + Vite live monitoring UI
│   └── src/components/
│       ├── LiveMetrics.jsx
│       ├── AnomalyTimeline.jsx
│       ├── ChaosPanel.jsx
│       ├── EventLog.jsx
│       ├── ServiceTopology.jsx
│       └── PredictionDashboard.jsx        # ← pre-outage forecasts
│
├── k8s/
│   ├── cluster/                 # k3d cluster config
│   ├── infrastructure/          # Helm values for Prometheus, Loki, Grafana
│   ├── microservices/           # Deployments, Services, HPAs, ConfigMaps
│   └── rbac/                    # ServiceAccount + ClusterRole for platform
│
├── scripts/                     # Load generator, mock server, demo runner
├── tests/                       # Unit + integration tests (pytest)
├── Makefile                     # One-command orchestration of everything
└── setup.sh                     # Prerequisite installer (k3d, helm, kubectl)
```

---

## What Makes This Different

Most "chaos engineering" demos ship Chaos Mesh + a Grafana dashboard and call it a day. SKAM is different in three ways:

1. **The ML is real.** The repo ships actual trained model weights (`lstm_autoencoder.pt`, `xgboost_lstm.pkl`, `attention_net.pt`, etc.), a reproducible training pipeline, and a benchmark harness that compares **nine** detection approaches on a balanced, labeled dataset. The `0.9919` AUC-ROC isn't a marketing number — it's in [`ml/benchmark_results/benchmark_results.json`](ml/benchmark_results/benchmark_results.json).

2. **The loop is closed.** Detection without automated recovery is just a prettier PagerDuty. SKAM's decision engine doesn't just alert — it executes `kubectl patch`, waits, re-queries Prometheus, and *verifies the recovery worked*. If it didn't, it escalates to the next policy.

3. **The platform is decoupled.** No sidecars. No service mesh. No framework lock-in. Any workload that exposes a `/metrics` endpoint can be protected by SKAM, because the ML sees telemetry, not application code.

---

## Other Completed Projects

A selection of other shipped work from [@theNeuralHorizon](https://github.com/theNeuralHorizon) — *shipping neural systems at the edge of production · robotics · fintech · on-device ML · cybersecurity.*

<div align="center">

| Project | Domain | Stack | Summary |
|:--------|:------:|:------|:--------|
| [**claimrail**](https://github.com/theNeuralHorizon/claimrail) | Fintech / SaaS | TypeScript · LLM | AI-powered SaaS SLA credit recovery — monitors every vendor, detects breaches, auto-drafts claims. |
| [**drifting-oracle**](https://github.com/theNeuralHorizon/drifting-oracle) | Fintech / MLOps | Python · LLM | Credit Risk Model Drift Detection + LLM Hallucination Monitoring. **HackBricks 2026 submission.** |
| [**FakeCallShield**](https://github.com/theNeuralHorizon/FakeCallShield-) | Cybersecurity | Android · On-device ML | Detects fake, spoofed, and AI-generated phone calls **entirely on-device.** |
| [**MetaPyTorchScalerHackathon**](https://github.com/theNeuralHorizon/MetaPyTorchScalerHackathon) | MLOps / SRE | Python · PyTorch | IncidentEnv: production incident root cause analysis environment. **Meta × HuggingFace × PyTorch OpenEnv Hackathon.** |
| [**mobilityyy**](https://github.com/theNeuralHorizon/mobilityyy) | Robotics | ROS 2 · Gazebo · Python | Exploration-based autonomous navigation for ARTPARK grid arena. **MIT Hackathon R3 submission.** |
| [**predictive-maintenance-system**](https://github.com/theNeuralHorizon/predictive-maintenance-system) | Industrial ML | JavaScript · ML | Deployed predictive maintenance platform — [live demo](https://predictive-maintenance-system-two.vercel.app). |
| [**Credit-Card-Financial-Dashboard**](https://github.com/theNeuralHorizon/Credit-Card-Financial-Dashboard) | Data / BI | Power BI · Docker | Containerized Power BI dashboard for credit-card financial analytics. |
| [**cat-vs-dog-classification**](https://github.com/theNeuralHorizon/cat-vs-dog-classification) | Computer Vision | Jupyter · CNN | Classic image-classification pipeline with Jupyter training notebooks. |

</div>

---

## Roadmap

- [x] 6-service Go microservices with Prometheus instrumentation
- [x] Chaos engine with 5 failure scenarios
- [x] Isolation Forest + LSTM Autoencoder baseline
- [x] XGBoost + LSTM + Attention ensemble
- [x] XGBoost Meta-Learner with PCA compression
- [x] React dashboard with WebSocket live feed
- [x] K3d single-command deploy
- [x] Predictive (pre-outage) detection + preemptive healing
- [ ] Distributed training + federated model updates across clusters
- [ ] LLM-powered postmortem generation from anomaly trace + healing timeline
- [ ] Multi-tenant control plane (one detector, many clusters)
- [ ] Cost-aware healing (prefer scale-up during off-peak, restart during peak)

---

## License

MIT — see [`LICENSE`](LICENSE). Use it, fork it, ship it to production, write a paper about it.
