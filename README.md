# SKAM — Autonomous Chaos Engineering & Self-Healing Platform

A production-grade platform that deploys microservices on Kubernetes, programmatically injects failures, detects anomalies using ML on live telemetry, and autonomously recovers — without human intervention.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        k3d Cluster                              │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │API Gateway│→│User Svc  │  │Product   │  │Order Svc │       │
│  │  :8080   │  │ +Postgres│  │Svc+PG   │  │+PG+Redis │       │
│  └────┬─────┘  └──────────┘  └──────────┘  └────┬─────┘       │
│       │         ┌──────────┐  ┌──────────┐       │             │
│       └────────→│Payment   │  │Notif Svc │←──────┘             │
│                 │  Svc     │  │  +Redis  │                     │
│                 └──────────┘  └──────────┘                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Platform Components                         │   │
│  │  ┌─────────────┐  ┌────────────┐  ┌─────────────────┐  │   │
│  │  │Chaos Engine │→ │Anomaly     │→ │Decision Engine  │  │   │
│  │  │(inject)     │  │Detector(ML)│  │(heal)           │  │   │
│  │  └─────────────┘  └────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Prometheus│  │  Loki    │  │ Grafana  │  │Dashboard │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Closed Loop

```
Inject Failure → Detect Anomaly → Decide Action → Recover Automatically
     ↑                                                      │
     └──────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | k3d (K3s in Docker) |
| Microservices | Go 1.22, chi router, pgx, go-redis |
| Databases | PostgreSQL 15, Redis 7 |
| Chaos Engine | Python, kubernetes-client |
| Anomaly Detection | Isolation Forest + LSTM Autoencoder (PyTorch) |
| Self-Healing | Python, FastAPI, kubernetes-client |
| Observability | Prometheus, Loki, Grafana |
| Dashboard | React 18, TypeScript, Vite, TailwindCSS, Recharts |

## Quick Start

```bash
# 1. Run the setup script (installs k3d, helm, etc.)
bash setup.sh

# 2. Build all container images
make build

# 3. Deploy everything to the cluster
make deploy

# 4. Start the load generator (generates baseline traffic)
make load-test

# 5. Run the chaos demo (5 failure scenarios)
make chaos-demo

# 6. Launch the real-time dashboard
make dashboard
```

## Demo Scenarios

| # | Scenario | Fault Injected | Recovery Action |
|---|----------|---------------|-----------------|
| 1 | Pod Kill Recovery | Kill order-service pod | Auto-restart via K8s API |
| 2 | Memory Pressure | Set 64Mi memory limit | Increase limit + restart |
| 3 | Network Partition | Block payment-service | Remove NetworkPolicy |
| 4 | Cascading Failure | Inject 500ms latency | HPA scale-up |
| 5 | Cache Failure | Kill Redis pod | Restart + cache warm-up |

## Project Structure

```
skam/
├── services/               # Go microservices (6 services)
│   ├── api-gateway/
│   ├── user-service/
│   ├── product-service/
│   ├── order-service/
│   ├── payment-service/
│   └── notification-service/
├── platform/               # Python platform components
│   ├── chaos-engine/       # Fault injection
│   ├── anomaly-detector/   # ML-based detection
│   └── decision-engine/    # Self-healing logic
├── dashboard/              # React real-time UI
├── k8s/                    # Kubernetes manifests
│   ├── cluster/            # k3d config
│   ├── infrastructure/     # Helm values (Prometheus, Loki, Grafana)
│   ├── microservices/      # Deployments, Services, HPAs
│   └── rbac/               # RBAC for platform access
├── scripts/                # Load generator, demo runner
├── Makefile                # Top-level automation
└── setup.sh                # One-click prerequisite installer
```

## License

MIT
