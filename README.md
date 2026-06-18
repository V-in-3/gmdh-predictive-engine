# GMDH Architecture Efficiency Engine 🚀

A **self-learning monitoring and predictive system** for cloud architecture efficiency. The project demonstrates an end-to-end MLOps pipeline — from synthetic data generation and model training to real-time simulation and automated resource cleanup — orchestrated entirely by Apache Airflow.

> **Core idea:** Instead of hardcoding alert thresholds (e.g. CPU > 80% → alarm), the system learns non-linear dependencies between infrastructure metrics and predicts degradation *before* it becomes critical.

---

## 🎯 Why This Project Exists

This is a portfolio project that showcases:

- **MLOps lifecycle** — data generation → training → model persistence → inference → cleanup
- **Transparent ML** — GMDH produces an interpretable polynomial, not a black-box prediction
- **Data pipeline engineering** — Kafka ingestion, MySQL sync, DLQ handling, recursive reconciliation
- **Infrastructure as Code** — fully Dockerized, reproducible with a single `docker-compose up`
- **Production patterns** — idempotency, dead letter queues, self-healing DAGs, parallel processing

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION (Airflow)                      │
├──────────────────┬──────────────────┬───────────────────────────┤
│  ML PREDICTION   │  DATA INTEGRITY  │     CODE QUALITY          │
│                  │                  │                           │
│  CSV → Spark →   │  Kafka → MySQL   │  DAG files → Pylint →    │
│  GMDH model →    │  (audit + DLQ)   │  Athena (trend analysis) │
│  efficiency      │                  │                           │
│  scoring         │  Lag monitor →   │                           │
│                  │  recursive sync  │                           │
└──────────────────┴──────────────────┴───────────────────────────┘
         │                   │                      │
         ▼                   ▼                      ▼
   "Is the system      "Is all data         "Is the codebase
    healthy?"           accounted for?"       degrading?"
```

The system monitors **3 levels of health**:
1. **Infrastructure efficiency** — ML model predicts system health from latency, auth status, and CPU load
2. **Data integrity** — guarantees zero data loss between Kafka and MySQL with recursive reconciliation
3. **Code quality** — tracks Pylint scores over time and detects regression

---

## 🧠 The GMDH Algorithm

**GMDH (Group Method of Data Handling)** is a self-organizing approach to building polynomial models, invented by Alexei Ivakhnenko (1968).

### Why GMDH instead of Neural Networks / XGBoost?

| Criterion | GMDH | Black-box ML |
|-----------|------|--------------|
| Interpretability | Final formula is a readable polynomial | Opaque |
| Auditability | Can explain *why* an alert fired | Cannot |
| Model size | ~500 bytes JSON | MB–GB |
| Inference speed | Single formula, microseconds | Requires ML runtime |
| Self-organization | Automatically selects important interactions | Manual feature engineering |

### How It Works (2-Layer Architecture)

```
Inputs: x1 (API latency), x2 (auth status), x3 (CPU load)
                    │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │node_x1_x2│ │node_x1_x3│ │node_x2_x3│   ← Layer 1: all C(3,2) pairs
   └────┬─────┘ └────┬─────┘ └────┬─────┘
        │             │             │
        └── RMSE selection (top 2) ─┘         ← External criterion
                    │
              ┌─────┴─────┐
              ▼           ▼
           z1 (best)   z2 (2nd best)
              │           │
              ▼           ▼
        ┌─────────────────────┐
        │    Master Node      │               ← Layer 2
        │ f(z1, z2, z1·z2)   │
        └─────────┬───────────┘
                  ▼
           efficiency score (0–1)
```

Each neuron computes: `ŷ = β₀ + β₁·xᵢ + β₂·xⱼ + β₃·(xᵢ·xⱼ)`

The final deployed model is a **4th-order polynomial** with fully interpretable coefficients.

### Output Thresholds

| Efficiency | Status | Action |
|-----------|--------|--------|
| > 75% | ✅ OK | No action |
| 45–75% | ⚠️ WARN | Investigate |
| < 45% | 🚨 CRITICAL | Immediate response |

---

## 🛠 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Apache Airflow 2.10.5 | DAG scheduling, task dependencies, recursive triggers |
| Processing | Apache Spark (Scala) | GMDH model training & simulation |
| Streaming | Apache Kafka (KRaft, Confluent 7.6) | Event ingestion, DLQ |
| Storage | AWS S3 (via S3A) + MySQL | Model artifacts, operational data |
| Cloud Emulation | LocalStack | Kinesis & S3 for local development |
| Containerization | Docker Compose | Full environment in one command |
| Code Quality | Pylint + AWS Athena | Historical score tracking |

---

## 📂 Project Structure

```
gmdh-predictive-engine/
├── dags/
│   ├── kafka_event_generator.py      # Produces random subscription events to Kafka
│   ├── kafka_queue_monitor.py        # Lag detection → triggers audit if gap found
│   ├── market_transaction_generator.py # SP-API + Cybersource events → Kinesis
│   └── marketplace_audit.py          # Recursive Kafka→MySQL sync with DLQ
├── dags_backup/
│   └── gmdh_predictive_engine_it.py  # Core ML DAG (train → simulate → cleanup)
├── jobs/
│   ├── airflow_pylint_qa/            # Pylint analysis framework
│   └── marketplace_audit/            # Integration tests
├── scripts/
│   └── generate_it_dataset.py        # Synthetic dataset generator (10K records)
├── data/
│   └── fintech_transactions_raw.csv  # Generated training data
├── kafka/
│   └── docker-compose.yaml           # Kafka KRaft single-node cluster
├── docker-compose.yaml               # Airflow + LocalStack
├── Dockerfile                        # Custom Airflow image with Kafka client
└── gmdh_secrets.json                 # Config (excluded from git)
```

---

## 🚀 Quick Start

### Prerequisites

```bash
brew install apache-spark jq
```

### 1. Start Infrastructure

```bash
# Start Kafka
cd kafka && docker-compose up -d && cd ..

# Start Airflow + LocalStack
docker-compose up -d
```

### 2. Generate Training Data

```bash
cd scripts && python generate_it_dataset.py
```

### 3. Run the ML Pipeline

Trigger `gmdh_predictive_engine_it` DAG from Airflow UI (`http://localhost:8080`).

The pipeline will:
1. Write Scala scripts to `/tmp`
2. Train GMDH model → save to S3
3. Run live simulation with random inputs
4. Clean up all artifacts (S3 + local)

---

## 🔄 Data Integrity Pipeline

The audit system guarantees **zero data loss** between Kafka and MySQL:

```
kafka_lag_monitor (every 3 min)
        │
        ├─ gap = 0 → sleep
        │
        └─ gap > 0 → trigger marketplace_audit
                            │
                            ├─ parallel consume (3 partitions)
                            ├─ INSERT IGNORE (deduplication)
                            ├─ bad JSON → DLQ (Kafka + MySQL)
                            │
                            └─ validate → gap still > 0?
                                    │           │
                                    yes         no
                                    │           │
                              trigger self    done ✅
                              (recursion)
```

Key patterns:
- **Idempotent replay** — always scans from offset 0, `UNIQUE` constraint prevents duplicates
- **Dual DLQ** — errors persist in both Kafka topic and MySQL table
- **Self-healing** — recursive trigger until convergence

---

## 🧹 Clean Data Lake Principles

The pipeline ensures **zero-footprint execution**:

- Temporary Scala source files are deleted after runtime
- Model artifacts are removed from S3 after simulation
- Local temp directories are purged
- `trigger_rule='all_done'` ensures cleanup runs even on failure

---

## 📸 Screenshots

### DAG Workflow
<img src="img/graph.png" width="600" alt="DAG Graph">

### Model Training Output
<img src="img/train_model.png" width="600" alt="Training Results">

### Live Monitoring Simulation
<img src="img/monitoring.png" width="600" alt="Monitoring Output">

### Automated Cleanup
<img src="img/cleanup.png" width="600" alt="Cleanup Task">

---

## 📝 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Scala for Spark ML | Access to Spark MLlib with native performance; demonstrates polyglot engineering |
| Inline Scala in Python DAG | Single-file deployment for Airflow; no JAR build step needed |
| GMDH over deep learning | Interpretability required for infrastructure alerting |
| Recursive DAG for audit | Guarantees eventual consistency without external schedulers |
| KRaft mode Kafka | Modern, ZooKeeper-free setup |
| LocalStack for AWS | Full local development without cloud costs |

---

## 🔮 Possible Extensions

- [ ] Add adaptive layer depth (grow GMDH until RMSE plateaus)
- [ ] Replace inline Scala with SBT project + `spark-submit`
- [ ] Add Prometheus/Grafana for real-time metric visualization
- [ ] Integrate AWS SSM Parameter Store for secrets
- [ ] Add model versioning (champion-challenger pattern)
- [ ] Schema Registry for Kafka event validation

---

