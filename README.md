# GMDH Predictive Engine — Adaptive Fraud & Infrastructure Intelligence

A **self-learning monitoring and predictive system** for cloud architecture efficiency. The project demonstrates an end-to-end MLOps pipeline — from synthetic data generation and model training to real-time simulation and automated resource cleanup — orchestrated entirely by Apache Airflow.

> **Core idea:** A self-learning platform where GMDH polynomial acts as a mathematical leash on LLM (Bedrock) — the model automatically reduces trust in unreliable AI signals through coefficient evolution, while gating all business decisions on verified system health.

---

##  Why This Project Exists

This is a portfolio project that showcases:

- **MLOps lifecycle** — data generation → training → model persistence → inference → cleanup
- **Transparent ML** — GMDH produces an interpretable polynomial, not a black-box prediction
- **Data pipeline engineering** — Kafka ingestion, MySQL sync, DLQ handling, recursive reconciliation
- **Infrastructure as Code** — fully Dockerized, reproducible with a single `docker-compose up`
- **Production patterns** — idempotency, dead letter queues, self-healing DAGs, parallel processing
- **Connected architecture** — all DAGs are linked into a single feedback-driven ecosystem

---

##  Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION (Airflow)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  kafka_event_generator (*/5 min)                                    │
│       │ writes subscription events to Kafka                         │
│       ▼                                                             │
│  kafka_lag_monitor (*/3 min)                                        │
│       │ compares Kafka watermark vs MySQL count                     │
│       │ gap > threshold? → triggers ↓                               │
│       ▼                                                             │
│  marketplace_audit (recursive, self-healing)                        │
│       │ parallel consume (3 partitions) → MySQL                     │
│       │ bad JSON → DLQ (Kafka + MySQL)                              │
│       │ validates → gap still > 0? → triggers itself                │
│       │                                                             │
│  ─────┼──────────────────────────────────────────────────────────   │
│       │                                                             │
│  market_transaction_generator (*/10 min)                            │
│       │ generates SP-API + Cybersource events → Kinesis             │
│       │ notifies system-monitor topic                               │
│       ▼                                                             │
│  fraud_detection_engine (auto-triggered)                            │
│       │ Bedrock (mock) → semantic feature extraction                │
│       │ Python GMDH → trains fraud model (Model A) + health model (Model B)  │
│       │ check_system_health() → queries MySQL sync state            │
│       │   └─ connects to data integrity layer                       │
│       │   └─ if system degraded → DISABLE inference (fallback)      │
│       ▼                                                             │
│  fraud inference: applies polynomial → BLOCK / ALLOW                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The system monitors **3 levels of health** that feed into each other:

1. **Data integrity** — guarantees zero data loss between Kafka and MySQL
2. **Infrastructure efficiency** — ML model predicts system health from latency, auth status, and CPU load
3. **Business logic (Fraud)** — ML model scores transactions, but only when infrastructure is healthy

---

##  How DAGs Connect

| Source DAG | Trigger | Target DAG | Connection Type |
|-----------|---------|-----------|-----------------|
| `kafka_event_generator` | schedule (*/5 min) | — | Produces Kafka events |
| `kafka_lag_monitor` | detects gap > 5 | `marketplace_audit` | `trigger_dag()` |
| `marketplace_audit` | gap still > 0 | `marketplace_audit` | Recursive self-trigger |
| `market_transaction_generator` | after success | `fraud_detection_engine` | `TriggerDagRunOperator` |
| `fraud_detection_engine` | health check task | reads `raw_subscriptions` | MySQL query (cross-DAG data dependency) |

### Fraud Detection Engine (internal flow)

```
enrich_with_bedrock
        |
    +---+---+
    |       |
    v       v
  train   train
  Model A Model B
  (fraud) (health)
    |       |
    +---+---+
        |
        v
check_system_health  <-- reads model_b_health.json (produced by Model B)
        |
        v
run_fraud_inference  <-- uses Model A coefficients, gated by Model B score
        |
        v
    cleanup
```

Model A and Model B train **in parallel**. Model B produces `health_score`.
If `health_score < 0.45` then Model A inference is **disabled** (fallback mode).

### Feedback Loop

```
Transactions generated --> Fraud model trained --> Health checked -->
    If system unhealthy --> Disable fraud scoring -->
        Wait for marketplace_audit to fix sync -->
            System recovers --> Re-enable fraud scoring
```

---

##  The GMDH Algorithm

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
   │node_x1_x2│ │node_x1_x3│ │node_x2_x3│   ← Layer 1: all C(n,2) pairs
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
           output score (0–1)
```

Each neuron computes: `ŷ = β₀ + β₁·xᵢ + β₂·xⱼ + β₃·(xᵢ·xⱼ)`

The final deployed model is a **4th-order polynomial** with fully interpretable coefficients.

---

##  Dual-Model Architecture (Fraud + Health)

The system runs **two GMDH models** trained with the same algorithm on different domains:

```
┌─────────────────────────────────────────────────────────┐
│                 GMDH ENGINE (same algorithm)            │
├──────────────────────────┬──────────────────────────────┤
│   MODEL A: Fraud         │   MODEL B: System Health     │
│                          │                              │
│   Inputs:                │   Inputs:                    │
│   • semantic_risk        │   • cpu_load                 │
│     (from Bedrock LLM)   │   • api_latency              │
│   • velocity_1h          │   • auth_status              │
│   • proxy_score          │                              │
│   • amount_deviation     │                              │
│                          │                              │
│   Output: fraud_prob     │   Output: health_score       │
│   Action: Block/Allow    │   Action: Scale/Alert        │
├──────────────────────────┴──────────────────────────────┤
│                    FALLBACK LOGIC                       │
│   If Model B health < 0.45 → disable Model A inference  │
│   (degraded system = unreliable fraud predictions)      │
└─────────────────────────────────────────────────────────┘
```

### Why Two Models, Not One?

- **Isolation** — CPU spike ≠ fraud. Separate models prevent false positives
- **Different cadence** — fraud = milliseconds, system health = minutes
- **Auditability** — regulators want separate audit trail for fraud decisions
- **Independent retraining** — if Bedrock drifts, only Model A retrains

### Bedrock Integration

Amazon Bedrock acts as a **feature extractor**, not a decision-maker:
1. Bedrock receives raw transaction text
2. Returns `semantic_risk` score (0–1)
3. GMDH uses this as one input alongside numeric metrics
4. If Bedrock's scores prove unreliable (detected via reconciliation), GMDH reduces the coefficient weight automatically on retrain

### Output Thresholds

**Model A (Fraud):**

| Score | Decision | Action |
|-------|----------|--------|
| > 0.55 |  BLOCK | Transaction rejected |
| ≤ 0.55 |  ALLOW | Transaction proceeds |

**Model B (Health):**

| Efficiency | Status | Action |
|-----------|--------|--------|
| > 75% |  OK | No action |
| 45–75% |  WARN | Investigate |
| < 45% |  CRITICAL | Disable Model A, fallback mode |

---

##  Data Integrity Pipeline

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
                              trigger self    done 
                              (recursion)
```

Key patterns:
- **Idempotent replay** — always scans from offset 0, `UNIQUE` constraint prevents duplicates
- **Dual DLQ** — errors persist in both Kafka topic and MySQL table
- **Self-healing** — recursive trigger until convergence
- **Cross-DAG dependency** — `fraud_detection_engine` queries `raw_subscriptions` count to estimate system health

---

##  Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Apache Airflow 2.10.5 | DAG scheduling, task dependencies, recursive triggers |
| Processing | Apache Spark (Scala) + Python | GMDH model training (Scala for Spark, Python for containerized) |
| Streaming | Apache Kafka (KRaft, Confluent 7.6) | Event ingestion, DLQ |
| Storage | AWS S3 (via S3A) + MySQL 8.0 | Model artifacts, operational data |
| Cloud Emulation | LocalStack 2.3.2 | Kinesis & S3 for local development |
| LLM Integration | Amazon Bedrock (mock) | Semantic feature extraction |
| Containerization | Docker Compose | Full environment in one command |
| Code Quality | Pylint + AWS Athena | Historical score tracking |

---

##  Project Structure

```
gmdh-predictive-engine/
├── dags/
│   ├── fraud_detection_dag.py        # Dual-model DAG: Bedrock → train → fallback → inference
│   ├── kafka_event_generator.py      # Produces random subscription events to Kafka
│   ├── kafka_queue_monitor.py        # Lag detection → triggers audit if gap found
│   ├── market_transaction_generator.py # SP-API + Cybersource → Kinesis → triggers fraud engine
│   └── marketplace_audit.py          # Recursive Kafka→MySQL sync with DLQ
├── dags_backup/
│   └── gmdh_predictive_engine_it.py  # Core ML DAG: Model B (train → simulate → cleanup)
├── jobs/
│   ├── bedrock_extractor.py          # Bedrock LLM semantic feature extractor (mock)
│   ├── gmdh_fraud_trainer.py         # Python GMDH trainer for Model A (fraud, 2-layer)
│   ├── gmdh_fraud_trainer.scala      # Scala reference implementation (Model A)
│   ├── gmdh_health_trainer.py        # Python GMDH trainer for Model B (system health)
│   ├── airflow_pylint_qa/            # Pylint analysis framework
│   └── marketplace_audit/            # Integration tests
├── scripts/
│   ├── generate_it_dataset.py        # Synthetic infra dataset (10K records)
│   └── generate_fraud_dataset.py     # Synthetic fraud dataset (5K records)
├── data/
│   ├── fintech_transactions_raw.csv  # Model B training data
│   ├── fraud_transactions.csv        # Model A training data
│   └── enriched_transaction.json     # Sample Bedrock-enriched events
├── kafka/
│   └── docker-compose.yaml           # Kafka standalone config (reference)
├── docker-compose.yaml               # ALL services: MySQL + Kafka + LocalStack + Airflow
├── Dockerfile                        # Custom Airflow image with Kafka client
└── .gitignore
```

---

##  Quick Start

### Prerequisites

- Docker Desktop / Rancher Desktop
- Python 3.9+ (for dataset generation only)

### 1. Start Everything

```bash
docker-compose up -d
```

This starts **all services** in one command:
- MySQL 8.0 (Airflow metadata + operational data)
- Kafka (KRaft mode, single-node)
- LocalStack (S3 + Kinesis emulation)
- Airflow (webserver + scheduler)

Services start in dependency order: MySQL → Kafka → LocalStack → Airflow.

### 2. Create Kafka Topics

```bash
docker exec gmdh-kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic raw-subscriptions --partitions 3 --replication-factor 1

docker exec gmdh-kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic subscriptions_dlq --partitions 1 --replication-factor 1

docker exec gmdh-kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic system-monitor --partitions 1 --replication-factor 1
```

### 3. Generate Training Data

```bash
pip install pandas numpy
python scripts/generate_it_dataset.py
python scripts/generate_fraud_dataset.py
```

### 4. Access Airflow

Open `http://localhost:8080`

Set admin password:
```bash
docker exec gmdh-airflow airflow users reset-password --username admin --password admin
```

### 5. Activate DAGs

```bash
docker exec gmdh-airflow airflow dags unpause kafka_event_generator
docker exec gmdh-airflow airflow dags unpause kafka_lag_monitor
docker exec gmdh-airflow airflow dags unpause marketplace_audit
docker exec gmdh-airflow airflow dags unpause market_transaction_generator
docker exec gmdh-airflow airflow dags unpause fraud_detection_engine
```

### 6. Watch It Work

Once activated, the system runs autonomously:
1. `kafka_event_generator` produces events every 5 min
2. `kafka_lag_monitor` detects gap every 3 min → triggers `marketplace_audit`
3. `marketplace_audit` syncs recursively until gap = 0
4. `market_transaction_generator` sends to Kinesis every 10 min → triggers `fraud_detection_engine`
5. `fraud_detection_engine` checks system health → runs fraud inference

---

##  Clean Data Lake Principles

The pipeline ensures **zero-footprint execution**:

- Temporary Scala source files are deleted after runtime
- Model artifacts are removed from S3 after simulation
- Local temp directories are purged
- `trigger_rule='all_done'` ensures cleanup runs even on failure

---

##  Screenshots

### DAG: Fraud Detection Engine
<img src="img/fraud_detection_dag.png" width="600" alt="Fraud Detection DAG">

### DAG: Kafka Event Generator
<img src="img/kafka_event_generator.png" width="600" alt="Kafka Event Generator DAG">

### DAG: Kafka Lag Monitor
<img src="img/kafka_lag_monitor.png" width="600" alt="Kafka Lag Monitor DAG">

### DAG: Marketplace Audit
<img src="img/marketplace_audit.png" width="600" alt="Marketplace Audit DAG">

### DAG: Market Transaction Generator
<img src="img/market_transaction_generator.png" width="600" alt="Market Transaction Generator DAG">

### Model Training Output
<img src="img/train_model.png" width="600" alt="Training Results">

### Live Monitoring Simulation
<img src="img/monitoring.png" width="600" alt="Monitoring Output">

### Automated Cleanup
<img src="img/cleanup.png" width="600" alt="Cleanup Task">

---

##  Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single docker-compose | One command to start everything; no external dependencies |
| Scala for Spark ML | Access to Spark MLlib with native performance; polyglot engineering |
| GMDH over deep learning | Interpretability required for infrastructure alerting |
| Recursive DAG for audit | Guarantees eventual consistency without external schedulers |
| Dual-model with fallback | System health gates fraud decisions; prevents unreliable predictions |
| Bedrock as feature extractor | LLM enriches data but doesn't make decisions; GMDH stays in control |
| KRaft mode Kafka | Modern, ZooKeeper-free setup |
| LocalStack for AWS | Full local development without cloud costs |
| Cross-DAG triggers | Creates a connected ecosystem, not isolated scripts |

---

##  Production Architecture Reference

This project simulates a real-world **Adaptive Closed-Loop Fraud Detection** system. Here's how the production cycle works:

```
┌───────────────────────────────────────────────────────────────────┐
│                    24-HOUR EVOLUTION CYCLE                        │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  REAL-TIME (Fast Path, <100ms):                                   │
│    Transaction arrives                                            │
│      → Trust registered user (session, device fingerprint)        │
│      → Cybersource decision (ACCEPT/REJECT/REVIEW)                │
│      → If both OK → ALLOW instantly                               │
│      → Shadow-write event to archive (Kafka/Kinesis)              │
│                                                                   │
│  PARALLEL (Background, continuous):                               │
│    GMDH model trains on accumulating data                         │
│    Formula coefficients evolve as patterns shift                  │
│    New fraud vectors → new interaction terms gain weight          │
│                                                                   │
│  NIGHTLY (Reconciliation, batch):                                 │
│    Amazon SP-API sync                                             │
│      → Compare "what we allowed" vs "actual chargebacks/refunds"  │
│      → Identify false negatives (fraud we missed)                 │
│      → Small dataset (~0.1% of traffic)                           │
│                                                                   │
│  TARGETED ENRICHMENT (cost-efficient):                            │
│    Send ONLY bad cases to Bedrock                                 │
│      → LLM explains WHY it was fraud (semantic analysis)          │
│      → Extracts semantic_risk feature for retraining              │
│      → 99.9% of traffic never touches Bedrock (cost savings)      │
│                                                                   │
│  MODEL EVOLUTION:                                                 │
│    Retrain GMDH with enriched data                                │
│      → Coefficients shift (old patterns lose weight)              │
│      → New interaction terms emerge                               │
│      → Hot-reload model to production (JSON on S3)                │
│                                                                   │
│  REPEAT → system gets smarter every night                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### How GMDH "Feels" Fraud Pattern Changes

GMDH doesn't detect drift through heuristics — it detects it **mathematically**:

1. **Coefficient Drift** — When fraudsters change tactics (e.g., from high-velocity to proxy-based), the nightly retrain shifts `β` weights automatically. Old indicators lose influence, new ones gain it.

2. **Node Selection** — GMDH builds all C(n,2) pairwise nodes and selects winners by RMSE. If a fraud pattern dies, its node shows higher error on validation data and gets replaced.

3. **Bedrock as Leash** — GMDH controls how much to trust the LLM. If Bedrock's `semantic_risk` scores prove unreliable (detected via reconciliation), the `β_semantic` coefficient shrinks on retrain. The polynomial mathematically "leashes" the LLM.

### What This Project Demonstrates

| Production Component | Project Implementation |
|---------------------|------------------------|
| Real-time scoring | `fraud_detection_engine` → inference task |
| Cybersource integration | `market_transaction_generator` → paired events |
| SP-API reconciliation | `kafka_lag_monitor` + `marketplace_audit` |
| Bedrock enrichment | `bedrock_extractor.py` (mock) |
| GMDH evolution | `gmdh_fraud_trainer.py` → nightly retrain |
| Model hot-reload | JSON coefficients on filesystem/S3 |
| System health gating | `check_system_health` → fallback logic |

---

##  Possible Extensions

- [ ] Add adaptive layer depth (grow GMDH until RMSE plateaus)
- [ ] Replace inline Scala with SBT project + `spark-submit`
- [ ] Add Prometheus/Grafana for real-time metric visualization
- [ ] Integrate AWS SSM Parameter Store for secrets
- [ ] Add model versioning (champion-challenger pattern)
- [ ] Schema Registry for Kafka event validation
- [ ] Real Bedrock API integration (replace mock)
- [ ] Flink streaming inference (replace batch simulation)
- [ ] Add alerting (SNS/Slack) when Model B health drops below threshold

---

##  License As IS
