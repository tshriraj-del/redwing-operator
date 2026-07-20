# RedWing Operator

FastAPI backend for the RedWing fraud prevention platform. Runs on port 8000 and serves the ML scoring engine, autonomous AI fraud agent, rule factory pipeline, LLM proxy, network graph API, and XAI engine.

It also carries three layers built on top of that scorer:

- an **ingestion pipeline**: a validating schema contract, a durable streaming transport with backpressure and a dead-letter queue, and three source connectors (JSONL file drop, SQL table polled by watermark, and an HMAC-authenticated webhook),
- an **Actor Intelligence layer** that answers who is acting and why (motive, offender lifecycle, victim scam-arc, mule witting-ness, first-party intent, vulnerability, loophole exploitation) and maps that to a proportionate response rather than a binary block,
- a **labeling substrate** that records every decision with its point-in-time features, keeps a monitored holdout so outcomes stay observable, captures adjudicated intent labels, and reports when a heuristic has enough ground truth to graduate into a trained model.

The actor layer runs on expert-set deterministic rules, not trained models. That is deliberate: the historical ledger never labelled intent, so the heuristics bootstrap the data that would train their replacement. See `core/graduation.py` for the gate that decides when that is worthwhile.

---

## Requirements

- Python 3.9+
- The ML backend at `~/pulseml_models/` - provides the trained models **and** the shared
  feature foundation (`features.py`, `graph_layer.py`). The operator imports these so it
  computes features identically to training, eliminating training-serving skew. Override
  the location with the `REDWING_MODELS_DIR` environment variable.

The operator prefers the retrained, skew-free model (`xgboost_retrained.pkl` +
`scaler_retrained.pkl`) when present, and falls back to the originals otherwise.

---

## Setup

```bash
pip install -r requirements.txt
```

Create an `.env` file in this directory:

```env
# LLM - used by Rule Factory and the /llm/proxy endpoint
ANTHROPIC_API_KEY=sk-ant-...

# Optional: switch to a different LLM provider for the proxy
# LLM_PROVIDER=openai        # openai | groq | mistral
# LLM_API_KEY=sk-...
# LLM_MODEL=gpt-4o

# Integration Hub - add credentials as you onboard each agency
# OFAC_API_KEY=
# FINCEN_API_KEY=
# FINCEN_ORG_ID=
# EWS_API_KEY=
# EWS_ORG_ID=
# THREATMETRIX_API_KEY=
# THREATMETRIX_ORG_ID=
# PLAID_CLIENT_ID=
# PLAID_SECRET=
# FBI_IC3_API_KEY=
```

Start the server:

```bash
python -m uvicorn main:app --port 8000 --reload
```

The autonomous SyntheticID agent starts automatically on startup (requires trained models in `~/pulseml_models/`).

---

## Endpoints

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System status, model info, transaction count |
| GET | `/patterns` | Full fraud pattern library (static + deployed rules) |

### Scoring

| Method | Path | Description |
|--------|------|-------------|
| POST | `/score` | Score a single transaction (XGBoost + rule engine) |
| GET | `/monitor/stream` | SSE stream of live transaction scoring |
| GET | `/alerts` | Recent high-confidence fraud alerts |

### Ingestion Pipeline

Schema contract, durable transport, and source connectors. Every entry point validates before
anything reaches the scorer: a non-numeric or negative amount is a `422`, never a silent zero.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Score one transaction through the full cascade (schema-validated) |
| POST | `/ingest/batch` | Batch inject up to 1 000; rejects routed to a dead-letter list |
| GET | `/ingest/stats` | Injection health: ring buffer occupancy, log size |
| GET | `/ingest/schema` | The ingestion contract: required/recommended fields, rails, label-only fields |
| POST | `/stream/publish` | Validate and enqueue onto the durable transport (`429` under backpressure) |
| GET | `/stream/stats` | Transport health: ready backlog, processed, dead-letter, capacity |
| GET | `/stream/dead_letter` | Events that failed scoring past the retry limit |
| POST | `/stream/replay` | Replay dead-lettered events back to ready |
| GET | `/connectors` | Source connectors, durable checkpoints, authenticated webhook sources |
| POST | `/connectors/file/poll` | Poll the JSONL drop file (resumable, checkpointed) |
| POST | `/connectors/db/poll` | Incrementally poll a SQL table by watermark, with field mapping |
| POST | `/connectors/webhook/{source}` | Real-time push, HMAC-authenticated per source (`X-Signature`) |

### Actor Intelligence

Who is acting and why, and therefore the proportionate response. Runs on reported behavioural
telemetry only, never on the case typology (that would be leakage). With no telemetry the layer
stays silent, because motive cannot be inferred from an amount and a rail.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/telemetry` | Ingest behavioural telemetry for a subject (what a client SDK reports) |
| GET | `/actor/{subject}` | Offender read (motive, lifecycle, intervention) + victim read (scam arc, coercion-in-flight) |

### Training Substrate

Records every decision with its point-in-time features so the heuristics can eventually be
replaced by trained models, and reports when that is worthwhile.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/substrate/stats` | Decisions logged, enforced vs shadow, observed vs censored, label provenance |
| GET | `/substrate/readiness` | Per-target graduation verdict (heuristic-vs-gold agreement, Cohen's kappa) |

### Autonomous Agent (SyntheticID)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agent/status` | Agent state - running, blocked/flagged/allowed counts, uptime |
| GET | `/agent/events` | SSE fan-out - real-time block/flag/allow decisions (per-client queue) |
| POST | `/agent/start` | Start the agent (idempotent, guards model availability) |
| POST | `/agent/stop` | Gracefully stop the agent loop |
| GET | `/agent/config` | Current agent config (thresholds, toggles, speed) |
| PUT | `/agent/config` | Update config live - no restart needed |
| GET | `/agent/cases` | Case review queue; supports `?status=pending\|approved\|declined` |
| POST | `/agent/cases/{case_id}/resolve` | Approve or decline a flagged case (analyst override) |
| POST | `/agent/override/{tx_id}` | Direct action override on a specific transaction |

**Agent config schema:**
```json
{
  "block_threshold": 0.65,
  "flag_threshold": 0.45,
  "per_threat": {
    "card_testing_bot":        { "block": 0.60, "flag": 0.40, "enabled": true },
    "credential_stuffing":     { "block": 0.65, "flag": 0.45, "enabled": true },
    "ato_bot":                 { "block": 0.70, "flag": 0.50, "enabled": true },
    "synthetic_identity_farm": { "block": 0.70, "flag": 0.50, "enabled": true },
    "deepfake_bypass":         { "block": 0.80, "flag": 0.60, "enabled": true },
    "adversarial_ml":          { "block": 0.75, "flag": 0.55, "enabled": true }
  },
  "toggles": {
    "self_learning":         true,
    "auto_deploy_rules":     false,
    "high_alert_mode":       false,
    "zero_tolerance_bot":    false,
    "human_review_required": false
  },
  "speed": 0.25
}
```

### Network Graph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/network/graph` | Fraud ring graph - nodes and edges from transaction data |
| GET | `/network/typologies` | Distinct fraud typologies available for filtering |

### Rule Factory

| Method | Path | Description |
|--------|------|-------------|
| GET | `/rule-factory/gaps` | Transactions where ML fired but rules missed |
| POST | `/rule-factory/run` | Run the full pipeline: gap extraction → LLM rule generation → backtest → save |
| GET | `/rule-factory/rules` | All generated rules with status and backtest metrics |
| POST | `/rule-factory/deploy/{rule_id}` | Promote a shadow rule to deployed |
| POST | `/rule-factory/retire/{rule_id}` | Retire a rule |
| POST | `/rule-factory/test` | Backtest a candidate rule before saving |

### XAI Engine

| Method | Path | Description |
|--------|------|-------------|
| POST | `/xai/explain` | SHAP explanation for a transaction |
| GET | `/xai/governance` | Model drift + EU AI Act + SR 26-02 governance report |

### LLM Proxy

| Method | Path | Description |
|--------|------|-------------|
| POST | `/llm/proxy` | Routes LLM requests server-side - supports Anthropic, OpenAI, Groq, Mistral. API key never touches the browser. |
| POST | `/llm/stream` | Streaming variant of the LLM proxy (SSE) |

### Integration Hub

| Method | Path | Description |
|--------|------|-------------|
| GET | `/integrations/connectors` | List all 15 connectors with configuration status |
| GET | `/integrations/health` | Health check across all connectors |
| POST | `/integrations/enrich` | Enrich a transaction concurrently across selected connectors |
| POST | `/integrations/report` | File a SAR, CTR, or fraud referral to selected agencies |

---

## Integration Hub

The hub connects to external agencies and bureaus for transaction enrichment and regulatory reporting. All 15 connectors are registered but return `UNCONFIGURED` until credentials are added to `.env`.

**Credit Bureaus** - Equifax, Experian, TransUnion  
**Financial Intelligence** - FinCEN (SAR/CTR), OFAC SDN screening, FCA  
**Fraud Consortiums** - Early Warning Services, ThreatMetrix, NICE Actimize  
**Law Enforcement** - FBI IC3, INTERPOL, Europol EC3  
**Open Banking** - Plaid, Finicity, TrueLayer  

---

## Core modules

Everything in `core/` is pure Python over the store, deliberately free of the ML stack so it is
testable without loading models. 93 tests (`python3 tests/test_core_store.py`, `tests/test_redwing.py`).

| Layer | Modules | What it does |
|-------|---------|--------------|
| Substrate | `store.py`, `record.py`, `loop.py` | Entity/event backbone, decisions + labels, checkpoints, the closed loop |
| Ingestion | `ingest_schema.py`, `stream.py`, `connectors.py`, `webhook.py` | Schema contract, durable transport, file/SQL/webhook sources |
| Signals | `attributes.py`, `telemetry.py` | Device and identity attribute fabric; behavioural telemetry to actor tells |
| Actor Intelligence | `motive.py`, `scam_arc.py`, `mule_network.py`, `first_party.py`, `vulnerability.py`, `loophole.py`, `onboarding.py` | Who and why: offender motive, victim grooming arc, mule witting-ness, first-party intent, victimization risk, policy exploitation, onboarding gauntlet |
| Learning | `holdout.py`, `graduation.py`, `train.py`, `seed_substrate.py` | Monitored holdout, graduation gate, stdlib trainer, synthetic cohort |
| Decisioning | `liability.py`, `narrative.py`, `graph.py`, `consortium.py` | Liability pricing, scam narrative, fraud graph, DP consortium |

## Part of the RedWing Platform

| Repo | Role |
|------|------|
| [redwing-fraud-os](https://github.com/tshriraj-del/redwing-fraud-os) | React command center - dashboard, all analyst tools, SyntheticID Agent UI |
| [redwing-operator](https://github.com/tshriraj-del/redwing-operator) | This repo - FastAPI backend, ML scoring, autonomous agent, rule factory |
| [fraudsense](https://github.com/tshriraj-del/fraudsense) | Standalone LLM-powered fraud investigation copilot |
| [rulebreaker](https://github.com/tshriraj-del/rulebreaker) | Standalone adversarial rule stress-tester |
| [sar-writer](https://github.com/tshriraj-del/sar-writer) | Standalone FinCEN SAR narrative generator |
