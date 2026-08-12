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
| POST | `/score` | Score a single transaction (XGBoost + rule engine + novelty gate) |
| POST | `/score/payment` | Score against the real-label card model (ULB), calibrated |
| GET | `/payment/meta` | That model's provenance and honest headline metric |
| GET | `/monitor/stream` | SSE stream of live transaction scoring |
| GET | `/alerts` | Recent high-confidence fraud alerts, ranked by value at risk |
| GET | `/case/{transaction_id}` | The investigator case file for one transaction |
| POST | `/case` | Case file for an ad-hoc transaction body |
| POST | `/narrative` | Plain-language scam narrative for a decision |
| POST | `/authorization-iq` | Push-rail authorization signals (the AQF-equivalent pack) |
| GET | `/observability/skew` | Training-serving skew: the delta between offline and served features |

### The ML Layer

Six `pickle.load` calls used to sit in `main.py`, each with its own glue, and
`decisions.model_version` was set on **0 of 692 rows**. Workable with two models; not with the
five this is heading for.

`model_registry.py` is the only sanctioned load path, and it enforces four things rather than
documenting them:

| | |
|---|---|
| **contract** | a model declares the feature set it was fitted on; a mismatch refuses the load |
| **lifecycle** | champion / challenger / shadow / retired. Only a champion may affect a decision, and a challenger cannot become one because a caller passed the wrong flag |
| **version** | a content hash of the artifact bytes, so it cannot disagree with what actually scored |
| **fail-safe** | a model that will not load is reported and skipped; the decision path continues |

Risk tier is recorded per model (tier 1 moves money, tier 3 informs a human), so
`GET /model/inventory` answers *what is in production and how closely is it watched* without a
side spreadsheet that goes stale.

### Decision Policy

The score is signal; the policy is what the institution will actually do with it.
`liability.py` prices both sides and recommends the action the money supports;
`decision_policy.py` bounds that recommendation by a **floor** (the least we will do at this
risk on this rail) and a **ceiling** (the most we will do without a human), keyed by rail,
direction, score band and customer tier.

The policy never picks an action on its own. It only clamps a priced one, because a policy that
could decide would be a second risk opinion with no evidence behind it. A ceiling **may** soften
an action, since real policies do that, and when it does the decision carries
`policy_deescalated` and the rule responsible.

Every decision is stamped with a content hash of the policy table, which finally populates the
`decisions.policy_version` column that has existed since the substrate was built and was never
written. In the US the compliance target moves through litigation rather than rulemaking, so
attributing an outcome change to the policy live at the time is not bookkeeping, it is the only
way the change is explainable afterwards.

### Decline Contracts

A decline today is a dead end: the member sees "declined", the merchant sees `05 Do Not Honor`,
and a good customer is lost without the issuer learning it was wrong. False declines cost US
ecommerce more than card fraud does.

`decline_contract.py` makes a decline an object carrying four things a bare code does not: our
own **recoverability** judgement (`05` hides both a member who needs to verify and a card we
will never approve), the **price** of getting it wrong for this member, a **remediation
contract** at four disclosure levels, and an HMAC-bound, expiring **recovery token**.

The token is what makes varying disclosure safe. Without it, "verify and retry" is advice an
attacker can also follow; with it, the retry arrives carrying proof that the remediation was
performed, by the member it was issued to, inside its window. Tokens are bound to the member,
carry no reason text, and are never attached to the opaque disclosure level.

The module deliberately does **not** choose a level. That choice is a priced trade between
recovery uplift and information handed to an adversary, and it needs a causal estimate that does
not exist yet.

### The Novelty Gate

A supervised model is blind by construction to patterns unlike its training labels. An
unsupervised isolation forest (trained in the ML repo, calibrated there, loaded here) supplies
the one view XGBoost structurally lacks: *we have never seen this before*.

It is a **gate, not a blend**, because blending was measured to dilute the supervised model on
known fraud. XGBoost and the consortium speak first; the gate may then raise a score to the
alert line and no further, and may never lower one. An unsupervised detector saying "this is
unusual" is a reason to look, not a reason to be sure.

Measured on held-out test: recovers **751 of the 1,681 frauds XGBoost missed**, taking catch
from **11.2% to 50.9%** for **1.04%** of legitimate traffic sent to review. Reported as a
`novelty` block beside the score, never folded into it, so an analyst can see that a payment
arrived because it was unusual rather than because the model was confident.

If the artifact is missing, throws, or was trained on a different feature set, the gate declines
to load and supervised scoring continues untouched.

### The Closed Loop

An analyst disposition moves the recipient's reputation, writes gold labels, and returns a
receipt showing what that one decision bought.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/feedback` | Record a disposition (+ optional adjudicated intent and `effective_ts`) |
| GET | `/feedback/status` | Labelled totals, online updates applied, retrain queue depth |

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
| GET | `/substrate/graduation` | The full evidence chain: gate verdict, label counts, held-out model-vs-rule result |
| GET | `/substrate/next-questions` | Which case to adjudicate next, ranked by how much it moves the gate |
| GET | `/substrate/maturity` | Label arrival-lag curve and the maturity floor it implies |
| GET | `/adjudication/schema` | What an analyst can be asked when closing a case, and why each answer is wanted |

### Outcome Ledger

Where gold labels come from once a human clicking is no longer the only source. Of the five
sources the graduation gate trusts, only `analyst` previously had a live path; these are the
other four.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/outcomes` | Ingest chargebacks, recalls, confirmed losses, victim reports |
| GET | `/outcomes/stats` | Outcome supply by source, plus reversals and disagreements |
| GET | `/outcomes/disagreements` | Cases where two ground-truth sources disagree: labelled misses, with features attached |

Source precedence is enforced in `store.add_label()` rather than by convention, so a nightly
feed cannot overwrite a considered adjudication. A weaker source is still recorded, arriving
already superseded, because evidence should not be discarded merely because it lost.

### Model Performance

`/drift/status` computes PSI over distributions, which is label-free: it can say the input
moved and can never say the model decayed. This is the other half, and it separates the three
explanations for a bad-looking month because they demand opposite responses.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/model/performance` | `degraded` / `population_shift` / `unmeasurable` / `degraded_unconfirmed`, with the evidence |
| GET | `/drift/status` | PSI on score and feature distributions (input movement only) |
| POST | `/drift/reset` | Clear the rolling buffers |

Metrics are named `*_on_allowed`, never plain precision or recall, because outcomes exist only
where the payment was allowed: the frauds that were caught and blocked are exactly the ones
missing from the denominator. Where a holdout exists, the released sample estimates the blocked
population's fraud rate.

### Autonomous Agent (SyntheticID)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agent/status` | Agent state - running, blocked/flagged/allowed counts, uptime |
| GET | `/agent/events` | SSE fan-out - real-time block/flag/allow decisions (per-client queue) |
| POST | `/agent/start` | Start the agent (idempotent, guards model availability) |
| POST | `/agent/stop` | Gracefully stop the agent loop |
| GET | `/agent/config` | Current agent config (thresholds, toggles, speed) |
| PUT | `/agent/config` | Update config live - no restart needed |
| POST | `/syntheticid/ingest` | Accept a SyntheticID Lab result; bypassed attacks become rule-factory input |
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
| GET | `/graph/stats` | Ring and component topology over the backbone |
| GET | `/gnn/stats` | Graph-feature summary used by the scorer |
| GET | `/backbone/stats` | Entity and event counts on the durable backbone |
| GET | `/backbone/recent` | Most recent events across the graph |
| GET | `/backbone/graph` | Backbone-derived subgraph |
| GET | `/backbone/liability` | Reimbursement dollars currently exposed |
| GET | `/backbone/entity/{entity_id}` | One entity: reputation, events, linked counterparties |

### Consortium (n=2, differentially private)

Two synthetic member institutions with genuinely different customers, rails and payee exposure.
Members share only clamped, differentially private aggregates; the privacy budget is split
across the statistics released rather than spent per statistic.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/consortium/demo` | The n=2 network view and what it adds over one bank's own book |
| GET | `/consortium/recipient/{recipient_id}` | Pooled view of one payee across members |
| GET | `/consortium/mules` | Payees the pool flags that no single member would |
| GET | `/privacy/curve` | Epsilon-versus-utility curve for the shared aggregates |

### Verifiable Agent Environment

Fraud scenarios with programmatic verifiers, so an agent is graded on whether it actually
investigated rather than on whether its answer reads well.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/env/spec` | The scenario contract and available verifiers |
| POST | `/env/run` | Run one scenario end to end |
| POST | `/env/run-all` | Run the whole suite |
| POST | `/env/step` | Single-step an episode |
| POST | `/env/agent` | Run the LLM investigator, graded by the same verifiers (`compare: true` ranks it against the reference policies) |
| GET | `/adversary/strategies` | Evasion strategies the red-team simulator can apply |
| POST | `/adversary/simulate` | Attack a rule or model and report what survives |

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
| GET | `/xai/explain/{transaction_id}` | The stored explanation for one scored transaction |
| GET | `/xai/explanations` | Recent explanations with their feature attributions |
| GET | `/xai/model-card` | Model card: training data, validation regime, known limitations |
| GET | `/xai/governance` | Model drift + EU AI Act + SR 26-02 governance report |

### LLM Proxy

| Method | Path | Description |
|--------|------|-------------|
| POST | `/llm/proxy` | Routes LLM requests server-side - supports Anthropic, OpenAI, Groq, Mistral. API key never touches the browser. |

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
testable without loading models. **242 tests across 12 files**; run any of them directly
(`python3 tests/test_core_store.py`) or the lot under pytest. One file (`test_live_path.py`)
needs numpy and so wants the venv.

| Layer | Modules | What it does |
|-------|---------|--------------|
| Substrate | `store.py`, `record.py`, `loop.py` | Entity/event backbone, decisions + labels, checkpoints, the closed loop |
| Ingestion | `ingest_schema.py`, `stream.py`, `connectors.py`, `webhook.py` | Schema contract, durable transport, file/SQL/webhook sources |
| Signals | `attributes.py`, `telemetry.py` | Device and identity attribute fabric; behavioural telemetry to actor tells |
| Actor Intelligence | `motive.py`, `scam_arc.py`, `mule_network.py`, `mule_behaviour.py`, `first_party.py`, `vulnerability.py`, `loophole.py`, `onboarding.py` | Who and why: offender motive, victim grooming arc, mule witting-ness from observable tells, first-party intent, victimization risk, policy exploitation, onboarding gauntlet |
| Label supply | `outcome_ledger.py`, `label_maturity.py`, `backfill_outcome_labels.py` | Outcomes from chargebacks and recalls with source precedence; arrival-lag curve and maturity floor; recovering the machine's call onto historical decisions |
| Learning | `holdout.py`, `graduation.py`, `train.py`, `seed_substrate.py`, `active_learning.py` | Monitored holdout, graduation gate, stdlib trainer, synthetic cohort, next-best-label queue |
| Agents | `investigator_agent.py` | An LLM investigator driven through `fraud_env.py` and graded by its verifiers, never by itself |
| ML layer | `model_registry.py` | Model inventory, feature contracts, champion/challenger lifecycle, content-hash versions |
| Measurement | `model_performance.py` | Did the model decay, did the population shift, or have the labels not arrived |
| Tooling | `adjudication.py`, `replay.py`, `phase2_report.py`, `seed_from_csv.py`, `seed_consortium_demo.py` | Adjudication vocabularies, the replay harness, the evidence report, and the seeders |
| Decisioning | `decline_contract.py`, `decision_policy.py`, `liability.py`, `narrative.py`, `graph.py`, `consortium.py`, `authorization_iq.py`, `sar_draft.py` | Liability pricing, scam narrative, fraud graph, DP consortium, push-rail authorization signals, SAR drafting behind a grounding gate |

## Part of the RedWing Platform

| Repo | Role |
|------|------|
| [redwing-fraud-os](https://github.com/tshriraj-del/redwing-fraud-os) | React command center - dashboard, all analyst tools, SyntheticID Agent UI |
| [redwing-operator](https://github.com/tshriraj-del/redwing-operator) | This repo - FastAPI backend, ML scoring, autonomous agent, rule factory |
| [fraudsense](https://github.com/tshriraj-del/fraudsense) | Standalone LLM-powered fraud investigation copilot |
| [rulebreaker](https://github.com/tshriraj-del/rulebreaker) | Standalone adversarial rule stress-tester |
| [sar-writer](https://github.com/tshriraj-del/sar-writer) | Standalone FinCEN SAR narrative generator |
