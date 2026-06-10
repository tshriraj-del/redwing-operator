"""
SyntheticID Autonomous Agent — ML-backed real-time AI fraud detection.

Runs as a FastAPI background task. On each tick:
  1. Scores a transaction via build_event() (real XGBoost inference)
  2. Detects AI-specific behavioral signals (bots, synthetic IDs, deepfakes)
  3. Makes an autonomous block/flag/allow decision
  4. Creates a Case Review entry for flagged/escalated transactions
  5. Buffers novel attack patterns → triggers Rule Factory self-learning

Config is persisted at ~/pulseml_models/agent_config.json and applied live
without restart — PUT /agent/config takes effect on the next tick.
"""

import asyncio
import copy
import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / "pulseml_models" / "agent_config.json"

DEFAULT_CONFIG: dict = {
    "block_threshold": 0.65,
    "flag_threshold":  0.45,
    "per_threat": {
        "card_testing_bot":        {"block": 0.60, "flag": 0.40, "enabled": True},
        "credential_stuffing":     {"block": 0.65, "flag": 0.45, "enabled": True},
        "ato_bot":                 {"block": 0.70, "flag": 0.50, "enabled": True},
        "synthetic_identity_farm": {"block": 0.70, "flag": 0.50, "enabled": True},
        "deepfake_bypass":         {"block": 0.80, "flag": 0.60, "enabled": True},
        "adversarial_ml":          {"block": 0.75, "flag": 0.55, "enabled": True},
    },
    "toggles": {
        "self_learning":         True,
        "auto_deploy_rules":     False,
        "high_alert_mode":       False,
        "zero_tolerance_bot":    False,
        "human_review_required": False,
    },
    "speed":             0.25,
    "novel_buffer_size": 10,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    try:
        raw = json.loads(CONFIG_PATH.read_text())
        return _deep_merge(DEFAULT_CONFIG, raw)
    except Exception:
        return copy.deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    tmp.replace(CONFIG_PATH)


def validate_config(cfg: dict) -> dict:
    """Deep-merge with defaults and clamp numeric values. Returns validated copy."""
    merged = _deep_merge(DEFAULT_CONFIG, cfg)
    merged["block_threshold"] = max(0.01, min(0.99, float(merged["block_threshold"])))
    merged["flag_threshold"]  = max(0.01, min(0.99, float(merged["flag_threshold"])))
    merged["speed"]           = max(0.10, min(10.0, float(merged["speed"])))
    merged["novel_buffer_size"] = max(1, min(100, int(merged["novel_buffer_size"])))
    for t, thresholds in merged["per_threat"].items():
        thresholds["block"]   = max(0.01, min(0.99, float(thresholds.get("block", 0.65))))
        thresholds["flag"]    = max(0.01, min(0.99, float(thresholds.get("flag",  0.45))))
        thresholds["enabled"] = bool(thresholds.get("enabled", True))
    for k, v in merged["toggles"].items():
        merged["toggles"][k] = bool(v)
    return merged


# Mutable singleton — mutated in place by PUT /agent/config so run_agent
# picks up changes on the next tick without restart.
agent_config: dict = load_config()

# ── Threat metadata ───────────────────────────────────────────────────────────

THREAT_META: dict = {
    "card_testing_bot":        {"label": "Card Testing Bot",       "color": "#22c55e"},
    "synthetic_identity_farm": {"label": "Synthetic ID Farm",      "color": "#f59e0b"},
    "ato_bot":                 {"label": "ATO Bot",                "color": "#c084fc"},
    "deepfake_bypass":         {"label": "Deepfake Bypass",        "color": "#38bdf8"},
    "adversarial_ml":          {"label": "Adversarial ML Attack",  "color": "#ef4444"},
    "credential_stuffing":     {"label": "Credential Stuffing",    "color": "#f97316"},
    "clean":                   {"label": "Clean",                  "color": "#22c55e"},
}

# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    running:          bool    = False
    blocked_count:    int     = 0
    flagged_count:    int     = 0
    allowed_count:    int     = 0
    patterns_learned: int     = 0
    start_time:       object  = None   # datetime | None
    recent_events:    deque   = field(default_factory=lambda: deque(maxlen=200))
    case_queue:       deque   = field(default_factory=lambda: deque(maxlen=100))


agent_state: AgentState      = AgentState()
_event_subscribers: set      = set()     # one asyncio.Queue per SSE client
novel_attack_buffer: list    = []

# ── AI signature detection ────────────────────────────────────────────────────

AI_SIGNATURES: dict = {
    "timing_regularity":     {"weight": 0.30, "desc": "Sub-100ms velocity → automated timing"},
    "micro_amount_sequence": {"weight": 0.25, "desc": "Micro-amounts ($0.01–$1.99) → card testing"},
    "headless_device":       {"weight": 0.20, "desc": "Unknown/synthetic device fingerprint"},
    "ip_reputation":         {"weight": 0.15, "desc": "High-risk rail + elevated ML score"},
    "identity_clone":        {"weight": 0.10, "desc": "Card testing / synthetic identity pattern hit"},
}


def detect_ai_signature(tx: dict) -> dict:
    """
    Detect AI/bot-specific behavioral signals in a scored transaction event.
    These supplement the financial signals in patterns.py.
    Returns {is_bot, is_synthetic, confidence, signals[]}.
    """
    signals_hit: list = []
    confidence: float = 0.0

    ml_score        = tx.get("ml_score", 0.0)
    amount          = tx.get("amount", 0.0)
    rail            = tx.get("rail", "")
    top_pattern     = tx.get("top_pattern") or ""
    matched_signals = tx.get("matched_signals", [])
    matched_labels  = " ".join(
        (s.get("label", "") if isinstance(s, dict) else str(s)).lower()
        for s in matched_signals
    )

    # Signal 1: timing regularity — velocity/automated keywords in matched signals
    if any(kw in matched_labels for kw in ("velocity", "automated", "high velocity", "inter_tx")):
        signals_hit.append("timing_regularity")
        confidence += AI_SIGNATURES["timing_regularity"]["weight"]

    # Signal 2: micro-amount card testing
    if amount < 2.00 and ml_score > 0.50:
        signals_hit.append("micro_amount_sequence")
        confidence += AI_SIGNATURES["micro_amount_sequence"]["weight"]

    # Signal 3: headless/synthetic device
    if any(kw in matched_labels for kw in ("bot", "unknown device", "device", "headless")):
        signals_hit.append("headless_device")
        confidence += AI_SIGNATURES["headless_device"]["weight"]

    # Signal 4: high-risk rail + elevated ML score → IP/proxy heuristic
    if rail in ("crypto", "wire", "zelle", "fednow") and ml_score > 0.60:
        signals_hit.append("ip_reputation")
        confidence += AI_SIGNATURES["ip_reputation"]["weight"]

    # Signal 5: known AI-attack pattern already matched
    if any(kw in top_pattern.lower() for kw in ("card testing", "synthetic identity", "ato")):
        signals_hit.append("identity_clone")
        confidence += AI_SIGNATURES["identity_clone"]["weight"]

    is_bot       = confidence >= 0.40 and "timing_regularity" in signals_hit
    is_synthetic = "identity_clone" in signals_hit or "AI Synthetic" in top_pattern

    return {
        "is_bot":       is_bot,
        "is_synthetic": is_synthetic,
        "confidence":   round(min(confidence, 1.0), 4),
        "signals":      signals_hit,
    }


def classify_threat(tx: dict, ml_score: float, ai_sig: dict) -> str:
    """Map a scored transaction to one of 7 threat type string keys."""
    top_pattern = tx.get("top_pattern") or ""
    amount      = tx.get("amount", 0.0)
    ai_signals  = ai_sig.get("signals", [])
    is_synthetic = ai_sig.get("is_synthetic", False)
    ai_conf     = ai_sig.get("confidence", 0.0)

    # Card testing: micro amount + card testing sequence signal
    if amount < 2.00 and "micro_amount_sequence" in ai_signals:
        return "card_testing_bot"

    # Credential stuffing: headless + timing + modest amount + ML signal
    if (
        "headless_device" in ai_signals
        and "timing_regularity" in ai_signals
        and amount < 500
        and ml_score > 0.55
    ):
        return "credential_stuffing"

    # ATO bot: ATO pattern OR headless device + large amount
    if "ATO" in top_pattern or ("headless_device" in ai_signals and amount > 1000):
        return "ato_bot"

    # Synthetic identity farm: pattern match OR synthetic flag + strong ML
    if "Synthetic Identity" in top_pattern or (is_synthetic and ml_score > 0.65):
        return "synthetic_identity_farm"

    # Deepfake bypass: large amount + no headless device (authorized push) + strong ML
    if "Deepfake" in top_pattern or (amount > 5000 and "headless_device" not in ai_signals and ml_score > 0.70):
        return "deepfake_bypass"

    # Adversarial ML: ML fires strongly but no pattern/AI signal matches
    if ml_score > 0.80 and not top_pattern and ai_conf < 0.20:
        return "adversarial_ml"

    # Clean or weak signal
    return "clean"


def autonomous_decision(
    threat_type: str,
    combined_score: float,
    ai_sig: dict,
    config: dict,
    graph_risk: float = 0.0,
) -> dict:
    """
    Make a block/flag/allow decision using per-threat thresholds from config.
    Config changes (via PUT /agent/config) apply immediately on next tick.

    Effective score blends three signals (4-tier cascade, Tier 1+3):
      60% XGBoost combined score — primary ML signal
      25% AI behavioral confidence — bot/synthetic ID patterns
      15% graph risk score — ring membership, shared device, recipient fraud rate
    """
    per_threat = config["per_threat"].get(threat_type, {"block": 0.65, "flag": 0.45, "enabled": True})
    toggles    = config["toggles"]

    # Threat type disabled by analyst → always allow
    if not per_threat.get("enabled", True):
        return {"action": "allow", "confidence": 0.0, "reason": f"{threat_type} detection disabled by analyst", "escalate_human": False}

    block_thr = per_threat["block"]
    flag_thr  = per_threat["flag"]

    # Mode overrides
    if toggles.get("high_alert_mode"):
        block_thr = max(0.01, block_thr - 0.10)
        flag_thr  = max(0.01, flag_thr  - 0.10)
    if toggles.get("zero_tolerance_bot") and threat_type == "card_testing_bot":
        block_thr = 0.30

    # Blend: XGBoost (60%) + AI behavioral (25%) + graph context (15%)
    ai_conf         = ai_sig.get("confidence", 0.0)
    effective_score = round(combined_score * 0.60 + ai_conf * 0.25 + graph_risk * 0.15, 4)

    if effective_score >= block_thr:
        action    = "block"
        reason    = f"{THREAT_META.get(threat_type, {}).get('label', threat_type)} detected — score {effective_score:.2f} ≥ block threshold {block_thr:.2f}"
        escalate  = effective_score >= 0.90 or threat_type in ("deepfake_bypass", "adversarial_ml")
    elif effective_score >= flag_thr:
        action    = "flag"
        reason    = f"{THREAT_META.get(threat_type, {}).get('label', threat_type)} suspected — score {effective_score:.2f}, monitoring"
        escalate  = False
    else:
        action    = "allow"
        reason    = f"Below threat thresholds (score {effective_score:.2f})"
        escalate  = False

    return {
        "action":         action,
        "confidence":     effective_score,
        "reason":         reason,
        "escalate_human": escalate,
    }


# ── Case queue helpers ────────────────────────────────────────────────────────

def _should_create_case(action: str, escalate: bool, config: dict) -> bool:
    if action == "flag":
        return True
    if action == "block" and config["toggles"].get("human_review_required"):
        return True
    if escalate:
        return True
    return False


def _make_case(tx: dict, ai_sig: dict, decision: dict, threat_type: str) -> dict:
    meta = THREAT_META.get(threat_type, {"label": threat_type, "color": "#64748b"})
    return {
        "case_id":        uuid.uuid4().hex[:12],
        "transaction_id": tx.get("transaction_id", ""),
        "created_at":     datetime.utcnow().isoformat() + "Z",
        "status":         "pending",
        "agent_action":   decision["action"],
        "threat_type":    threat_type,
        "threat_label":   meta["label"],
        "threat_color":   meta["color"],
        "combined_score": tx.get("combined_score", 0.0),
        "ml_score":       tx.get("ml_score", 0.0),
        "ai_confidence":  ai_sig.get("confidence", 0.0),
        "reason":         decision["reason"],
        "ai_signals":     ai_sig.get("signals", []),
        "amount":         tx.get("amount", 0.0),
        "rail":           tx.get("rail", ""),
        "top_pattern":    tx.get("top_pattern"),
        "matched_signals":tx.get("matched_signals", []),
        "escalate_human": decision.get("escalate_human", False),
        # Filled on resolution:
        "analyst_action": None,
        "analyst_id":     None,
        "analyst_note":   "",
        "resolved_at":    None,
    }


# ── Self-learning ─────────────────────────────────────────────────────────────

async def trigger_learning() -> None:
    """
    Pull a snapshot of novel_attack_buffer, run Rule Factory pipeline in a
    thread pool (it uses blocking urllib I/O), update counters.
    Respects the self_learning toggle — exits early if disabled.
    """
    global novel_attack_buffer

    if not agent_config["toggles"].get("self_learning", True):
        return

    import os as _os
    api_key = _os.environ.get("LLM_API_KEY") or _os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return

    buf_size = agent_config.get("novel_buffer_size", 10)
    snapshot = novel_attack_buffer[:buf_size]
    novel_attack_buffer = novel_attack_buffer[buf_size:]

    try:
        from patterns import PATTERNS as _static_patterns
        from rule_factory import run_pipeline as _run_pipeline

        existing = [{"name": p["name"], "tier": 0, "reason": p.get("description", "")} for p in _static_patterns]

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_pipeline, api_key, existing)

        if isinstance(result, dict) and result.get("status") != "error":
            agent_state.patterns_learned += 1
            learn_event = {
                "type":      "pattern_learned",
                "patterns_generated": result.get("candidates", 1),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            _broadcast(learn_event)
    except Exception:
        pass   # learning failures are non-fatal; agent keeps running


# ── Fan-out helper ────────────────────────────────────────────────────────────

def _broadcast(event: dict) -> None:
    """Push an event to all registered SSE subscriber queues."""
    dead = set()
    for q in _event_subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass
        except Exception:
            dead.add(q)
    _event_subscribers.difference_update(dead)


# ── Main agent loop ───────────────────────────────────────────────────────────

async def run_agent(build_event_fn, df_all, features) -> None:
    """
    Autonomous detection loop. Accepts build_event from main.py as a parameter
    to avoid circular imports — agent.py never imports from main.py.

    Runs continuously until agent_state.running is set to False.
    """
    import pandas as pd

    # Model not loaded → abort gracefully
    if df_all is None or (hasattr(df_all, "empty") and df_all.empty):
        agent_state.running = False
        return

    agent_state.running    = True
    agent_state.start_time = datetime.utcnow()

    # Build a realistic sample: 60 confirmed fraud + 240 legit, shuffled
    try:
        if "is_fraud" in df_all.columns:
            fraud_rows = df_all[df_all["is_fraud"] == True].head(60)
            legit_rows = df_all[df_all["is_fraud"] == False].head(240)
            sample     = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            sample = df_all.head(300).sample(frac=1, random_state=42).reset_index(drop=True)
    except Exception:
        agent_state.running = False
        return

    idx = 0
    while agent_state.running:
        try:
            row = sample.iloc[idx % len(sample)]
            idx += 1

            # ── Real XGBoost ML inference via build_event ─────────────────
            tx = build_event_fn(row)

            ml_score    = tx.get("ml_score", 0.0)
            c_score     = tx.get("combined_score", 0.0)
            top_pattern = tx.get("top_pattern") or ""

            # ── AI behavioral signature detection ─────────────────────────
            ai_sig      = detect_ai_signature(tx)
            threat_type = classify_threat(tx, ml_score, ai_sig)
            ai_sig["threat_type"] = threat_type

            # ── Autonomous decision (graph risk feeds the 60/25/15 blend) ─
            graph_risk = tx.get("graph_risk_score", 0.0)
            decision   = autonomous_decision(threat_type, c_score, ai_sig, agent_config, graph_risk)

            # ── Update counters ───────────────────────────────────────────
            action = decision["action"]
            if action == "block":
                agent_state.blocked_count += 1
            elif action == "flag":
                agent_state.flagged_count += 1
            else:
                agent_state.allowed_count += 1

            # ── Build SSE event ───────────────────────────────────────────
            meta  = THREAT_META.get(threat_type, {"label": threat_type, "color": "#64748b"})
            event = {
                "transaction_id":   tx.get("transaction_id", ""),
                "amount":           tx.get("amount", 0.0),
                "rail":             tx.get("rail", ""),
                "ml_score":         ml_score,
                "combined_score":   c_score,
                "top_pattern":      top_pattern or None,
                "pattern_color":    tx.get("pattern_color", "#64748b"),
                "threat_type":      threat_type,
                "threat_label":     meta["label"],
                "threat_color":     meta["color"],
                "action":           action,
                "action_confidence":decision["confidence"],
                "reason":           decision["reason"],
                "escalate_human":   decision.get("escalate_human", False),
                "ai_signals":       ai_sig.get("signals", []),
                "ai_confidence":    ai_sig.get("confidence", 0.0),
                "graph_risk_score": graph_risk,
                "graph_context":    tx.get("graph_context", {}),
                "timestamp":        datetime.utcnow().isoformat() + "Z",
            }

            # ── Case queue ────────────────────────────────────────────────
            if action != "allow" and _should_create_case(action, decision.get("escalate_human", False), agent_config):
                case = _make_case(tx, ai_sig, decision, threat_type)
                agent_state.case_queue.appendleft(case)

            # ── Novel pattern buffer ──────────────────────────────────────
            is_novel = (
                c_score   >= 0.65
                and ml_score  >= 0.70
                and not top_pattern
            )
            if is_novel:
                novel_attack_buffer.append(event)
                if len(novel_attack_buffer) >= agent_config.get("novel_buffer_size", 10):
                    asyncio.create_task(trigger_learning())

            # ── Broadcast to SSE clients ──────────────────────────────────
            agent_state.recent_events.appendleft(event)
            _broadcast(event)

        except Exception:
            pass   # individual tick failures are non-fatal

        await asyncio.sleep(agent_config.get("speed", 0.25))
