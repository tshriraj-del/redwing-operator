"""
case_file.py - Investigator case-file assembler.

Turns a single transaction into the full decisioning surface a real fraud analyst
works from: Customer 360 / CDD, card-usage detail, card-fraud typology signals,
dispute / chargeback evidence study, behavioural-device-network context, an activity
timeline, and a recommended disposition.

DESIGN NOTE - enrichment, not fabrication:
  The 880k synthetic dataset carries amount, rail, merchant_category, mcc_code,
  fraud_typology, device/recipient familiarity, velocity and graph context, plus the
  ground-truth is_fraud label. It does NOT carry card-entry detail, AVS/CVV/3DS,
  disputes, or a KYC/risk profile. This module DERIVES those deterministically per-ID
  and keeps them COHERENT with the ground truth (a card_testing_bot row gets a
  card-testing signal and CNP ecommerce entry; a legit row gets clean AVS/CVV, etc.).
  In production these same fields arrive from the connector hub (credit bureaus,
  device intelligence, dispute systems). Deriving them here lets the panel be real
  without regenerating the dataset or retraining - which would risk re-introducing the
  training-serving skew the platform exists to catch. Seeded by ID → stable on reload.

SAR is deliberately NOT the centre of gravity. It is a downstream action that only
becomes eligible after a "confirm fraud" disposition and a reporting threshold.
"""

import hashlib
import random
from datetime import datetime, timedelta

# ── Reference tables ─────────────────────────────────────────────────────────

# MCC → (label, base card-risk 0-1). Keyed on the integer part of mcc_code.
MCC_RISK = {
    "6051": ("Quasi-cash / crypto purchase", 0.90),
    "6099": ("Cash services",                0.82),
    "4829": ("Money transfer / wire",        0.88),
    "6211": ("Securities / brokerage",       0.62),
    "7995": ("Gambling",                     0.85),
    "5816": ("Digital goods / games",        0.58),
    "5968": ("Subscription / direct mktg",   0.50),
    "7372": ("Computer / IT services",       0.42),
    "7395": ("Photo / ID services",          0.55),
    "4121": ("Taxi / rideshare",             0.30),
    "6012": ("Financial institution",        0.55),
    "4900": ("Utilities",                    0.15),
    "4941": ("Utilities",                    0.15),
    "4911": ("Utilities",                    0.15),
    "5311": ("Department store",             0.22),
    "5999": ("Misc / general retail",        0.25),
}

# Card-network from BIN first digit.
_NETWORK = {"3": "Amex", "4": "Visa", "5": "Mastercard", "6": "Discover"}

# Visa/Mastercard dispute reason codes (representative subset).
DISPUTE_REASONS = {
    "10.4": "Fraud - Card-Absent Environment (CNP)",
    "10.1": "Fraud - EMV Liability Shift Counterfeit",
    "10.3": "Fraud - Card-Present Environment",
    "13.1": "Consumer Dispute - Merchandise/Service Not Received",
    "13.7": "Consumer Dispute - Cancelled Merchandise/Service",
    "11.3": "Authorization - No Authorization",
}

# Fraud typologies that, when CONFIRMED, can meet a SAR reporting threshold.
SAR_REPORTABLE = {
    "pig_butchering", "APP_scam", "deepfake_social_engineering",
    "synthetic_id_ai", "account_takeover_ai",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rng(*parts) -> random.Random:
    """Deterministic RNG seeded by the given id parts (stable across reloads)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _num(v, default=0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _mcc_key(row) -> str:
    raw = str(row.get("mcc_code", "") or "").strip()
    return raw.split(".")[0] if raw else ""


def _is_card(row) -> bool:
    return str(row.get("payment_rail", row.get("rail", "card"))).lower() == "card"


# ── Customer 360 / CDD / risk profile ────────────────────────────────────────

def build_customer(row, graph_ctx) -> dict:
    """Customer due-diligence view + behavioural baseline + risk rating.
    Customer-level fields are seeded by user_id so they're stable for that user
    across every one of their transactions."""
    uid = str(row.get("user_id", "unknown"))
    r = _rng("cust", uid)

    account_age = r.randint(20, 3200)          # days
    tenure_band = ("New (<90d)" if account_age < 90 else
                   "Established (90d–2y)" if account_age < 730 else
                   "Tenured (2y+)")

    kyc_full = r.random() > 0.08
    pep = r.random() < 0.03
    sanctions_hit = r.random() < 0.01
    nat = r.choice(["US", "US", "US", "GB", "CA", "IN", "NG", "DE", "BR"])
    loc = r.choice(["New York, US", "Austin, US", "London, GB", "Toronto, CA",
                    "Mumbai, IN", "Lagos, NG", "Berlin, DE", "São Paulo, BR"])

    prior_disputes = r.choices([0, 0, 1, 2, 4], weights=[55, 20, 12, 8, 5])[0]
    prior_cases = r.choices([0, 0, 1, 3], weights=[60, 20, 12, 8])[0]
    prior_sars = r.choices([0, 0, 0, 1], weights=[80, 10, 6, 4])[0]

    # Risk rating is DERIVED from drivers so it's defensible, not random.
    drivers = []
    score = 0
    if account_age < 90:
        score += 2; drivers.append("Account < 90 days old")
    if not kyc_full:
        score += 2; drivers.append("KYC not fully verified")
    if pep:
        score += 2; drivers.append("Politically Exposed Person")
    if sanctions_hit:
        score += 4; drivers.append("Sanctions/watchlist potential match")
    if prior_sars:
        score += 3; drivers.append(f"{prior_sars} prior SAR(s) filed")
    if prior_disputes >= 2:
        score += 1; drivers.append(f"{prior_disputes} prior disputes")
    if (graph_ctx or {}).get("graph_risk_score", 0) > 0.5:
        score += 2; drivers.append("Linked to flagged counterparties")
    rating = "High" if score >= 5 else "Medium" if score >= 2 else "Low"
    if not drivers:
        drivers.append("No elevated risk indicators on file")

    return {
        "user_id": uid,
        "name_masked": f"Customer ••••{uid[-4:] if len(uid) >= 4 else uid}",
        "account_age_days": account_age,
        "tenure_band": tenure_band,
        "kyc_status": "Verified" if kyc_full else "Partial - review",
        "cip_verified": kyc_full,
        "id_type": r.choice(["Passport", "Driver's License", "National ID"]),
        "nationality": nat,
        "location": loc,
        "pep": pep,
        "sanctions_screened": True,
        "sanctions_match": sanctions_hit,
        "risk_rating": rating,
        "risk_drivers": drivers,
        "products": r.sample(["Checking", "Credit Card", "Savings", "Debit Card",
                              "Brokerage", "Wallet"], k=r.randint(1, 3)),
        "lifetime_value_band": r.choice(["Low", "Medium", "High", "VIP"]),
        "prior_cases": prior_cases,
        "prior_sars": prior_sars,
        "prior_disputes": prior_disputes,
        "baseline": {
            "avg_txn": round(_num(row.get("user_avg"), 0.0), 2),
            "typical_max": round(_num(row.get("user_max"), 0.0), 2),
            "known_devices": int(_num(row.get("dev_count"), 1)),
            "known_recipients": int(_num(row.get("recv_count"), 0)),
            "home_geo": loc,
        },
    }


# ── Card / payment-instrument detail ─────────────────────────────────────────

def build_instrument(row) -> dict:
    """Card-usage detail (entry mode, AVS/CVV/3DS) when the rail is card; a
    rail-appropriate stub otherwise. Coherent with the fraud label."""
    rail = str(row.get("payment_rail", row.get("rail", "card")))
    if not _is_card(row):
        return {"type": "non_card", "rail": rail,
                "instant": _truthy(row.get("is_instant_rail")),
                "irrevocable": _truthy(row.get("rail_irrevocable"))}

    r = _rng("card", row.get("transaction_id"))
    is_fraud = _truthy(row.get("is_fraud"))
    typ = str(row.get("fraud_typology", "none"))
    mcc_key = _mcc_key(row)
    high_risk_mcc = MCC_RISK.get(mcc_key, ("", 0.0))[1] >= 0.55

    bin6 = r.choice(["4", "4", "5", "5", "3", "6"]) + "".join(str(r.randint(0, 9)) for _ in range(5))
    network = _NETWORK.get(bin6[0], "Visa")
    funding = r.choices(["credit", "debit", "prepaid"], weights=[55, 35, 10])[0]

    # Card-present vs card-not-present. Fraud + high-risk MCC skews heavily CNP.
    cnp_bias = 0.55 + (0.3 if is_fraud else 0.0) + (0.15 if high_risk_mcc else 0.0)
    cnp = r.random() < min(cnp_bias, 0.97)
    if cnp:
        entry = "ecommerce" if r.random() < 0.8 else "keyed"
        present = "Card-Not-Present"
    else:
        entry = r.choices(["chip", "contactless", "swipe"], weights=[60, 30, 10])[0]
        present = "Card-Present"

    # AVS/CVV/3DS. Fraud → far more mismatches / not-provided.
    def _result(good_label, bad_label, bad_prob):
        return bad_label if r.random() < bad_prob else good_label
    bad = 0.55 if is_fraud else 0.06
    avs = _result("Match (Y)", r.choice(["No Match (N)", "Address only (A)", "ZIP only (Z)"]), bad)
    cvv = _result("Match (M)", r.choice(["No Match (N)", "Not Provided (P)"]), bad)
    three_ds = (_result("Authenticated", r.choice(["Not Enrolled", "Attempted", "Failed"]), bad)
                if cnp else "N/A (card-present)")

    return {
        "type": "card",
        "rail": rail,
        "bin": bin6,
        "last4": "".join(str(r.randint(0, 9)) for _ in range(4)),
        "network": network,
        "funding": funding,
        "presence": present,
        "entry_mode": entry,
        "avs_result": avs,
        "cvv_result": cvv,
        "three_ds_result": three_ds,
        "pos_country": r.choice(["US", "US", "US", "GB", "RU", "NG", "CN"]) if cnp else "US",
        "cross_border": r.random() < (0.4 if is_fraud else 0.07),
    }


# ── Card-fraud typology signals - "does this ring a bell?" ────────────────────

def card_fraud_signals(row, instrument, graph_ctx) -> list:
    """Pattern-match card usage against known card-fraud playbooks.
    Each signal: {code, label, severity (low/medium/high), detail}."""
    out = []
    if instrument.get("type") != "card":
        return out

    typ = str(row.get("fraud_typology", "none"))
    amt = _num(row.get("amount"))
    amt_vs_max = _num(row.get("amount_vs_max"))
    dev_fam = _num(row.get("device_familiarity"))
    vel1 = _num(row.get("velocity_1h"))
    vel24 = _num(row.get("velocity_24h"))
    mcc_key = _mcc_key(row)
    mcc_label, mcc_risk = MCC_RISK.get(mcc_key, ("Unclassified MCC", 0.3))

    def add(code, label, sev, detail):
        out.append({"code": code, "label": label, "severity": sev, "detail": detail})

    # 1. Card testing / BIN attack - many small auths, high velocity.
    if typ == "card_testing_bot" or (amt < 5 and vel1 > 0.5):
        vel_clause = (f"with elevated 1h velocity ({vel1:.2f})" if vel1 > 0.4
                      else "flagged as automated card-validation activity")
        add("card_testing", "Card-testing / BIN-attack pattern", "high",
            f"Low-value auth (${amt:.2f}) {vel_clause} - consistent with testing a "
            "stolen BIN range before cash-out.")

    # 2. CNP into a high-risk MCC.
    if instrument["presence"] == "Card-Not-Present" and mcc_risk >= 0.55:
        add("cnp_high_risk_mcc", "CNP purchase at high-risk merchant category", "high",
            f"Card-not-present at MCC {mcc_key} ({mcc_label}) - a common cash-out "
            "channel. No physical card or 3DS step-up to anchor the cardholder.")

    # 3. AVS / CVV mismatch.
    if "No Match" in instrument["avs_result"] or "No Match" in instrument["cvv_result"] \
            or "Not Provided" in instrument["cvv_result"]:
        add("avs_cvv_mismatch", "AVS / CVV verification failed", "medium",
            f"AVS={instrument['avs_result']}, CVV={instrument['cvv_result']} - "
            "billing details don't match issuer record; hallmark of CNP fraud.")

    # 4. Account-takeover signature: new device + high value.
    if dev_fam < 0.2 and amt_vs_max > 0.9:
        add("ato_new_device", "Account-takeover signature", "high",
            "Near-record-value purchase from an unrecognised device - "
            "classic post-ATO cash-out after a credential/session compromise.")
    elif typ == "account_takeover_ai":
        add("ato_new_device", "Account-takeover signature", "high",
            "Behavioural drift from baseline consistent with account takeover.")

    # 5. Magstripe fallback (counterfeit / skimming).
    if instrument.get("entry_mode") == "swipe":
        add("magstripe_fallback", "Magstripe fallback on a chip card", "medium",
            "Swipe used where chip was available - fallback abuse is a counterfeit / "
            "skimming tell.")

    # 6. Cross-border / impossible-travel.
    if instrument.get("cross_border"):
        add("cross_border", "Cross-border authorisation", "low",
            f"POS country {instrument.get('pos_country')} differs from the customer's "
            "home geography.")

    # 7. Linked to a known fraud ring (graph).
    if (graph_ctx or {}).get("graph_risk_score", 0) > 0.5:
        add("ring_link", "Counterparty in flagged network", "high",
            "Recipient/device overlaps with accounts already tied to confirmed fraud.")

    if not out:
        add("clean", "No card-fraud playbook matched", "low",
            "Card usage is consistent with the customer's established behaviour.")
    return out


# ── Dispute / chargeback evidence study ──────────────────────────────────────

def dispute_analysis(row, instrument, customer) -> dict:
    """Study the dispute proof: build the evidence matrix and separate genuine
    third-party fraud from first-party (friendly) fraud."""
    r = _rng("dispute", row.get("transaction_id"))
    is_fraud = _truthy(row.get("is_fraud"))
    is_card = instrument.get("type") == "card"

    # Is there an ACTIVE dispute on this transaction? Fraud rows and customers with a
    # dispute history are likelier to have one.
    active = is_fraud or r.random() < (0.25 if customer["prior_disputes"] else 0.08)

    history = customer["prior_disputes"]
    if not active:
        return {
            "active": False,
            "history_count": history,
            "summary": f"No active dispute. {history} prior dispute(s) on file.",
        }

    # Evidence matrix - what proof the system pulled to adjudicate.
    if is_fraud:
        # True third-party fraud: evidence points AWAY from the cardholder.
        ev = {
            "cardholder_device_match": False,
            "ip_geo_match": False,
            "avs_match": "Match" in instrument.get("avs_result", "") if is_card else None,
            "cvv_match": "Match" in instrument.get("cvv_result", "") if is_card else None,
            "three_ds_authenticated": instrument.get("three_ds_result") == "Authenticated" if is_card else None,
            "delivery_to_known_address": False,
            "prior_merchant_relationship": False,
        }
    else:
        # Likely first-party / friendly fraud: evidence points TOWARD the cardholder.
        ev = {
            "cardholder_device_match": True,
            "ip_geo_match": True,
            "avs_match": True if is_card else None,
            "cvv_match": True if is_card else None,
            "three_ds_authenticated": True if is_card else None,
            "delivery_to_known_address": True,
            "prior_merchant_relationship": r.random() < 0.6,
        }

    # First-party-fraud likelihood: how much evidence ties the cardholder to the txn.
    pro_cardholder = sum(1 for k in (
        "cardholder_device_match", "ip_geo_match", "delivery_to_known_address",
        "prior_merchant_relationship") if ev.get(k) is True)
    pro_cardholder += sum(1 for k in ("avs_match", "cvv_match", "three_ds_authenticated")
                          if ev.get(k) is True)
    first_party_risk = round(min(pro_cardholder / 7.0, 1.0), 2)

    reason_code = ("10.4" if is_fraud and instrument.get("presence") == "Card-Not-Present"
                   else "10.3" if is_fraud else "13.1")

    if first_party_risk >= 0.6:
        assessment = ("Evidence places the cardholder at the transaction (device, IP, "
                      "delivery and verification all match). Pattern is consistent with "
                      "FIRST-PARTY / friendly fraud - customer disputing a charge they made.")
        representment = "Represent / deny - strong evidence to win representment."
    elif first_party_risk <= 0.3:
        assessment = ("Evidence does NOT place the cardholder at the transaction "
                      "(unknown device, geo mismatch, failed/absent verification). "
                      "Consistent with genuine THIRD-PARTY fraud.")
        representment = "Accept liability - issue provisional credit; pursue issuer recovery."
    else:
        assessment = ("Mixed evidence - neither cardholder presence nor third-party "
                      "compromise is conclusive. Request additional documentation.")
        representment = "Request info - delivery proof / cardholder statement before deciding."

    return {
        "active": True,
        "history_count": history,
        "reason_code": reason_code,
        "reason": DISPUTE_REASONS.get(reason_code, "Fraud"),
        "evidence": ev,
        "first_party_fraud_risk": first_party_risk,
        "assessment": assessment,
        "representment_recommendation": representment,
    }


# ── Activity timeline ────────────────────────────────────────────────────────

def build_timeline(row, instrument) -> list:
    """Chronological account-activity log leading up to the flagged transaction."""
    r = _rng("tl", row.get("transaction_id"))
    is_fraud = _truthy(row.get("is_fraud"))
    now = datetime.utcnow()

    def ts(mins_ago):
        return (now - timedelta(minutes=mins_ago)).isoformat() + "Z"

    events = []
    if is_fraud and instrument.get("type") == "card":
        events += [
            (1440, "login", "Login from new device / unrecognised IP", "warn"),
            (1435, "profile_change", "Notification email/phone changed", "warn"),
            (1430, "device_add", "New device enrolled to the account", "warn"),
            (60,   "velocity", "Multiple authorisation attempts in short window", "warn"),
        ]
    else:
        events += [
            (4320, "login", "Login from known device", "ok"),
            (180,  "login", "Login from known device", "ok"),
        ]
    events.append((0, "transaction",
                   f"Flagged transaction - ${_num(row.get('amount')):.2f} "
                   f"at {row.get('merchant_category', 'merchant')}", "flag"))
    return [{"ts": ts(m), "type": t, "detail": d, "level": lvl}
            for (m, t, d, lvl) in sorted(events, key=lambda e: -e[0])]


# ── Disposition recommendation ───────────────────────────────────────────────

def recommend(row, scored, signals, dispute, customer) -> dict:
    """Recommend the analyst action. Mirrors how a real queue would pre-triage."""
    score = _num(scored.get("combined_score"), _num(scored.get("ml_score")))
    high_sev = [s for s in signals if s["severity"] == "high" and s["code"] != "clean"]

    if dispute.get("active") and dispute.get("first_party_fraud_risk", 0) >= 0.6:
        action, conf = "deny_dispute_first_party", 0.8
        rationale = ("Dispute evidence indicates first-party / friendly fraud. "
                     "Deny the dispute and represent.")
    elif score >= 0.85 and high_sev:
        action, conf = "confirm_fraud", 0.85
        rationale = (f"High model score ({score:.2f}) corroborated by "
                     f"{len(high_sev)} high-severity card-fraud signal(s). "
                     "Confirm fraud, block the instrument, issue provisional credit.")
    elif score >= 0.5 or high_sev:
        action, conf = "escalate_stepup", 0.6
        rationale = ("Elevated but not conclusive - step-up authenticate the customer "
                     "and request additional context before deciding.")
    else:
        action, conf = "clear_false_positive", 0.7
        rationale = ("Signals align with the customer's baseline; no corroborating "
                     "card-fraud pattern. Likely a false positive - clear.")

    return {"action": action, "confidence": conf, "rationale": rationale}


# ── Top-level assembly ───────────────────────────────────────────────────────

DISPOSITION_OPTIONS = [
    {"id": "confirm_fraud",            "label": "Confirm fraud",          "tone": "danger"},
    {"id": "clear_false_positive",     "label": "Clear (false positive)", "tone": "ok"},
    {"id": "deny_dispute_first_party", "label": "Deny dispute (1st-party)","tone": "warn"},
    {"id": "escalate_stepup",          "label": "Step-up / request info", "tone": "warn"},
    {"id": "block_instrument",         "label": "Block card / account",   "tone": "danger"},
    {"id": "place_hold",               "label": "Place hold",             "tone": "warn"},
]


def assemble(row, scored, graph_ctx=None, explanation=None) -> dict:
    """Build the complete investigator case file from a scored transaction row."""
    row = dict(row)
    graph_ctx = graph_ctx or scored.get("graph_context") or {}

    customer   = build_customer(row, graph_ctx)
    instrument = build_instrument(row)
    signals    = card_fraud_signals(row, instrument, graph_ctx)
    dispute    = dispute_analysis(row, instrument, customer)
    timeline   = build_timeline(row, instrument)
    rec        = recommend(row, scored, signals, dispute, customer)

    score = _num(scored.get("combined_score"), _num(scored.get("ml_score")))
    verdict = ("CRITICAL" if score >= 0.85 else "HIGH" if score >= 0.6
               else "MEDIUM" if score >= 0.35 else "LOW")
    priority = {"CRITICAL": "P1", "HIGH": "P2", "MEDIUM": "P3", "LOW": "P4"}[verdict]

    txn_id = str(row.get("transaction_id", "unknown"))
    case_seed = _rng("case", txn_id)
    opened = datetime.utcnow()

    # SAR only matters once fraud is confirmed AND the typology is reportable.
    typ = str(row.get("fraud_typology", "none"))
    sar_eligible = (rec["action"] == "confirm_fraud" and
                    (typ in SAR_REPORTABLE or score >= 0.85))

    # Top model features for the alert panel (from XAI if available).
    top_features = []
    if explanation and isinstance(explanation, dict):
        for f in (explanation.get("top_features") or explanation.get("feature_contributions") or [])[:5]:
            if isinstance(f, dict):
                top_features.append({
                    "feature": f.get("feature") or f.get("name"),
                    "contribution": f.get("contribution") or f.get("shap_value") or f.get("value"),
                })

    return {
        "case_id": f"CASE-{txn_id[-6:].upper() if len(txn_id) >= 6 else txn_id.upper()}",
        "transaction_id": txn_id,
        "opened_at": opened.isoformat() + "Z",
        "sla_due_at": (opened + timedelta(hours=24 if priority in ("P3", "P4") else 4)).isoformat() + "Z",
        "status": "Open - under investigation",
        "priority": priority,
        "queue": "Card Fraud" if instrument.get("type") == "card" else "Payments Fraud",
        "assigned_to": "unassigned",

        "alert": {
            "trigger": scored.get("top_pattern_id") or "model_score",
            "trigger_label": scored.get("top_pattern") or "ML risk score",
            "model_score": round(_num(scored.get("ml_score")), 4),
            "combined_score": round(score, 4),
            "verdict": verdict,
            "fraud_typology": typ,
            "matched_signals": scored.get("matched_signals", []),
            "top_features": top_features,
            "ground_truth_label": "fraud" if _truthy(row.get("is_fraud")) else "legitimate",
        },

        "customer": customer,
        "transaction": {
            "amount": round(_num(row.get("amount")), 2),
            "currency": "USD",
            "timestamp": str(row.get("timestamp", opened.isoformat() + "Z")),
            "rail": str(row.get("payment_rail", row.get("rail", "card"))),
            "merchant_category": row.get("merchant_category", "unknown"),
            "mcc_code": _mcc_key(row),
            "mcc_label": MCC_RISK.get(_mcc_key(row), ("Unclassified", 0))[0],
            "recipient_id": row.get("recipient_id", ""),
            "is_new_recipient": _truthy(row.get("is_new_recipient")),
        },
        "instrument": instrument,
        "card_fraud_signals": signals,
        "dispute": dispute,
        "device_network": {
            "device_id": row.get("device_id", ""),
            "is_known_device": _num(row.get("device_familiarity")) >= 0.5,
            "graph_risk_score": round(_num((graph_ctx or {}).get("graph_risk_score")), 3),
            "linked_accounts": (graph_ctx or {}).get("linked_accounts",
                                (graph_ctx or {}).get("ring_size", 0)),
            "ring_flag": _num((graph_ctx or {}).get("graph_risk_score")) > 0.5,
        },
        "timeline": timeline,

        "recommended_disposition": rec,
        "disposition_options": DISPOSITION_OPTIONS,
        "sar_eligible": sar_eligible,
        "sar_note": ("Eligible after confirming fraud - SAR is the final step, not the first."
                     if sar_eligible else
                     "Not yet eligible - SAR follows a confirm-fraud disposition + threshold."),

        "_enrichment_note": ("Card-entry, AVS/CVV/3DS, dispute evidence and CDD fields are "
                             "derived deterministically and kept coherent with ground truth; "
                             "in production they arrive from the connector hub. Scores and "
                             "graph context are from the live model pipeline."),
    }
