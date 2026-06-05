# Pattern Library — structured attack fingerprints for all 6 fraud typologies.
# Each pattern maps directly to the FRAUD_TYPOLOGIES in the ML Fraud Engine.
# Signals reference the 10 engineered features in the XGBoost model.

PATTERNS = [
    {
        "id": "pig_butchering",
        "name": "Pig Butchering",
        "icon": "🐷",
        "risk": "Critical",
        "color": "#ef4444",
        "prevalence": 0.30,
        "description": (
            "Long grooming phase (14–90 days) builds victim trust through fake relationships "
            "or investment advice, followed by a large one-way exit via crypto or FedNow. "
            "The recipient appears NEW because the trust was built off-platform."
        ),
        "evasion": "Gradual amount escalation; warm recipient slowly over weeks before strike.",
        "signals": [
            {"feature": "recipient_familiarity", "op": "lt", "threshold": 0.15, "weight": 0.30, "label": "New/unknown recipient"},
            {"feature": "amount_vs_max",         "op": "gt", "threshold": 0.70, "weight": 0.25, "label": "Unusually large vs. user history"},
            {"feature": "rail_risk",             "op": "gt", "threshold": 0.80, "weight": 0.20, "label": "Irrevocable high-risk rail"},
            {"feature": "is_crypto",             "op": "eq", "threshold": 1.0,  "weight": 0.15, "label": "Crypto payment"},
            {"feature": "amount_zscore",         "op": "gt", "threshold": 2.0,  "weight": 0.10, "label": "Statistically anomalous amount"},
        ],
    },
    {
        "id": "app_scam",
        "name": "APP Scam",
        "icon": "📲",
        "risk": "High",
        "color": "#f97316",
        "prevalence": 0.20,
        "description": (
            "Authorized push payment where victim is socially engineered into sending money "
            "voluntarily — impersonation of bank, HMRC/IRS, police, or romantic partner. "
            "Transaction is technically authorized but victim is manipulated."
        ),
        "evasion": "Payment rail is legitimate; victim authorizes. High amount, urgent framing.",
        "signals": [
            {"feature": "amount_vs_max",         "op": "gt", "threshold": 0.50, "weight": 0.30, "label": "High amount vs. normal behavior"},
            {"feature": "rail_risk",             "op": "gt", "threshold": 0.75, "weight": 0.25, "label": "Instant/irrevocable rail"},
            {"feature": "recipient_familiarity", "op": "lt", "threshold": 0.20, "weight": 0.25, "label": "Unfamiliar recipient"},
            {"feature": "is_p2p",               "op": "eq", "threshold": 1.0,  "weight": 0.10, "label": "P2P payment channel"},
            {"feature": "hour_risk",             "op": "gt", "threshold": 0.50, "weight": 0.10, "label": "Off-hours or unusual time"},
        ],
    },
    {
        "id": "account_takeover_ai",
        "name": "AI-Powered ATO",
        "icon": "🤖",
        "risk": "Critical",
        "color": "#c084fc",
        "prevalence": 0.20,
        "description": (
            "AI-assisted account takeover: LLM-generated phishing, deepfake voice for 2FA bypass, "
            "or credential stuffing. New device + immediate high-value transfer is the signature. "
            "Behavioral shift is abrupt, not gradual."
        ),
        "evasion": "Mimics victim behavior with AI; may wait hours before initiating transfer.",
        "signals": [
            {"feature": "device_familiarity",   "op": "lt", "threshold": 0.05, "weight": 0.35, "label": "Unrecognized/new device"},
            {"feature": "amount_zscore",        "op": "gt", "threshold": 2.5,  "weight": 0.25, "label": "Extreme amount outlier"},
            {"feature": "rail_risk",            "op": "gt", "threshold": 0.80, "weight": 0.20, "label": "High-risk irrevocable rail"},
            {"feature": "recipient_familiarity","op": "lt", "threshold": 0.10, "weight": 0.15, "label": "Unknown destination"},
            {"feature": "hour_risk",            "op": "gt", "threshold": 0.50, "weight": 0.05, "label": "Unusual transaction hour"},
        ],
    },
    {
        "id": "deepfake_social_engineering",
        "name": "Deepfake Social Eng.",
        "icon": "🎭",
        "risk": "Critical",
        "color": "#38bdf8",
        "prevalence": 0.15,
        "description": (
            "AI-generated voice or video impersonates a trusted authority (CEO, bank official, family). "
            "Victim is convinced during business hours to authorize an unusually large transfer. "
            "Device and recipient may be familiar — the manipulation happens out-of-band."
        ),
        "evasion": "Known device, business hours, victim authorizes — evades device/time signals.",
        "signals": [
            {"feature": "amount_vs_max",         "op": "gt", "threshold": 0.80, "weight": 0.35, "label": "Extremely large relative to history"},
            {"feature": "amount_zscore",         "op": "gt", "threshold": 3.0,  "weight": 0.25, "label": "Far outside normal distribution"},
            {"feature": "rail_risk",             "op": "gt", "threshold": 0.70, "weight": 0.25, "label": "High-risk rail selected"},
            {"feature": "recipient_familiarity", "op": "lt", "threshold": 0.15, "weight": 0.15, "label": "Destination not in known payees"},
        ],
    },
    {
        "id": "synthetic_id_ai",
        "name": "AI Synthetic Identity",
        "icon": "🪪",
        "risk": "High",
        "color": "#f59e0b",
        "prevalence": 0.10,
        "description": (
            "AI-generated synthetic identity: fabricated SSN, address, employment history. "
            "Gradual credit building over months, then bust-out — maxing every available credit "
            "line in rapid succession before disappearing. High velocity, all new payees."
        ),
        "evasion": "Slow build phase mimics legitimate user; bust-out happens in a single window.",
        "signals": [
            {"feature": "velocity_1h",           "op": "gt", "threshold": 0.60, "weight": 0.30, "label": "Abnormally high transaction velocity"},
            {"feature": "recipient_familiarity", "op": "lt", "threshold": 0.10, "weight": 0.25, "label": "All payees are new"},
            {"feature": "device_familiarity",   "op": "lt", "threshold": 0.10, "weight": 0.25, "label": "Unknown device fingerprint"},
            {"feature": "amount_vs_max",         "op": "gt", "threshold": 0.90, "weight": 0.20, "label": "Near-limit bust-out transaction"},
        ],
    },
    {
        "id": "card_testing_bot",
        "name": "Card Testing Bot",
        "icon": "🃏",
        "risk": "Medium",
        "color": "#22c55e",
        "prevalence": 0.05,
        "description": (
            "Automated bot validates stolen card credentials using micro-transactions ($0.01–$1.99) "
            "at high velocity across subscription or gaming merchants. Successful cards are sold "
            "or used for full-value purchases immediately after."
        ),
        "evasion": "Low amount evades threshold rules; distributed across merchants to avoid velocity blocks.",
        "signals": [
            {"feature": "amount_zscore", "op": "lt", "threshold": -1.5, "weight": 0.35, "label": "Micro-transaction (below normal)"},
            {"feature": "velocity_1h",  "op": "gt", "threshold": 0.70, "weight": 0.35, "label": "Extremely high transaction velocity"},
            {"feature": "device_familiarity", "op": "lt", "threshold": 0.05, "weight": 0.20, "label": "Bot-like / unknown device"},
            {"feature": "hour_risk",    "op": "gt", "threshold": 0.50, "weight": 0.10, "label": "Off-hours automated activity"},
        ],
    },
]

# Index for fast lookup by id
PATTERN_INDEX = {p["id"]: p for p in PATTERNS}
