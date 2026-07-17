"""
core/seed_consortium_demo.py - inject a clearly-labeled cross-institution demo mule (WS3).

The synthetic ledger was never built with cross-institution structure (each fraud is
local), so a random split of it into two tenants produces no clean cross-bank mule.
This constructs one legible, HONEST example so the network effect is demonstrable:

  a mule payee that receives clean-looking authorized victim payments at Northwind
  Neobank (the victim bank never flags them) and flagged cash-outs at Coastline
  Exchange (the off-ramp caught them). Neither bank can see both legs; the DP
  consortium reveals the mule to Northwind without Coastline sharing raw data.

Everything is prefixed DEMO- so it can never be mistaken for organic data. Idempotent
(fixed ids). This is a labeled demonstration of the mechanism, not a measured result.

Run:  python3 -m core.seed_consortium_demo
"""

from __future__ import annotations

from .store import Store, DEFAULT_DB_PATH, eid
from .record import record_scored_event
from .consortium import institution_of

MULE = "DEMO-MULE-PIG-01"


def _users_for(institution: str, n: int, prefix: str) -> list:
    """Find n user ids that deterministically hash to the given institution."""
    out, i = [], 0
    while len(out) < n:
        uid = f"{prefix}{i}"
        if institution_of(uid) == institution:
            out.append(uid)
        i += 1
    return out


def seed(store: Store) -> dict:
    victims = _users_for("inst_neobank", 10, "DEMO-victim-")   # bank at Northwind
    cashout = _users_for("inst_crypto",   9, "DEMO-cashout-")  # cash out at Coastline
    flagged = 7                                                # Coastline caught 7 of 9

    for i, u in enumerate(victims):
        # authorized scam payments; the victim bank does NOT flag them locally
        record_scored_event(
            store, {"transaction_id": f"DEMO-nw-{i}", "amount": 2200.0, "rail": "zelle",
                    "ml_score": 0.4, "combined_score": 0.4, "is_alert": False},
            {"user_id": u, "recipient_id": MULE, "is_fraud": False, "fraud_typology": "pig_butchering"})

    for i, u in enumerate(cashout):
        fr = i < flagged
        record_scored_event(
            store, {"transaction_id": f"DEMO-cx-{i}", "amount": 1900.0, "rail": "crypto",
                    "ml_score": 0.9 if fr else 0.3, "combined_score": 0.9 if fr else 0.3, "is_alert": fr},
            {"user_id": u, "recipient_id": MULE, "is_fraud": fr, "fraud_typology": "pig_butchering"})

    # reputation so the scan pre-filter (fraud >= 3) surfaces the payee
    store.upsert_entity(eid("recipient", MULE), "recipient",
                        reputation={"tx": len(victims) + len(cashout), "fraud": flagged})

    return {"mule": MULE, "victims_neobank": len(victims),
            "cashouts_crypto": len(cashout), "flagged_at_crypto": flagged}


if __name__ == "__main__":
    s = Store(DEFAULT_DB_PATH)
    info = seed(s)
    print(f"seeded {info['mule']}: {info['victims_neobank']} Northwind victims (clean locally), "
          f"{info['cashouts_crypto']} Coastline cash-outs ({info['flagged_at_crypto']} flagged)")
    s.close()
