"""
core/consortium.py - privacy-preserving cross-institution recipient reputation (WS3).

This is the moat at n=2, and the one capability incumbents structurally cannot copy.

The problem: in an authorized-push-payment scam, a mule receives victim funds at one
institution and cashes out at another. Neither institution sees both legs, so neither
can flag the mule alone. The victim's bank sees only normal-looking outgoing payments.

The move: compute a payee's fraud reputation ACROSS institutions WITHOUT any of them
sharing raw customer data. Each institution publishes only differentially-private
(Laplace-noised) aggregate counts for a payee; those are combined into a smoothed
rate. The querying institution learns "this payee is a known mule elsewhere" while
learning nothing about anyone's customers or transactions. Every institution that
joins makes the signal better for all of them: that is the compounding network effect.

Two FICTIONAL tenants for the n=2 demo (not real organisations):
  inst_neobank  "Northwind Neobank"   - where victims bank (the senders)
  inst_crypto   "Coastline Exchange"  - the crypto off-ramp (the cash-out)

DP note: event-level DP, sensitivity 1 per published count, Laplace(1/epsilon). The
rigorous user-level treatment (contribution clamping, sequential composition
accounting) lives in pulseml_models/privacy_layer.py; this is its online, per-lookup
sibling, implemented in the stdlib so it stays testable without the ML stack.
"""

from __future__ import annotations

import hashlib
import math
import random

INSTITUTIONS = {
    "inst_neobank": "Northwind Neobank",
    "inst_crypto":  "Coastline Exchange",
}

PRIOR = 0.0065            # global fraud prior (matches graph_layer's empirical Bayes)
SMOOTH_K = 20.0          # pseudo-count: history required before trusting a rate
ALERT_THRESHOLD = 0.05   # a payee "alerts" at/above this smoothed fraud rate
MIN_EVIDENCE_TX = 8      # DP is a SCALE advantage: below this combined volume the
                         # Laplace noise dominates, so the network stays silent rather
                         # than alert on noise. This is the honest floor for n=2.


def institution_of(user_id: str) -> str:
    """Assign a sender (customer) to exactly ONE institution - a customer banks in one
    place. Deterministic hash so tenancy is stable and needs no stored mapping. A typed
    entity prefix ("user:") is stripped first, so the bare id and the store's entity id
    map to the SAME institution (they are the same customer)."""
    uid = str(user_id)
    if ":" in uid:
        uid = uid.split(":", 1)[1]
    h = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
    return "inst_neobank" if h % 2 == 0 else "inst_crypto"


def _laplace(scale: float, rng: random.Random) -> float:
    """Laplace(0, scale) via inverse-CDF sampling (stdlib, no numpy)."""
    if scale <= 0:
        return 0.0
    u = rng.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


def _smoothed(fraud: float, tx: float) -> float:
    return (max(0.0, fraud) + SMOOTH_K * PRIOR) / (max(0.0, tx) + SMOOTH_K)


def local_views(sender_labels) -> dict:
    """sender_labels: iterable of (user_id, is_fraud) for one payee. Returns, per
    institution, {tx, fraud, rate, alerts} - the payee AS EACH INSTITUTION SEES IT in
    its own data alone. This is the pre-sharing, private-to-each-bank view."""
    agg = {k: {"tx": 0, "fraud": 0} for k in INSTITUTIONS}
    for uid, fr in sender_labels:
        d = agg[institution_of(uid)]
        d["tx"]    += 1
        d["fraud"] += int(bool(fr))
    for k, d in agg.items():
        d["rate"]   = round(_smoothed(d["fraud"], d["tx"]), 6)
        d["alerts"] = d["rate"] >= ALERT_THRESHOLD
        d["name"]   = INSTITUTIONS[k]
    return agg


def consortium_view(local: dict, epsilon: float = 1.0, seed: int = 0) -> dict:
    """Combine institutions' DP-published counts into a network reputation. Each
    institution adds Laplace(1/epsilon) noise to its OWN fraud and tx counts before
    sharing, so raw data never leaves the institution. Returns the DP-combined rate."""
    rng = random.Random(seed)
    noisy_fraud = noisy_tx = 0.0
    for d in local.values():
        noisy_fraud += max(0.0, d["fraud"] + _laplace(1.0 / epsilon, rng))
        noisy_tx    += max(0.0, d["tx"]    + _laplace(1.0 / epsilon, rng))
    noisy_tx = max(noisy_tx, noisy_fraud)
    rate = round(_smoothed(noisy_fraud, noisy_tx), 6)
    evidence_tx = sum(d["tx"] for d in local.values())   # raw combined volume
    sufficient  = evidence_tx >= MIN_EVIDENCE_TX
    return {
        "combined_rate_dp":   rate,
        "epsilon":            epsilon,
        # never alert below the evidence floor: at low volume the rate is noise
        "alerts":             bool(rate >= ALERT_THRESHOLD and sufficient),
        "sufficient_evidence": sufficient,
        "evidence_tx":        evidence_tx,
        "institutions":       len(local),
    }


def build_index(edges) -> dict:
    """One pass over all transaction edges -> {recipient_id: {institution: [tx, fraud]}}.
    Built once and cached: a live scan cannot afford a JOIN per recipient."""
    from collections import defaultdict
    idx: dict = defaultdict(lambda: {k: [0, 0] for k in INSTITUTIONS})
    for recip, usr, fr in edges:
        d = idx[recip][institution_of(usr)]
        d[0] += 1
        d[1] += int(bool(fr))
    return idx


def views_from_counts(counts: dict) -> dict:
    """{institution: [tx, fraud]} -> the local_views shape (per-institution rate/alert)."""
    out = {}
    for k in INSTITUTIONS:
        tx, fraud = counts.get(k, [0, 0])
        rate = round(_smoothed(fraud, tx), 6)
        out[k] = {"tx": tx, "fraud": fraud, "rate": rate,
                  "alerts": rate >= ALERT_THRESHOLD, "name": INSTITUTIONS[k]}
    return out


def find_mules_in_index(index: dict, epsilon: float = 1.0, limit: int = 20) -> list:
    """Cross-institution mules from the cached index: flagged by the DP network yet
    below the alert line at an institution that banks with them."""
    out = []
    for rid, counts in index.items():
        local = views_from_counts(counts)
        cons = consortium_view(local, epsilon)
        blind = [k for k, d in local.items() if d["tx"] > 0 and not d["alerts"]]
        if cons["alerts"] and blind:
            out.append({
                "recipient_id": rid,
                "blind_to": [{"institution": k, "name": INSTITUTIONS[k],
                              "local_rate": local[k]["rate"], "tx": local[k]["tx"]} for k in blind],
                "institutions": {k: {"tx": d["tx"], "fraud": d["fraud"], "rate": d["rate"],
                                     "alerts": d["alerts"], "name": d["name"]} for k, d in local.items()},
                "consortium": cons,
            })
    out.sort(key=lambda m: m["consortium"]["combined_rate_dp"], reverse=True)
    return out[:limit]


def network_reveal(local: dict, querying_institution: str,
                   epsilon: float = 1.0, seed: int = 0) -> dict:
    """The payoff: does the consortium reveal a mule the querying institution could not
    see alone? True when that institution's OWN rate is below the alert line but the
    DP-combined network rate is above it - i.e. the victim's bank can now block a payee
    it had no local reason to suspect, using a signal no bank shared in the raw."""
    cons = consortium_view(local, epsilon, seed)
    mine = local.get(querying_institution, {})
    only_network = (not mine.get("alerts", False)) and cons["alerts"]
    return {
        "querying_institution":       querying_institution,
        "querying_institution_name":  INSTITUTIONS.get(querying_institution, querying_institution),
        "local_view":                 mine,
        "consortium_view":            cons,
        "only_visible_via_network":   only_network,
        "alert_threshold":            ALERT_THRESHOLD,
    }
