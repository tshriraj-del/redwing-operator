"""
core/authorization_iq.py - authorization-time network intelligence for PUSH rails.

WHY THIS EXISTS. The feature dictionaries the big card issuers score on are full of fields
like `ffsl_mc_aqf_frq_ovr_spnd_insg` - Mastercard AUTHORIZATION IQ fields. They give an issuer,
at authorization time, signals it could never compute from its own book: how this spend
compares to the whole population, how the merchant looks across every OTHER issuer, cross-issuer
velocity. The card network can do this because it sees every issuer's traffic. That consortium
network effect is real, it is enormously valuable, and for CARDS it already exists and the card
networks own it.

It does not exist for the PUSH rails - Zelle, FedNow, RTP, wire, crypto - which is where the
authorised-push-payment scam lives. There, each bank sees only its own leg: the victim's bank
sees an ordinary-looking outgoing payment, the mule's bank sees an ordinary-looking inbound.
Neither sees the whole, so neither can produce a network-level signal at the moment it matters,
which is BEFORE the irrevocable payment settles.

So the move is not to compete with Authorization IQ. It is to BE Authorization IQ for the rails
it does not cover. This module turns the privacy-preserving cross-institution consortium
(core/consortium.py) into an authorization-time INSIGHT PACK: the push-rail analog of the AQF
fields, returned as decision-ready codes with a composed network-risk contribution.

THE ONE IDEA that makes this different from a normal feature vector: every field carries its
NETWORK DELTA - the network's view minus what the querying bank could see on its own. A field
whose network delta is zero told the bank nothing it did not already know. The value of the
consortium is exactly the sum of those deltas, and the headline field is the reveal: a payee
below the alert line at YOUR bank but above it across the network - the mule you would have
paid, caught with a signal no bank shared in the raw.

HONESTY. Demonstrated at n=2 (two fictional institutions, see consortium.py). What is being
shown is the STRUCTURE - that the network sees what the single bank cannot - not a production
number. Below the consortium's evidence floor the pack stays silent rather than alert on noise.
Synthetic data throughout; nothing here is a measurement about real payees.

Pure stdlib, deterministic, unit-testable. Composes consortium.py rather than reimplementing DP.

    idx = build_index(edges)                         # edges: {recipient, sender, amount, rail, is_fraud}
    pack = authorize(payment, idx, querying_institution="inst_neobank")
"""

from __future__ import annotations

import math
from collections import defaultdict

from . import consortium as C

# -- Reason codes. Authorization IQ returns codes, not just a score; a real integration keys
# playbook actions off them, and they make the pack auditable. Each is a network-derived
# insight that a single institution could not have produced from its own book alone.
AIQ_CODES = {
    "AIQ01_NETWORK_MULE":     "Payee is a known mule across the network, below your local alert line",
    "AIQ02_CROSS_BANK_FANIN": "Payee receives from many unrelated senders across multiple institutions",
    "AIQ03_AMOUNT_OVER_NORM": "Amount is far above the network norm for this rail",
    "AIQ04_NEW_TO_NETWORK":   "Payee is new to the entire network, not just to you",
    "AIQ05_CASHOUT_CORRIDOR": "Sender-to-payee corridor is a known irrevocable cash-out path",
    "AIQ06_NETWORK_REVEAL":   "Network reveals risk you had no local reason to see",
}

# Thresholds. Named, sourced to a rationale, changed here not inline.
FANIN_ALERT = 12          # a personal payee taking money from this many DISTINCT senders is not
                          # a person being repaid; it is a collector. Merchants are excluded by
                          # the multi-institution test below, not by a raw count.
FANIN_SATURATE = 60       # fan-in risk reaches 1.0 here
Z_ALERT = 3.0             # amount z-score vs the rail's network norm at which the insight fires
Z_SATURATE = 8.0
NEW_TO_NETWORK_TX = 5     # at or below this network-wide tx count a payee is "new to the network"
MIN_RAIL_NORM_N = 30      # a rail needs this many network observations before its norm is trusted

# How the individual insight risks compose into the single authorization-time network signal.
# Reputation and the reveal dominate because they are the consortium's core product; the others
# are corroborating. Weights are a modelling choice, documented as such, not a measurement.
COMPOSE_WEIGHTS = {
    "network_rep": 0.34, "reveal": 0.24, "fanin": 0.18,
    "amount_norm": 0.10, "new_to_network": 0.08, "corridor": 0.06,
}

# Institutions whose inbound is a classic irrevocable cash-out (the off-ramp leg of a scam).
_CASHOUT_INSTITUTIONS = {"inst_crypto"}
_IRREVOCABLE_RAILS = {"crypto", "wire", "FedNow", "RTP", "Zelle"}


def _norm(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 ramp from lo to hi, clamped. Below lo is 0 (no signal), above hi is 1."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# -- The network index: precomputed once from the edge stream, queried per authorization -------

class AIQIndex:
    """Precomputed network aggregates. Built once from all edges (a per-payment scan of the
    ledger cannot run at authorization latency), then queried in O(1) per payment.

    Deliberately stores COUNTS, not raw rows: distinct-sender SETS are collapsed to counts at
    finalize so the index does not hold a copy of the ledger. The DP consortium index is the
    only cross-institution structure retained, and it already only holds noised-able counts.
    """

    def __init__(self):
        self.consortium_index: dict = {}          # recipient -> {institution: [tx, fraud]}
        self.fanin: dict = {}                     # recipient -> distinct sender count (network-wide)
        self.fanin_by_inst: dict = {}             # recipient -> {institution: distinct sender count}
        self.recipient_tx: dict = {}              # recipient -> total network tx count
        self.rail_norm: dict = {}                 # rail -> {"n","mean","std"}
        self.network_tx_total: int = 0

    def local_fanin(self, recipient: str, institution: str) -> int:
        """Distinct senders THIS institution can see paying the payee - the single-bank view,
        used to compute the fan-in network delta."""
        return self.fanin_by_inst.get(recipient, {}).get(institution, 0)


def build_index(edges) -> AIQIndex:
    """Build the network index from an iterable of edge dicts:
        {recipient, sender, amount, rail, is_fraud}
    The caller supplies edges (from the in-memory ledger), matching how core/graph.py and the
    consortium index are fed, so this module stays pure and testable.
    """
    idx = AIQIndex()
    cons_raw: dict = defaultdict(lambda: {k: [0, 0] for k in C.INSTITUTIONS})
    senders: dict = defaultdict(set)                              # recipient -> {sender}
    senders_by_inst: dict = defaultdict(lambda: defaultdict(set))  # recipient -> inst -> {sender}
    rec_tx: dict = defaultdict(int)
    # Welford per rail for a numerically stable network mean/std of amounts.
    rail_acc: dict = defaultdict(lambda: [0, 0.0, 0.0])          # rail -> [n, mean, M2]

    n_total = 0
    for e in edges:
        rid = e.get("recipient")
        uid = e.get("sender")
        if not rid or not uid:
            continue
        n_total += 1
        fr = int(bool(e.get("is_fraud")))
        inst = C.institution_of(uid)
        cons_raw[rid][inst][0] += 1
        cons_raw[rid][inst][1] += fr
        senders[rid].add(uid)
        senders_by_inst[rid][inst].add(uid)
        rec_tx[rid] += 1

        rail = str(e.get("rail") or "").strip() or "unknown"
        try:
            amt = float(e.get("amount") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt > 0:
            a = rail_acc[rail]
            a[0] += 1
            delta = amt - a[1]
            a[1] += delta / a[0]
            a[2] += delta * (amt - a[1])

    idx.consortium_index = {r: dict(v) for r, v in cons_raw.items()}
    idx.fanin = {r: len(s) for r, s in senders.items()}
    idx.fanin_by_inst = {r: {k: len(s) for k, s in per.items()}
                         for r, per in senders_by_inst.items()}
    idx.recipient_tx = dict(rec_tx)
    idx.network_tx_total = n_total
    for rail, (n, mean, m2) in rail_acc.items():
        std = math.sqrt(m2 / n) if n > 1 else 0.0
        idx.rail_norm[rail] = {"n": n, "mean": round(mean, 2), "std": round(std, 2)}
    return idx


# -- The insight fields. Each returns a dict with the value, the NETWORK DELTA, a risk in 0..1,
#    and whether it fired (a reason code). The network delta is the whole point: it is what the
#    consortium added on top of what the querying bank already knew.

def _insight_network_rep(recipient, idx, querying_institution, epsilon):
    counts = idx.consortium_index.get(recipient, {})
    local = C.views_from_counts(counts)
    reveal = C.network_reveal(local, querying_institution, epsilon)
    cons = reveal["consortium_view"]
    mine = reveal["local_view"] or {}
    combined = cons["combined_rate_dp"]
    local_rate = float(mine.get("rate", C.PRIOR))
    # risk scales the network rate against the consortium's own alert threshold
    risk = _norm(combined, C.ALERT_THRESHOLD, C.ALERT_THRESHOLD * 6) if cons["sufficient_evidence"] else 0.0
    fired = bool(cons["alerts"])
    reveal_fired = bool(reveal["only_visible_via_network"])
    return {
        "field": "aiq_recipient_network_rep",
        "value": combined,
        "local_value": round(local_rate, 6),
        "network_delta": round(combined - local_rate, 6),
        "risk": round(risk, 3),
        "fired": fired,
        "code": "AIQ01_NETWORK_MULE" if fired else None,
        "reveal": reveal_fired,
        "sufficient_evidence": cons["sufficient_evidence"],
        "note": (f"Network fraud rate on this payee is {combined:.1%} across "
                 f"{cons['institutions']} institutions; your book alone shows {local_rate:.1%}."),
    }, reveal


def _insight_fanin(recipient, idx, querying_institution):
    total = idx.fanin.get(recipient, 0)
    per_inst = idx.fanin_by_inst.get(recipient, {})
    local = per_inst.get(querying_institution, 0)
    institutions_seeing = sum(1 for v in per_inst.values() if v > 0)
    # a collector is a payee taking from many DISTINCT senders across MORE THAN ONE institution.
    # the multi-institution test is what separates a mule from a legitimate merchant, which also
    # has high fan-in but concentrates within the acquirer that banks it.
    multi_bank = institutions_seeing >= 2
    risk = _norm(total, FANIN_ALERT, FANIN_SATURATE) if multi_bank else 0.0
    fired = multi_bank and total >= FANIN_ALERT
    return {
        "field": "aiq_recipient_fanin",
        "value": total,
        "local_value": local,
        "network_delta": total - local,
        "institutions_seeing": institutions_seeing,
        "risk": round(risk, 3),
        "fired": fired,
        "code": "AIQ02_CROSS_BANK_FANIN" if fired else None,
        "note": (f"{total} distinct senders across {institutions_seeing} institutions pay this "
                 f"payee; you can see {local}. Cross-bank fan-in is the collector-mule signature."),
    }


def _insight_amount_over_norm(amount, rail, idx):
    norm = idx.rail_norm.get(rail)
    if not norm or norm["n"] < MIN_RAIL_NORM_N or norm["std"] <= 0:
        return {"field": "aiq_amount_over_norm", "value": amount, "risk": 0.0, "fired": False,
                "code": None, "network_delta": None,
                "note": f"No trusted network norm for the {rail} rail yet."}
    z = (amount - norm["mean"]) / norm["std"]
    risk = _norm(z, Z_ALERT, Z_SATURATE)
    fired = z >= Z_ALERT
    return {
        "field": "aiq_amount_over_norm",
        "value": amount,
        "z_vs_network": round(z, 2),
        "network_mean": norm["mean"],
        "network_std": norm["std"],
        "network_delta": round(z, 2),
        "risk": round(risk, 3),
        "fired": fired,
        "code": "AIQ03_AMOUNT_OVER_NORM" if fired else None,
        "note": (f"${amount:,.0f} is {z:.1f} sigma above the network norm for {rail} "
                 f"(${norm['mean']:,.0f} +/- ${norm['std']:,.0f})."),
    }


def _insight_new_to_network(recipient, amount, idx, local_seen):
    net_tx = idx.recipient_tx.get(recipient, 0)
    new_to_network = net_tx <= NEW_TO_NETWORK_TX
    # new-to-you but ESTABLISHED on the network is the reassuring case and must lower risk, not
    # raise it: the whole point is to stop treating "I have never paid them" as suspicious when
    # the network knows the payee is fine.
    established_elsewhere = (local_seen == 0 and net_tx > NEW_TO_NETWORK_TX)
    amt_weight = _norm(amount, 500, 10000)
    risk = _norm(NEW_TO_NETWORK_TX - net_tx, 0, NEW_TO_NETWORK_TX) * amt_weight if new_to_network else 0.0
    return {
        "field": "aiq_recipient_network_newness",
        "value": net_tx,
        "network_delta": net_tx - local_seen,
        "new_to_network": new_to_network,
        "established_elsewhere": established_elsewhere,
        "risk": round(risk, 3),
        "fired": bool(new_to_network and risk > 0),
        "code": "AIQ04_NEW_TO_NETWORK" if (new_to_network and risk > 0) else None,
        "note": ("Payee is new to the entire network." if new_to_network
                 else "Payee is new to you but established across the network (reassuring)."
                 if established_elsewhere else "Payee has network history."),
    }


def _insight_corridor(recipient, rail, idx, querying_institution):
    counts = idx.consortium_index.get(recipient, {})
    # where does the payee predominantly RECEIVE? a payee concentrated at a crypto off-ramp,
    # paid from a different (deposit) institution on an irrevocable rail, is the classic
    # scam cash-out corridor.
    recv_tx = {k: v[0] for k, v in counts.items()}
    total = sum(recv_tx.values()) or 1
    cashout_share = sum(recv_tx.get(k, 0) for k in _CASHOUT_INSTITUTIONS) / total
    irrevocable = rail in _IRREVOCABLE_RAILS
    cross_corridor = querying_institution not in _CASHOUT_INSTITUTIONS
    risk = cashout_share * (1.0 if irrevocable else 0.4) * (1.0 if cross_corridor else 0.5)
    fired = risk >= 0.4
    return {
        "field": "aiq_cashout_corridor",
        "value": round(cashout_share, 3),
        "irrevocable_rail": irrevocable,
        "network_delta": round(cashout_share, 3),   # your book cannot see where the payee cashes out
        "risk": round(risk, 3),
        "fired": fired,
        "code": "AIQ05_CASHOUT_CORRIDOR" if fired else None,
        "note": (f"{cashout_share:.0%} of this payee's network inbound settles at an irrevocable "
                 f"cash-out; you are sending on {rail}."),
    }


def authorize(payment: dict, index: AIQIndex, querying_institution: str = None,
              epsilon: float = 1.0) -> dict:
    """The authorization-time network intelligence pack for one push payment.

    payment: {sender, recipient, amount, rail, timestamp?}. Returns the AQF-style insight fields
    (each with its network delta), the composed authorization-time network_risk, the reason
    codes that fired, and the reveal headline. querying_institution defaults to the sender's
    institution - the bank asking is the bank the sender banks with.
    """
    sender = payment.get("sender")
    recipient = payment.get("recipient")
    rail = str(payment.get("rail") or "").strip() or "unknown"
    try:
        amount = float(payment.get("amount") or 0.0)
    except (TypeError, ValueError):
        amount = 0.0
    qi = querying_institution or (C.institution_of(sender) if sender else next(iter(C.INSTITUTIONS)))
    local_seen = index.local_fanin(recipient, qi) if recipient else 0

    rep, reveal = _insight_network_rep(recipient, index, qi, epsilon)
    fanin = _insight_fanin(recipient, index, qi)
    amt = _insight_amount_over_norm(amount, rail, index)
    newness = _insight_new_to_network(recipient, amount, index, local_seen)
    corridor = _insight_corridor(recipient, rail, index, qi)

    reveal_fired = rep["reveal"]
    reveal_risk = rep["risk"] if reveal_fired else 0.0

    # compose the single authorization-time network signal
    parts = {
        "network_rep": rep["risk"],
        "reveal": reveal_risk,
        "fanin": fanin["risk"],
        "amount_norm": amt["risk"],
        "new_to_network": newness["risk"],
        "corridor": corridor["risk"],
    }
    network_risk = round(min(1.0, sum(COMPOSE_WEIGHTS[k] * v for k, v in parts.items())), 3)

    insights = [rep, fanin, amt, newness, corridor]
    codes = [i["code"] for i in insights if i.get("code")]
    if reveal_fired and "AIQ06_NETWORK_REVEAL" not in codes:
        codes.append("AIQ06_NETWORK_REVEAL")

    # the pack is only as trustworthy as the evidence behind the reputation field
    sufficient = rep["sufficient_evidence"]

    # plain-language headline: lead with the reveal, because that is the thing a single bank
    # could not have done. otherwise lead with the strongest fired insight.
    if reveal_fired:
        headline = ("The network reveals this payee as a mule you had no local reason to flag: "
                    f"below your alert line, above the network's.")
    elif codes:
        top = max(insights, key=lambda i: i["risk"])
        headline = top["note"]
    else:
        headline = "The network adds no risk to this payment beyond what you can already see."

    return {
        "querying_institution": qi,
        "querying_institution_name": C.INSTITUTIONS.get(qi, qi),
        "network_risk": network_risk,
        "network_reveal": reveal_fired,
        "reason_codes": [{"code": c, "label": AIQ_CODES[c]} for c in codes],
        "insights": insights,
        "sufficient_evidence": sufficient,
        "total_network_delta": round(sum(
            abs(i["network_delta"]) for i in insights
            if isinstance(i.get("network_delta"), (int, float))), 4),
        "explanation": headline,
        "epsilon": epsilon,
    }


if __name__ == "__main__":
    # A worked example: a mule that is CLEAN at the victim's bank (Northwind Neobank) and
    # fraudulent at the crypto off-ramp (Coastline Exchange). The victim's bank is about to
    # send an ordinary-looking $6,000 Zelle to it. Alone, the neobank has no reason to hesitate.
    import random as _r

    def _senders(inst, n, salt):
        out, i = [], 0
        while len(out) < n:
            u = f"user_{salt}_{i}"
            if C.institution_of(u) == inst:
                out.append(u)
            i += 1
        return out

    rng = _r.Random(7)
    edges = []
    # the neobank side: 40 ordinary customers who have paid this payee, none fraud
    for s in _senders("inst_neobank", 40, "clean"):
        edges.append({"recipient": "recipient:mule_9f2", "sender": s, "is_fraud": 0,
                      "amount": rng.uniform(80, 400), "rail": "Zelle"})
    # the crypto side: 35 senders, most confirmed fraud - the cash-out leg the neobank cannot see
    for i, s in enumerate(_senders("inst_crypto", 35, "dirty")):
        edges.append({"recipient": "recipient:mule_9f2", "sender": s, "is_fraud": int(i < 24),
                      "amount": rng.uniform(3000, 12000), "rail": "crypto"})
    # background so the Zelle rail has a network norm
    for i in range(200):
        edges.append({"recipient": f"recipient:ord_{i%50}", "sender": f"user_bg_{i}",
                      "is_fraud": 0, "amount": rng.uniform(50, 600), "rail": "Zelle"})

    idx = build_index(edges)
    victim = _senders("inst_neobank", 1, "victim")[0]
    pack = authorize({"sender": victim, "recipient": "recipient:mule_9f2",
                      "amount": 6000.0, "rail": "Zelle"}, idx)

    print("AUTHORIZATION IQ  -  push-rail network intelligence at decision time")
    print(f"  querying bank : {pack['querying_institution_name']} (the victim's bank)")
    print(f"  payment       : $6,000 Zelle to recipient:mule_9f2\n")
    print(f"  {'field':34s} {'your book':>10s} {'network':>10s} {'delta':>10s}  fired")
    for i in pack["insights"]:
        lv = i.get("local_value", "-")
        v = i.get("value", "-")
        d = i.get("network_delta", "-")
        fv = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
        print(f"  {i['field']:34s} {fv(lv):>10s} {fv(v):>10s} {fv(d):>10s}  "
              f"{'YES' if i['fired'] else ''}")
    print(f"\n  network_risk  : {pack['network_risk']}  (composed authorization-time signal)")
    print(f"  network_reveal: {pack['network_reveal']}")
    print(f"  reason codes  : {', '.join(c['code'] for c in pack['reason_codes']) or '(none)'}")
    print(f"\n  {pack['explanation']}")
    print("\n  Every 'network' column the neobank could not have produced from its own book. "
          "That gap\n  is the consortium's product, delivered before an irrevocable payment "
          "settles. (n=2, synthetic.)")
