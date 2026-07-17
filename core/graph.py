"""
core/graph.py - a real fraud graph from the backbone (WS5 hygiene).

Replaces the Fraud Graph's curated demo with detected structure: the top mule
recipients (by fraud volume) become "rings", each drawn with the accounts paying it
and the devices they share; shared devices (used by more than one account) are
flagged, and a clean periphery is added for contrast.

This assembler is PURE: it takes a bounded list of edge dicts and builds the graph,
so it is fast and unit-testable. The caller (main.py) provides the edges from the
in-memory ledger plus the store-only demo mule (a per-recipient SQLite scan does not
scale). Returns the shape the force-graph renderer already expects.

  edge = {user, device (or None), recipient, is_fraud (0/1), amount, typology}
"""

from __future__ import annotations

from collections import Counter, defaultdict

_TONE = {
    "mule_cashout": "danger", "synthetic_identity": "purple", "pig_butchering": "orange",
    "ai_powered_ato": "warning", "app_scam": "danger", "account_takeover_ai": "warning",
    "synthetic_id_ai": "purple", "deepfake_social_engineering": "orange",
    "card_testing_bot": "success",
}


def build_fraud_graph(edges, rings: int = 4, per_ring: int = 26, clean: int = 30) -> dict:
    """Assemble a bounded fraud graph from an iterable of edge dicts."""
    rec_fraud = Counter()
    rec_fraud_edges: dict = defaultdict(list)
    clean_edges: list = []

    for e in edges:
        rid = e.get("recipient")
        if not rid or rid == "nan":
            continue
        if int(e.get("is_fraud") or 0):
            rec_fraud[rid] += 1
            if len(rec_fraud_edges[rid]) < per_ring:
                rec_fraud_edges[rid].append(e)
        elif len(clean_edges) < clean:
            clean_edges.append(e)

    # top mules by fraud volume; draw the cross-bank demo mule first
    top = sorted(rec_fraud, key=lambda r: (0 if "DEMO-MULE" in r else 1, -rec_fraud[r]))[:rings]

    nodes: dict = {}
    links: list = []
    dev_users: dict = defaultdict(set)

    def node(kind, raw, **props):
        nid = f"{kind[0]}:{raw}"
        n = nodes.get(nid)
        if n is None:
            n = nodes[nid] = {"id": nid, "type": kind, "label": str(raw),
                              "fraud_count": 0, "tx_count": 0, "ring": None, "typology": "none"}
        for k, v in props.items():
            if v is not None:
                n[k] = v
        return n

    ring_list = []
    for rid in top:
        ring_id = str(rid)[:16]
        rec = node("recipient", rid, ring=ring_id, mule_flag=True,
                   fraud_count=rec_fraud[rid], tx_count=rec_fraud[rid])
        exposure = 0.0
        typ_count = Counter()
        for e in rec_fraud_edges[rid]:
            uid = e.get("user")
            did = e.get("device")
            amt = float(e.get("amount") or 0.0)
            typ = str(e.get("typology") or "")
            if typ and typ != "none":
                typ_count[typ] += 1
            exposure += amt
            if uid:
                un = node("user", uid, ring=ring_id)
                un["tx_count"] += 1
                un["fraud_count"] += 1
                if typ and typ != "none":
                    un["typology"] = typ
                links.append({"source": f"u:{uid}", "target": f"r:{rid}", "amount": round(amt, 2), "is_fraud": True})
                if did and str(did) != "nan":
                    node("device", did, ring=ring_id)["tx_count"] += 1
                    dev_users[f"d:{did}"].add(f"u:{uid}")
                    links.append({"source": f"u:{uid}", "target": f"d:{did}", "amount": 0, "is_fraud": True})
        ring_typ = typ_count.most_common(1)[0][0] if typ_count else "none"
        rec["typology"] = ring_typ
        ring_list.append({"id": ring_id, "name": str(rid), "typology": ring_typ,
                          "exposure": round(exposure, 2), "fraud_count": rec_fraud[rid],
                          "tone": _TONE.get(ring_typ, "danger"),
                          "status": "Confirmed" if rec_fraud[rid] >= 10 else "Active"})

    # clean periphery (never reuse a mule recipient)
    for e in clean_edges:
        uid, rid = e.get("user"), e.get("recipient")
        if not rid or f"r:{rid}" in nodes:
            continue
        node("recipient", rid)["tx_count"] += 1
        if uid:
            node("user", uid)["tx_count"] += 1
            links.append({"source": f"u:{uid}", "target": f"r:{rid}",
                          "amount": round(float(e.get("amount") or 0.0), 2), "is_fraud": False})

    for did, users in dev_users.items():
        if did in nodes and len(users) >= 2:
            nodes[did]["shared_device"] = True
            nodes[did]["shared_users"]  = len(users)
    for n in nodes.values():
        if n["type"] == "user" and n["tx_count"]:
            n["fraud_score"] = round(n["fraud_count"] / n["tx_count"], 3)

    node_list = list(nodes.values())
    for r in ring_list:
        r["members"] = sum(1 for n in node_list if n["ring"] == r["id"])
    stats = {
        "total_nodes":    len(node_list),
        "total_edges":    len(links),
        "fraud_edges":    sum(1 for l in links if l["is_fraud"]),
        "mule_accounts":  sum(1 for n in node_list if n.get("mule_flag")),
        "shared_devices": sum(1 for n in node_list if n.get("shared_device")),
    }
    typologies = sorted({r["typology"] for r in ring_list if r["typology"] != "none"})
    return {"nodes": node_list, "links": links, "stats": stats,
            "rings": ring_list, "typologies": typologies}
