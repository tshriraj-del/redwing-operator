"""
core/seed_from_csv.py - one-time importer: transactions.csv -> entities + events.

The 318 MB ledger stops being something endpoints re-read on every request (the
due-diligence finding) and becomes a one-time seed of the durable backbone. After
this runs, the platform reads entities and events from SQLite, not the CSV.

What it builds:
  entities  one node per distinct user / device / recipient (small cardinality:
            ~1.4k users in the synthetic set), with first/last seen and a tx count.
            Recipients also carry raw {tx, fraud} counts in reputation, so the
            closed loop (WS2) has something to move and the network layer (WS3)
            has something to aggregate. Raw counts only - the fraud-rate math
            stays in graph_layer, the store just holds the numbers.
  events    one transaction event per row, linked to the entities it touches,
            carrying amount/rail/typology in payload and is_fraud/scores in derived.

Streaming: entities are deduped in memory (small); events are flushed in batches
so peak memory stays bounded even at 880k rows.

Run:  python3 -m core.seed_from_csv            # import everything
      python3 -m core.seed_from_csv --limit 50000   # quick partial seed
      python3 -m core.seed_from_csv --institution inst_a   # tag a tenant (WS3 uses this)
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from pathlib import Path

from .store import Entity, Event, Store, DEFAULT_DB_PATH, eid, _MODELS_DIR

csv.field_size_limit(sys.maxsize)

EVENT_BATCH = 10000


def _tobool(x) -> int:
    return 1 if str(x).strip().lower() in ("1", "true", "yes", "y", "t") else 0


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def seed(store: Store, csv_path: Path, limit: int | None = None,
         institution_id: str = "") -> dict:
    """Stream the ledger into the store. Returns import stats."""
    entities: dict[str, Entity] = {}
    ev_batch: list = []
    n_rows = n_events = 0

    def touch(kind: str, raw: str, ts: str, is_fraud: int) -> str | None:
        if not raw or raw in ("nan", ""):
            return None
        _id = eid(kind, raw)
        e = entities.get(_id)
        if e is None:
            rep = {"tx": 0, "fraud": 0} if kind == "recipient" else {}
            e = Entity(entity_id=_id, type=kind, institution_id=institution_id,
                       first_seen=ts, last_seen=ts,
                       attributes={"tx_count": 0}, reputation=rep)
            entities[_id] = e
        e.last_seen = ts
        if ts < (e.first_seen or ts):
            e.first_seen = ts
        e.attributes["tx_count"] = e.attributes.get("tx_count", 0) + 1
        if kind == "recipient":
            e.reputation["tx"] = e.reputation.get("tx", 0) + 1
            e.reputation["fraud"] = e.reputation.get("fraud", 0) + is_fraud
        return _id

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if limit and n_rows >= limit:
                break
            n_rows += 1
            ts = str(row.get("timestamp") or "") or ""
            is_fraud = _tobool(row.get("is_fraud"))

            uid = touch("user", str(row.get("user_id", "")).strip(), ts, is_fraud)
            did = touch("device", str(row.get("device_id", "")).strip(), ts, is_fraud)
            rid = touch("recipient", str(row.get("recipient_id", "")).strip(), ts, is_fraud)

            ev_batch.append(Event(
                event_id=str(row.get("transaction_id") or uuid.uuid4().hex),
                event_type="transaction",
                ts=ts or "",
                institution_id=institution_id,
                entities=[e for e in (uid, did, rid) if e],
                payload={
                    "amount":        _f(row.get("amount")),
                    "rail":          str(row.get("payment_rail", "") or ""),
                    "typology":      str(row.get("fraud_typology", "") or ""),
                    "merchant_category": str(row.get("merchant_category", "") or ""),
                },
                derived={
                    "is_fraud":       is_fraud,
                    "ensemble_score": _f(row.get("ensemble_score")),
                    "rule_score":     _f(row.get("rule_score")),
                },
            ))
            if len(ev_batch) >= EVENT_BATCH:
                n_events += store.bulk_append_events(ev_batch)
                ev_batch = []

    if ev_batch:
        n_events += store.bulk_append_events(ev_batch)
    n_entities = store.bulk_upsert_entities(entities.values())

    return {"rows_read": n_rows, "events_written": n_events,
            "entities_written": n_entities,
            "by_type": {k: sum(1 for e in entities.values() if e.type == k)
                        for k in ("user", "device", "recipient")}}


def main():
    ap = argparse.ArgumentParser(description="Seed the REDWING backbone from transactions.csv")
    ap.add_argument("--csv", default=str(_MODELS_DIR / "transactions.csv"))
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--institution", default="")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}\n"
              f"(the synthetic ledger is gitignored; regenerate it or point --csv at it)")
        sys.exit(1)

    print(f"Seeding backbone: {csv_path.name} -> {args.db}"
          + (f"  (limit {args.limit:,})" if args.limit else "  (full)")
          + (f"  institution={args.institution}" if args.institution else ""))
    store = Store(args.db)
    stats = seed(store, csv_path, limit=args.limit, institution_id=args.institution)
    print(f"  rows read:        {stats['rows_read']:,}")
    print(f"  events written:   {stats['events_written']:,}")
    print(f"  entities written: {stats['entities_written']:,}  {stats['by_type']}")
    print("  store stats:", store.stats()["events_by_type"])
    store.close()
    print("Done.")


if __name__ == "__main__":
    main()
