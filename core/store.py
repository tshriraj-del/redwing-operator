"""
core/store.py - the entity + event backbone (Phase 1, WS0).

One durable store holds two things:

  ENTITIES  the nodes of the fraud graph: users, devices, recipients, accounts,
            institutions. Each carries a reputation dict that the closed loop
            (WS2) updates online, and an institution_id that makes the network
            real (WS3).

  EVENTS    an append-only log of everything that happens: a transaction scored,
            an alert raised, an analyst disposition, an enrichment, a feedback
            label, a model update. Events reference the entities they touch.

Design choices (the why):
  * SQLite, not a JSON file and not Postgres. It gives real durability across a
    restart (the due-diligence gap) with zero ops, and it uses the SAME schema
    Postgres will use in Phase 2, so the migration is a connection string, not a
    rewrite. WAL mode so reads never block on a write.
  * Normalised event<->entity join table, so "every event touching recipient X"
    is an index lookup. WS2 (find pending payments to a just-confirmed mule) and
    WS3 (aggregate a recipient across institutions) both need exactly that.
  * The store is deliberately generic. It knows nothing about fraud math. The
    empirical-Bayes reputation logic stays in graph_layer / feedback; the store
    only persists whatever reputation dict they hand it. One substrate, many
    organs.

Python 3.9+ (stdlib only: sqlite3, json, uuid, datetime, threading).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


# ── Location ──────────────────────────────────────────────────────────────────
# Co-located with the ML artifacts so one directory holds the platform's state.
# Override with REDWING_MODELS_DIR (same knob main.py already uses).
_MODELS_DIR = Path(os.environ.get("REDWING_MODELS_DIR", Path.home() / "pulseml_models"))
DEFAULT_DB_PATH = _MODELS_DIR / "redwing.db"


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _loads(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ── Records ───────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A node in the fraud graph. `reputation` is updated online by the loop."""
    entity_id:      str
    type:           str                       # user | device | recipient | account | institution
    institution_id: str = ""
    first_seen:     str = ""
    last_seen:      str = ""
    attributes:     dict = field(default_factory=dict)
    reputation:     dict = field(default_factory=dict)


@dataclass
class Event:
    """An append-only entry on the backbone. `entities` are the ids it touches."""
    event_id:       str
    event_type:     str                       # transaction | alert | disposition | enrichment | feedback | model_update
    ts:             str
    institution_id: str = ""
    entities:       list = field(default_factory=list)
    payload:        dict = field(default_factory=dict)
    derived:        dict = field(default_factory=dict)


# Recognised event types (a soft contract, not enforced - the log stays open).
EVENT_TYPES = (
    "transaction", "alert", "disposition", "enrichment", "feedback", "model_update",
)
ENTITY_TYPES = ("user", "device", "recipient", "account", "institution")


# ── Store ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id      TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    institution_id TEXT NOT NULL DEFAULT '',
    first_seen     TEXT,
    last_seen      TEXT,
    attributes     TEXT NOT NULL DEFAULT '{}',
    reputation     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_inst ON entities(institution_id);

CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    event_type     TEXT NOT NULL,
    ts             TEXT NOT NULL,
    institution_id TEXT NOT NULL DEFAULT '',
    payload        TEXT NOT NULL DEFAULT '{}',
    derived        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id  TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_ee_entity ON event_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_ee_event  ON event_entities(event_id);
"""


class Store:
    """Durable entity + event backbone. Thread-safe for the operator's executor
    pool: WAL gives concurrent reads; a lock serialises writes."""

    def __init__(self, db_path: os.PathLike | str = DEFAULT_DB_PATH):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI runs handlers across a thread pool.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── Entities ──────────────────────────────────────────────────────────────

    def upsert_entity(
        self,
        entity_id: str,
        type: str,
        institution_id: str = "",
        attributes: Optional[dict] = None,
        reputation: Optional[dict] = None,
        ts: Optional[str] = None,
    ) -> None:
        """Insert or touch an entity. On an existing row: bump last_seen, and
        shallow-merge any attributes/reputation supplied (existing keys win only
        where the caller does not override). first_seen is set once and kept."""
        ts = ts or _now()
        attributes = attributes or {}
        reputation = reputation or {}
        with self._lock:
            row = self._conn.execute(
                "SELECT attributes, reputation FROM entities WHERE entity_id=?",
                (entity_id,),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO entities (entity_id, type, institution_id, first_seen, "
                    "last_seen, attributes, reputation) VALUES (?,?,?,?,?,?,?)",
                    (entity_id, type, institution_id, ts, ts,
                     json.dumps(attributes), json.dumps(reputation)),
                )
            else:
                merged_attr = {**_loads(row["attributes"]), **attributes}
                merged_rep = {**_loads(row["reputation"]), **reputation}
                self._conn.execute(
                    "UPDATE entities SET last_seen=?, institution_id="
                    "CASE WHEN ?<>'' THEN ? ELSE institution_id END, "
                    "attributes=?, reputation=? WHERE entity_id=?",
                    (ts, institution_id, institution_id,
                     json.dumps(merged_attr), json.dumps(merged_rep), entity_id),
                )
            self._conn.commit()

    def bulk_upsert_entities(self, entities: Iterable[Entity], batch: int = 5000) -> int:
        """Fast path for the CSV importer: UPSERT many entities via executemany +
        ON CONFLICT (one SELECT-per-row would be unusable at 880k rows). On
        conflict, bump last_seen and keep the earliest first_seen; attributes and
        reputation are replaced wholesale (the importer owns them at load time)."""
        sql = (
            "INSERT INTO entities (entity_id, type, institution_id, first_seen, "
            "last_seen, attributes, reputation) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(entity_id) DO UPDATE SET "
            "last_seen=excluded.last_seen, "
            "first_seen=MIN(entities.first_seen, excluded.first_seen), "
            "institution_id=CASE WHEN excluded.institution_id<>'' "
            "THEN excluded.institution_id ELSE entities.institution_id END, "
            "attributes=excluded.attributes, reputation=excluded.reputation"
        )
        n = 0
        rows = []
        with self._lock:
            for e in entities:
                rows.append((e.entity_id, e.type, e.institution_id,
                             e.first_seen or _now(), e.last_seen or _now(),
                             json.dumps(e.attributes), json.dumps(e.reputation)))
                if len(rows) >= batch:
                    self._conn.executemany(sql, rows); n += len(rows); rows = []
            if rows:
                self._conn.executemany(sql, rows); n += len(rows)
            self._conn.commit()
        return n

    def bulk_append_events(self, events: Iterable[Event], batch: int = 5000) -> int:
        """Fast path for the CSV importer: append many events + their entity links."""
        n = 0
        ev_rows, link_rows = [], []
        with self._lock:
            for e in events:
                ev_rows.append((e.event_id, e.event_type, e.ts, e.institution_id,
                                json.dumps(e.payload), json.dumps(e.derived)))
                link_rows.extend((e.event_id, ent) for ent in e.entities if ent)
                if len(ev_rows) >= batch:
                    self._flush_events(ev_rows, link_rows); n += len(ev_rows)
                    ev_rows, link_rows = [], []
            if ev_rows:
                self._flush_events(ev_rows, link_rows); n += len(ev_rows)
            self._conn.commit()
        return n

    def _flush_events(self, ev_rows: list, link_rows: list) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO events (event_id, event_type, ts, "
            "institution_id, payload, derived) VALUES (?,?,?,?,?,?)", ev_rows)
        self._conn.executemany(
            "INSERT OR IGNORE INTO event_entities (event_id, entity_id) VALUES (?,?)",
            link_rows)

    def update_reputation(self, entity_id: str, patch: dict) -> Optional[Entity]:
        """Merge keys into an entity's reputation dict (the loop's write path).
        Returns the refreshed entity, or None if it does not exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT reputation FROM entities WHERE entity_id=?", (entity_id,)
            ).fetchone()
            if row is None:
                return None
            merged = {**_loads(row["reputation"]), **patch}
            self._conn.execute(
                "UPDATE entities SET reputation=?, last_seen=? WHERE entity_id=?",
                (json.dumps(merged), _now(), entity_id),
            )
            self._conn.commit()
        return self.get_entity(entity_id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE entity_id=?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def entities_by_type(self, type: str, institution_id: Optional[str] = None,
                         limit: int = 1000) -> list:
        if institution_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE type=? AND institution_id=? LIMIT ?",
                (type, institution_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE type=? LIMIT ?", (type, limit)
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Events ────────────────────────────────────────────────────────────────

    def append_event(
        self,
        event_type: str,
        entities: Optional[Iterable[str]] = None,
        payload: Optional[dict] = None,
        derived: Optional[dict] = None,
        institution_id: str = "",
        ts: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Append one event and link it to the entities it touches. Returns the
        event_id. This is the platform's single write verb for 'something happened'."""
        event_id = event_id or uuid.uuid4().hex
        ts = ts or _now()
        ents = [e for e in (entities or []) if e]
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events (event_id, event_type, ts, "
                "institution_id, payload, derived) VALUES (?,?,?,?,?,?)",
                (event_id, event_type, ts, institution_id,
                 json.dumps(payload or {}), json.dumps(derived or {})),
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO event_entities (event_id, entity_id) VALUES (?,?)",
                [(event_id, e) for e in ents],
            )
            self._conn.commit()
        return event_id

    def events_for_entity(self, entity_id: str, event_type: Optional[str] = None,
                          limit: int = 200) -> list:
        """Every event touching an entity, newest first. The query WS2 and WS3
        are built on (pending payments to a mule; a recipient across banks)."""
        # Filter by entity FIRST (idx_ee_entity) via a subquery, then order/limit the
        # small result. A plain JOIN + ORDER BY e.ts makes the planner scan the entire
        # events table by the ts index, which is catastrophic for a low-volume entity
        # (measured 53s for a 19-event payee); the subquery keeps it in milliseconds.
        q = ("SELECT * FROM (SELECT e.event_id, e.event_type, e.ts, e.institution_id, "
             "e.payload, e.derived FROM events e WHERE e.event_id IN "
             "(SELECT event_id FROM event_entities WHERE entity_id=?)")
        args: list = [entity_id]
        if event_type:
            q += " AND e.event_type=?"
            args.append(event_type)
        q += ") ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [self._row_to_event(r) for r in self._conn.execute(q, args).fetchall()]

    def entity_event_summary(self, entity_id: str, event_type: str = "transaction") -> dict:
        """Fast aggregate for the loop receipt: count + summed payload.amount of the
        events touching an entity, in ONE query. Avoids materialising Event objects
        (the per-event entity-link lookup would be an N+1 over a large store)."""
        row = self._conn.execute(
            "SELECT COUNT(*) n, "
            "COALESCE(SUM(CAST(json_extract(e.payload,'$.amount') AS REAL)),0) amt, "
            "COALESCE(SUM(CAST(json_extract(e.derived,'$.expected_liability') AS REAL)),0) liab "
            "FROM events e JOIN event_entities ee ON e.event_id=ee.event_id "
            "WHERE ee.entity_id=? AND e.event_type=?", (entity_id, event_type),
        ).fetchone()
        return {"count": int(row["n"]), "exposure": round(float(row["amt"] or 0.0), 2),
                "liability": round(float(row["liab"] or 0.0), 2)}

    def liability_at_risk(self, event_type: str = "alert") -> dict:
        """Portfolio liability-at-risk: expected reimbursement dollars summed over open
        events of a type (default alerts). The number a fraud-ops lead reports upward."""
        row = self._conn.execute(
            "SELECT COUNT(*) n, "
            "COALESCE(SUM(CAST(json_extract(derived,'$.expected_liability') AS REAL)),0) liab "
            "FROM events WHERE event_type=?", (event_type,),
        ).fetchone()
        return {"events": int(row["n"]), "liability_at_risk": round(float(row["liab"] or 0.0), 2)}

    def recipient_sender_labels(self, recipient_id: str, limit: int = 5000) -> list:
        """For a payee, return (sender_user_entity_id, is_fraud) for each transaction
        touching it - the raw material the consortium aggregates per institution. One
        JOIN query (recipient link x user link on the same event), no N+1."""
        rows = self._conn.execute(
            "SELECT eu.entity_id uid, "
            "CAST(json_extract(e.derived,'$.is_fraud') AS INTEGER) fr "
            "FROM events e "
            "JOIN event_entities er ON e.event_id=er.event_id AND er.entity_id=? "
            "JOIN event_entities eu ON e.event_id=eu.event_id "
            "WHERE e.event_type='transaction' AND eu.entity_id LIKE 'user:%' "
            "LIMIT ?", (recipient_id, limit),
        ).fetchall()
        return [(r["uid"], int(r["fr"] or 0)) for r in rows]

    def all_transaction_edges(self):
        """Stream (recipient_entity_id, sender_user_entity_id, is_fraud) for every
        transaction, in ONE pass. The raw material for the cross-institution index,
        which is built once and cached (per-recipient JOINs do not scale to a scan)."""
        cur = self._conn.execute(
            "SELECT er.entity_id recip, eu.entity_id usr, "
            "CAST(json_extract(e.derived,'$.is_fraud') AS INTEGER) fr "
            "FROM events e "
            "JOIN event_entities er ON e.event_id=er.event_id AND er.entity_id LIKE 'recipient:%' "
            "JOIN event_entities eu ON e.event_id=eu.event_id AND eu.entity_id LIKE 'user:%' "
            "WHERE e.event_type='transaction'")
        for row in cur:
            yield row["recip"], row["usr"], int(row["fr"] or 0)

    def fraudy_recipients(self, min_fraud: int = 3, limit: int = 400) -> list:
        """Candidate mules: recipient entities whose seeded reputation shows at least
        `min_fraud` confirmed frauds, most-fraudulent first. The cheap pre-filter so
        the cross-institution scan only inspects plausible payees."""
        rows = self._conn.execute(
            "SELECT entity_id, "
            "CAST(json_extract(reputation,'$.fraud') AS INTEGER) f, "
            "CAST(json_extract(reputation,'$.tx') AS INTEGER) t "
            "FROM entities WHERE type='recipient' "
            "AND CAST(json_extract(reputation,'$.fraud') AS INTEGER) >= ? "
            "ORDER BY f DESC LIMIT ?", (min_fraud, limit),
        ).fetchall()
        return [(r["entity_id"], int(r["f"] or 0), int(r["t"] or 0)) for r in rows]

    def count_events(self, event_type: Optional[str] = None) -> int:
        """Indexed COUNT of events (optionally by type). Cheap - no GROUP BY scan."""
        if event_type:
            r = self._conn.execute(
                "SELECT COUNT(*) n FROM events WHERE event_type=?", (event_type,)).fetchone()
        else:
            r = self._conn.execute("SELECT COUNT(*) n FROM events").fetchone()
        return int(r["n"])

    def recent_events(self, event_type: Optional[str] = None, limit: int = 100) -> list:
        if event_type:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE event_type=? ORDER BY ts DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        c = self._conn
        by_type = {r["type"]: r["n"] for r in c.execute(
            "SELECT type, COUNT(*) n FROM entities GROUP BY type").fetchall()}
        by_event = {r["event_type"]: r["n"] for r in c.execute(
            "SELECT event_type, COUNT(*) n FROM events GROUP BY event_type").fetchall()}
        institutions = [r["institution_id"] for r in c.execute(
            "SELECT DISTINCT institution_id FROM entities WHERE institution_id<>''").fetchall()]
        return {
            "db_path":        self.path,
            "entities_total": sum(by_type.values()),
            "entities_by_type": by_type,
            "events_total":   sum(by_event.values()),
            "events_by_type": by_event,
            "institutions":   institutions,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── Row mappers ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_entity(r: sqlite3.Row) -> Entity:
        return Entity(
            entity_id=r["entity_id"], type=r["type"],
            institution_id=r["institution_id"], first_seen=r["first_seen"],
            last_seen=r["last_seen"], attributes=_loads(r["attributes"]),
            reputation=_loads(r["reputation"]),
        )

    def _row_to_event(self, r: sqlite3.Row) -> Event:
        ents = [row["entity_id"] for row in self._conn.execute(
            "SELECT entity_id FROM event_entities WHERE event_id=?", (r["event_id"],)
        ).fetchall()]
        return Event(
            event_id=r["event_id"], event_type=r["event_type"], ts=r["ts"],
            institution_id=r["institution_id"], entities=ents,
            payload=_loads(r["payload"]), derived=_loads(r["derived"]),
        )


# ── Convenience factory + typed id helpers ────────────────────────────────────

def open_store(db_path: os.PathLike | str = DEFAULT_DB_PATH) -> Store:
    return Store(db_path)


def eid(kind: str, raw: str) -> str:
    """Canonical typed entity id, e.g. eid('recipient', 'recv_9421')."""
    return f"{kind}:{raw}"


if __name__ == "__main__":
    # Smoke test against a throwaway db.
    import tempfile
    p = Path(tempfile.mkdtemp()) / "smoke.db"
    s = open_store(p)
    s.upsert_entity(eid("user", "u1"), "user", institution_id="inst_a")
    s.upsert_entity(eid("recipient", "r1"), "recipient", institution_id="inst_a",
                    reputation={"tx": 3, "fraud": 0, "fraud_rate": 0.006})
    ev = s.append_event("transaction",
                        entities=[eid("user", "u1"), eid("recipient", "r1")],
                        payload={"amount": 1820.0, "rail": "zelle"},
                        derived={"ml_score": 0.82}, institution_id="inst_a")
    s.update_reputation(eid("recipient", "r1"), {"fraud": 1, "fraud_rate": 0.31})
    print("event:", ev)
    print("recipient:", s.get_entity(eid("recipient", "r1")).reputation)
    print("events for r1:", len(s.events_for_entity(eid("recipient", "r1"))))
    print("stats:", json.dumps(s.stats(), indent=2))
    s.close()
    print("OK")
