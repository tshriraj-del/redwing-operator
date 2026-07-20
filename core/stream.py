"""
core/stream.py - a durable streaming transport between intake and scoring.

The ingestion surface scores synchronously on the request thread: a slow model blocks intake,
and a burst has nowhere to wait but the socket. A real pipeline puts a durable queue between
"accept the event" and "score the event", so intake stays fast, bursts are absorbed, and
nothing is lost if the scorer is behind or restarts.

This is that queue, in stdlib, backed by SQLite so it is genuinely durable (survives a restart)
rather than an in-memory buffer that evaporates. It is a single-node concept of a transport,
not distributed Kafka, but it has the semantics that matter:

  * a DURABLE LOG with monotonic offsets (seq), FIFO per topic,
  * IDEMPOTENT publish (dedupe by key), so a retry from the client cannot double-enqueue,
  * BACKPRESSURE: a bounded ready-depth that raises instead of growing without bound or
    silently dropping (the producer gets a clear signal to slow down),
  * AT-LEAST-ONCE consume with capped retries, then a DEAD-LETTER state,
  * REPLAY: reset dead (or done) messages back to ready to reprocess.

Pure stdlib, unit-testable without the ML stack.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class BackpressureError(Exception):
    """Raised when the ready-depth is at capacity: the producer must slow down or shed."""


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stream_log (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic        TEXT NOT NULL,
    key          TEXT NOT NULL DEFAULT '',
    payload      TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'ready',   -- ready | done | dead
    attempts     INTEGER NOT NULL DEFAULT 0,
    enqueued_ts  TEXT NOT NULL,
    updated_ts   TEXT NOT NULL,
    last_error   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_stream_topic_status ON stream_log(topic, status, seq);
CREATE INDEX IF NOT EXISTS idx_stream_key ON stream_log(topic, key);
"""


class DurableQueue:
    """A durable, offset-ordered, at-least-once queue with backpressure, dead-letter, replay."""

    def __init__(self, db_path: os.PathLike | str, max_depth: int = 10000,
                 max_attempts: int = 5):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.max_depth = max_depth
        self.max_attempts = max_attempts
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _depth(self, topic: str) -> int:
        r = self._conn.execute(
            "SELECT COUNT(*) n FROM stream_log WHERE topic=? AND status='ready'", (topic,)
        ).fetchone()
        return int(r["n"])

    def publish(self, topic: str, key: str, payload: dict):
        """Append one event. Idempotent by (topic, key): a duplicate key is ignored and returns
        None. Raises BackpressureError when the ready-depth is at capacity. Returns the seq."""
        key = str(key or "")
        with self._lock:
            if key:
                dup = self._conn.execute(
                    "SELECT seq FROM stream_log WHERE topic=? AND key=? "
                    "AND status IN ('ready','done') LIMIT 1", (topic, key)).fetchone()
                if dup:
                    return None                              # already enqueued or processed
            if self._depth(topic) >= self.max_depth:
                raise BackpressureError(
                    f"topic '{topic}' at capacity ({self.max_depth} ready); slow down or shed")
            ts = _now()
            cur = self._conn.execute(
                "INSERT INTO stream_log (topic, key, payload, status, attempts, enqueued_ts, "
                "updated_ts) VALUES (?,?,?,'ready',0,?,?)",
                (topic, key, json.dumps(payload or {}), ts, ts))
            self._conn.commit()
            return cur.lastrowid

    def consume_batch(self, topic: str, handler, batch: int = 50) -> dict:
        """Process up to `batch` ready messages oldest-first. `handler(payload)` runs each; on
        success the message is 'done', on exception its attempts increment and it either stays
        'ready' (retry) or, at max_attempts, becomes 'dead'. At-least-once by construction.

        SINGLE-CONSUMER PER TOPIC. Messages are selected as 'ready' and only marked terminal
        after the handler returns; there is no in-flight lease, so two consumers running the
        same topic concurrently would both pick up the same rows and double-process them. The
        operator runs exactly one consumer loop per topic, which is what makes this safe. A
        multi-consumer deployment needs a lease/visibility-timeout first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, payload, attempts FROM stream_log WHERE topic=? AND status='ready' "
                "ORDER BY seq LIMIT ?", (topic, batch)).fetchall()
        processed = succeeded = failed = dead = 0
        for r in rows:
            processed += 1
            try:
                handler(json.loads(r["payload"]))
                with self._lock:
                    self._conn.execute(
                        "UPDATE stream_log SET status='done', updated_ts=? WHERE seq=?",
                        (_now(), r["seq"]))
                    self._conn.commit()
                succeeded += 1
            except Exception as e:                           # noqa: BLE001 - transport must not die
                attempts = int(r["attempts"]) + 1
                new_status = "dead" if attempts >= self.max_attempts else "ready"
                with self._lock:
                    self._conn.execute(
                        "UPDATE stream_log SET status=?, attempts=?, updated_ts=?, last_error=? "
                        "WHERE seq=?", (new_status, attempts, _now(), str(e)[:500], r["seq"]))
                    self._conn.commit()
                failed += 1
                if new_status == "dead":
                    dead += 1
        return {"processed": processed, "succeeded": succeeded, "failed": failed, "dead": dead}

    def stats(self, topic: str | None = None) -> dict:
        where, args = ("WHERE topic=?", (topic,)) if topic else ("", ())
        rows = self._conn.execute(
            f"SELECT status, COUNT(*) n FROM stream_log {where} GROUP BY status", args).fetchall()
        by = {r["status"]: int(r["n"]) for r in rows}
        return {
            "topic": topic or "*",
            "ready": by.get("ready", 0),
            "done": by.get("done", 0),
            "dead": by.get("dead", 0),
            "depth": by.get("ready", 0),
            "max_depth": self.max_depth,
        }

    def dead_letters(self, topic: str | None = None, limit: int = 100) -> list:
        where, args = ("WHERE topic=? AND status='dead'", [topic]) if topic else ("WHERE status='dead'", [])
        args.append(limit)
        rows = self._conn.execute(
            f"SELECT seq, topic, key, payload, attempts, last_error, updated_ts "
            f"FROM stream_log {where} ORDER BY seq LIMIT ?", args).fetchall()
        return [{"seq": r["seq"], "topic": r["topic"], "key": r["key"],
                 "payload": json.loads(r["payload"]), "attempts": r["attempts"],
                 "last_error": r["last_error"], "updated_ts": r["updated_ts"]} for r in rows]

    def replay(self, topic: str | None = None, which: str = "dead") -> int:
        """Reset dead (default) or done messages back to ready for reprocessing. Returns count."""
        which = which if which in ("dead", "done") else "dead"
        cond = "status=?"
        args = [which]
        if topic:
            cond += " AND topic=?"
            args.append(topic)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE stream_log SET status='ready', attempts=0, updated_ts=? WHERE {cond}",
                [_now()] + args)
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
