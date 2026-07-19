"""
core/connectors.py - source connectors: pull events from a source into the transport.

The push path (POST /stream/publish) covers a source that sends events to us. The other half
of ingestion is PULL: a source we read from on our own schedule, a batch file drop, an export,
a polled export API. The hard parts of a pull connector are resumability and idempotency:
after a restart it must resume where it left off, not re-read from the start and not lose the
tail, and re-reading an overlap must not double-process.

This gives that as a small framework:
  * a durable CHECKPOINT per connector (store.get_checkpoint / set_checkpoint) records the last
    consumed offset, so poll() resumes from there,
  * every event is schema-VALIDATED (core/ingest_schema) before it enters the pipeline; invalid
    events are counted, not silently passed,
  * valid events are PUBLISHED to the durable transport keyed by transaction_id, so the
    transport's dedupe makes an overlapping re-read idempotent,
  * BACKPRESSURE stops the poll without advancing the checkpoint past the un-published event, so
    it is retried next poll rather than dropped.

FileConnector (a JSONL drop file, offset = line number) is the first concrete source. New
sources subclass SourceConnector and implement read(). Pure stdlib, unit-testable.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3

from .ingest_schema import validate_event
from .stream import BackpressureError


def _safe_ident(name: str) -> bool:
    """A SQL identifier we will interpolate must be a plain name (no injection). Values are
    always parameterised; only table/column names are interpolated, so they are allowlisted."""
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(name or "")))


class SourceConnector:
    """Base pull connector. Subclasses implement read(since_offset) -> yields (offset, raw)."""

    source_type = "base"

    def __init__(self, connector_id: str, transport, checkpoints, topic: str = "ingest"):
        self.connector_id = connector_id
        self.transport = transport          # a DurableQueue (publish)
        self.checkpoints = checkpoints      # anything with get_checkpoint / set_checkpoint
        self.topic = topic

    def read(self, since_offset: int):
        """Yield (offset, raw_event_or_None) for records after `since_offset`. offset is
        monotonic; yield None as the record to skip a line while still advancing the offset."""
        raise NotImplementedError

    @staticmethod
    def key_for(event: dict) -> str:
        return str(event.get("transaction_id") or "")

    def poll(self, max_events: int = 1000) -> dict:
        """Consume from the checkpoint forward: validate, publish valid events to the transport,
        advance the checkpoint. Idempotent (transport dedupe) and resumable (durable offset)."""
        start = self.checkpoints.get_checkpoint(self.connector_id)
        last_offset = start
        published = deduped = rejected = skipped = 0
        consumed = 0

        for offset, raw in self.read(start):
            if consumed >= max_events:
                break
            consumed += 1
            if raw is None:                                   # blank / skippable line
                last_offset = offset
                skipped += 1
                continue
            v = validate_event(raw, source=self.connector_id)
            if not v["valid"]:
                last_offset = offset
                rejected += 1
                continue
            try:
                seq = self.transport.publish(self.topic, self.key_for(v["event"]), v["event"])
            except BackpressureError:
                break                                         # stop; do NOT checkpoint past this one
            deduped += (seq is None)
            published += (seq is not None)
            last_offset = offset

        if last_offset != start:
            self.checkpoints.set_checkpoint(self.connector_id, last_offset)

        return {
            "connector": self.connector_id, "source_type": self.source_type,
            "from_offset": start, "to_offset": last_offset,
            "consumed": consumed, "published": published, "deduped": deduped,
            "rejected": rejected, "skipped": skipped,
        }


class FileConnector(SourceConnector):
    """A JSONL drop file: one event per line, offset = 1-based line number. The classic batch
    file-drop ingestion pattern (a processor writes a file, we tail it), made resumable."""

    source_type = "file"

    def __init__(self, connector_id: str, transport, checkpoints, path,
                 topic: str = "ingest"):
        super().__init__(connector_id, transport, checkpoints, topic)
        self.path = str(path)

    def read(self, since_offset: int):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            for idx, line in enumerate(f, start=1):
                if idx <= since_offset:
                    continue                                  # already consumed; resume past it
                s = line.strip()
                if not s:
                    yield idx, None                           # blank line: skip, advance offset
                    continue
                try:
                    yield idx, json.loads(s)
                except json.JSONDecodeError:
                    yield idx, {}                             # malformed: schema-rejected (visible), advance offset


class DBConnector(SourceConnector):
    """Incrementally poll a source SQL table (a core-banking / processor transactions table) by a
    monotonic integer watermark (an id / sequence / rowid). This is the canonical realistic fraud
    source, and it exercises two connector concerns the file source does not: watermark-based
    incremental querying (WHERE id > checkpoint), and FIELD MAPPING to translate the source
    schema into our canonical ingestion schema before validation.

    field_map maps source column -> canonical field, e.g. {"txn_amt": "amount",
    "cust_id": "user_id", "txn_ref": "transaction_id"}. Unmapped columns pass through untouched.
    """

    source_type = "db_table"

    def __init__(self, connector_id: str, transport, checkpoints, db_path, table: str,
                 id_column: str = "rowid", field_map: dict | None = None,
                 topic: str = "ingest", batch: int = 500):
        super().__init__(connector_id, transport, checkpoints, topic)
        self.db_path = str(db_path)
        self.table = table
        self.id_column = id_column
        self.field_map = dict(field_map or {})
        self.batch = batch

    def _map(self, row: dict) -> dict:
        out = dict(row)
        for src, dst in self.field_map.items():
            if src in row:
                out[dst] = row[src]
        return out

    def read(self, since_offset: int):
        # values are parameterised; the table/column identifiers are interpolated, so guard them
        if not (_safe_ident(self.table) and _safe_ident(self.id_column)):
            return
        if not os.path.exists(self.db_path):
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            q = (f"SELECT {self.id_column} AS __wm, * FROM {self.table} "
                 f"WHERE {self.id_column} > ? ORDER BY {self.id_column} LIMIT ?")
            for r in conn.execute(q, (int(since_offset), self.batch)):
                raw = {k: r[k] for k in r.keys() if k != "__wm"}
                yield int(r["__wm"]), self._map(raw)
        except sqlite3.Error:
            return                                            # a broken source must not crash ingestion
        finally:
            conn.close()
