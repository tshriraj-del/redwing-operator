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

from .ingest_schema import validate_event
from .stream import BackpressureError


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
