"""
REDWING core: the platform substrate.

Phase 1 turns REDWING from a set of pages that each re-read a CSV into one
organism with a shared nervous system. This package is that nervous system:
a single entity model and an append-only event log, on one durable store.

Everything the platform does reduces to three verbs against this store:
  read entities + recent events  ->  emit new events  ->  update entity reputation

The transaction is just one event_type, not the center of the universe. That
reframe is what makes REDWING graph-native and network-ready (WS3).
"""

from .store import Entity, Event, Store, open_store, DEFAULT_DB_PATH

__all__ = ["Entity", "Event", "Store", "open_store", "DEFAULT_DB_PATH"]
