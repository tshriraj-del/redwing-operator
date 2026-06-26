"""
feedback.py - closes the loop.

Until now, analyst dispositions (confirm fraud / clear) went nowhere: the model never
learned from the people using it. This captures every disposition as a labeled example
and does two things with it:

  1. ONLINE  - immediately updates the recipient-reputation layer, so the very next
               payment to a counterparty an analyst just confirmed as fraud scores
               higher. No retrain wait. This is the part that closes the loop in real
               time.
  2. LOGGED  - appends the labeled example to feedback_log.jsonl, a retraining queue
               the next full model rebuild consumes. The labels validate; the online
               update acts.

A fraud system that cannot learn from its own analysts is a scorer, not a system.
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path

# Dispositions that mean "this was fraud" vs "this was legitimate".
FRAUD_LABELS = {"fraud", "confirm_fraud", "block", "block_instrument",
                "confirmed_fraud", "deny_dispute_first_party", "chargeback"}
LEGIT_LABELS = {"legit", "clear", "clear_false_positive", "approve", "allow"}


class FeedbackStore:
    def __init__(self, log_path, reputation=None):
        self.log_path = Path(log_path)
        self.reputation = reputation          # shared RecipientReputation instance
        self.counts = {"fraud": 0, "legit": 0, "other": 0}
        self.recent = deque(maxlen=50)
        self.online_updates = 0
        self._load_existing()

    def _load_existing(self):
        if not self.log_path.exists():
            return
        try:
            for line in self.log_path.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                self._tally(e.get("label_class", "other"))
        except Exception:
            pass

    def _tally(self, label_class):
        self.counts[label_class] = self.counts.get(label_class, 0) + 1

    def record(self, transaction_id, label, recipient_id="", source="investigator"):
        label = str(label or "").lower()
        if label in FRAUD_LABELS:
            lc, is_fraud = "fraud", True
        elif label in LEGIT_LABELS:
            lc, is_fraud = "legit", False
        else:
            lc, is_fraud = "other", None

        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "transaction_id": str(transaction_id),
            "recipient_id": str(recipient_id or ""),
            "disposition": label,
            "label_class": lc,
            "source": source,
        }

        # Online loop: update reputation from a confirmed disposition.
        online = None
        if is_fraud is not None and recipient_id and self.reputation is not None:
            try:
                online = self.reputation.update(recipient_id, is_fraud)
                self.online_updates += 1
                entry["online_update"] = {"recipient_global_fraud_rate":
                                          online.get("recipient_global_fraud_rate")} if online else None
            except Exception:
                online = None

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        self._tally(lc)
        self.recent.appendleft(entry)

        return {
            "recorded": True,
            "label_class": lc,
            "online_reputation_update": online,
            "queued_for_retrain": self.counts["fraud"] + self.counts["legit"],
        }

    def status(self):
        labeled = self.counts["fraud"] + self.counts["legit"]
        return {
            "labeled_total": labeled,
            "by_class": dict(self.counts),
            "online_reputation_updates": self.online_updates,
            "queued_for_retrain": labeled,
            "loop": "closed" if labeled > 0 else "no feedback yet",
            "recent": list(self.recent)[:15],
        }
