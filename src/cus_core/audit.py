"""
SHA-256 hash-chained audit log.

Every event is appended with a hash that covers (previous_hash + event_content).
Tampering with any event invalidates every subsequent hash, making the log
tamper-evident (not tamper-proof — an attacker can still rewrite the whole
file, but they can't silently edit a single entry).

This is a standalone reimplementation of forest's audit chain pattern so that
cus-core has no dependency on forest. Both projects can use the same chain
format.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    """A single immutable event in the chain."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str = Field(..., description="e.g. 'TASK_QUEUED', 'GRADING_COMPLETED'")
    actor: str = Field(..., description="Who performed the action — service or agent name")
    payload: dict[str, Any] = Field(default_factory=dict)

    def canonical_string(self) -> str:
        """Deterministic serialization for hashing.

        Uses sorted keys and ISO-format timestamp. If this function's output
        ever changes, all existing hashes become unverifiable — so this is
        effectively frozen once shipped.
        """
        return json.dumps(
            {
                "timestamp": self.timestamp.isoformat(),
                "event_type": self.event_type,
                "actor": self.actor,
                "payload": self.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class ChainEntry(BaseModel):
    """One line in the chain file: event + its hash + the previous hash."""

    event: AuditEvent
    previous_hash: str
    hash: str


class AuditChain:
    """Append-only hash-chained event log backed by a JSONL file.

    Usage:
        chain = AuditChain(Path("~/BuddyVault/audit.jsonl").expanduser())
        chain.append(AuditEvent(event_type="TASK_QUEUED", actor="buddy", payload={...}))
        ok, bad_line = chain.verify()
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        """Read the last line's hash, or the genesis hash if file is empty."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.GENESIS_HASH
        with self.path.open("rb") as f:
            # Efficient last-line read for small-to-medium files.
            # For very large files, switch to a seek-based approach.
            last_line = f.readlines()[-1].decode()
        entry = ChainEntry.model_validate_json(last_line)
        return entry.hash

    def _compute_hash(self, previous_hash: str, event: AuditEvent) -> str:
        payload = previous_hash + event.canonical_string()
        return hashlib.sha256(payload.encode()).hexdigest()

    def append(self, event: AuditEvent) -> ChainEntry:
        """Add an event to the chain. Returns the resulting entry."""
        prev = self._last_hash()
        entry = ChainEntry(
            event=event,
            previous_hash=prev,
            hash=self._compute_hash(prev, event),
        )
        with self.path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")
        return entry

    def verify(self) -> tuple[bool, int | None]:
        """Walk the chain and verify every hash.

        Returns (True, None) if the chain is intact.
        Returns (False, line_number) if an entry fails verification.
        """
        if not self.path.exists():
            return True, None
        prev = self.GENESIS_HASH
        with self.path.open() as f:
            for line_num, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = ChainEntry.model_validate_json(raw)
                except Exception:
                    return False, line_num
                if entry.previous_hash != prev:
                    return False, line_num
                expected = self._compute_hash(prev, entry.event)
                if expected != entry.hash:
                    return False, line_num
                prev = entry.hash
        return True, None

    def __iter__(self):
        """Iterate over entries in order."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                yield ChainEntry.model_validate_json(raw)
