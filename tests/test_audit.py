"""Tests for cus.audit — chain integrity and tamper detection."""

import json
from pathlib import Path

import pytest

from cus.audit import AuditChain, AuditEvent


def test_empty_chain_verifies(tmp_path: Path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    ok, bad = chain.verify()
    assert ok is True
    assert bad is None


def test_single_event_verifies(tmp_path: Path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    entry = chain.append(AuditEvent(event_type="TEST", actor="pytest", payload={"x": 1}))
    assert entry.previous_hash == AuditChain.GENESIS_HASH
    assert len(entry.hash) == 64
    ok, bad = chain.verify()
    assert ok is True


def test_many_events_verify(tmp_path: Path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    for i in range(20):
        chain.append(AuditEvent(event_type="TEST", actor="pytest", payload={"i": i}))
    ok, bad = chain.verify()
    assert ok is True
    entries = list(chain)
    assert len(entries) == 20
    # Each entry's previous_hash == prior entry's hash
    for i in range(1, 20):
        assert entries[i].previous_hash == entries[i - 1].hash


def test_tampered_payload_is_detected(tmp_path: Path):
    chain_path = tmp_path / "audit.jsonl"
    chain = AuditChain(chain_path)
    chain.append(AuditEvent(event_type="A", actor="pytest", payload={"v": 1}))
    chain.append(AuditEvent(event_type="B", actor="pytest", payload={"v": 2}))
    chain.append(AuditEvent(event_type="C", actor="pytest", payload={"v": 3}))

    # Tamper with line 2: change the payload without fixing the hash
    lines = chain_path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["event"]["payload"]["v"] = 99
    lines[1] = json.dumps(entry)
    chain_path.write_text("\n".join(lines) + "\n")

    ok, bad_line = chain.verify()
    assert ok is False
    assert bad_line == 2


def test_tampered_previous_hash_is_detected(tmp_path: Path):
    chain_path = tmp_path / "audit.jsonl"
    chain = AuditChain(chain_path)
    chain.append(AuditEvent(event_type="A", actor="pytest"))
    chain.append(AuditEvent(event_type="B", actor="pytest"))

    # Corrupt the previous_hash of line 2
    lines = chain_path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["previous_hash"] = "f" * 64
    lines[1] = json.dumps(entry)
    chain_path.write_text("\n".join(lines) + "\n")

    ok, bad_line = chain.verify()
    assert ok is False
    assert bad_line == 2


def test_canonical_string_is_deterministic():
    # Same event content → same canonical string regardless of dict ordering
    e1 = AuditEvent(event_type="X", actor="a", payload={"b": 2, "a": 1})
    e2 = AuditEvent(
        event_type="X",
        actor="a",
        payload={"a": 1, "b": 2},
        timestamp=e1.timestamp,
    )
    assert e1.canonical_string() == e2.canonical_string()
