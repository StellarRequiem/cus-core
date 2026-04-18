"""
cus-core — Caste Unity System

A grading framework for agent outputs that scores adherence to declared intent,
including rewarding correctly-predicted failures.

Public API:
    Task              — declared contract between operator and agent
    Stage             — one step in the grading pipeline (queue/instruct/plan/execute/assess)
    Rubric            — scoring criteria with weights
    GradeResult       — output of grading, with per-dimension breakdown
    Grader            — orchestrates the grading pipeline for a given task and output
    AuditChain        — append-only SHA-256 hash-chained event log
    load_task         — parse a YAML task definition into a Task object
"""

from cus_core.audit import AuditChain, AuditEvent
from cus_core.grader import Grader
from cus_core.models import GradeResult, Rubric, Stage, StageResult, Task
from cus_core.task_loader import load_task

__version__ = "0.1.0"

__all__ = [
    "AuditChain",
    "AuditEvent",
    "GradeResult",
    "Grader",
    "Rubric",
    "Stage",
    "StageResult",
    "Task",
    "load_task",
]
