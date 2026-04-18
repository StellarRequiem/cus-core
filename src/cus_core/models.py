"""
Core data models for CUS.

These define the contract. Every other module in cus operates on these types.
Pydantic gives us validation, serialization, and clear error messages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class StageName(str, Enum):
    """Canonical stages in the CUS grading pipeline.

    The pipeline follows the operator's articulated flow:
        queue → instruct → plan → execute → assess
    """

    QUEUE = "queue"
    INSTRUCT = "instruct"
    PLAN = "plan"
    EXECUTE = "execute"
    ASSESS = "assess"


class Rubric(BaseModel):
    """A weighted scoring criterion applied at a stage.

    Example:
        Rubric(name="compliance", weight=0.45, question="Did the agent stay within bounds?")

    Weights across a stage's rubrics must sum to 1.0 (validated by Stage).
    """

    name: str = Field(..., description="Short identifier, e.g. 'compliance', 'usefulness'")
    weight: float = Field(..., ge=0.0, le=1.0, description="Contribution to stage score")
    question: str = Field(..., description="The yes/no or 0-100 question the grader answers")
    scoring: str = Field(
        default="numeric",
        description="'numeric' (0-100), 'binary' (0 or 100), or 'likert' (0, 25, 50, 75, 100)",
    )

    @field_validator("scoring")
    @classmethod
    def _validate_scoring(cls, v: str) -> str:
        allowed = {"numeric", "binary", "likert"}
        if v not in allowed:
            raise ValueError(f"scoring must be one of {allowed}, got {v!r}")
        return v


class Stage(BaseModel):
    """One stage of the grading pipeline.

    Each stage has its own rubrics and its own grader model. This lets you use
    a cheap local model for early stages and escalate to a stronger model for
    the final assessment.
    """

    name: StageName
    rubrics: list[Rubric] = Field(..., min_length=1)
    grader_model: str = Field(
        ...,
        description="Model identifier, e.g. 'ollama:qwen2.5:3b' or 'anthropic:claude-haiku-4-5'",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> Stage:
        total = sum(r.weight for r in self.rubrics)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Stage {self.name.value!r} rubric weights must sum to 1.0, got {total:.4f}"
            )
        return self


class Task(BaseModel):
    """A declared contract for what an agent is supposed to do.

    The key CUS innovation is `expected_failure`: if set to True, the grader
    rewards the agent for failing in the expected way. This is declared up
    front — an agent cannot claim credit for intentional failure retroactively.

    Example:
        Task(
            id="summarize_for_student",
            description="Summarize this file so a high-school student could follow it",
            expected_outcome="A 200-word summary using plain language, no jargon",
            expected_failure=False,
            stages=[...],
        )
    """

    id: str = Field(..., pattern=r"^[a-z0-9_]+$", description="Lowercase snake_case identifier")
    description: str = Field(..., min_length=10, description="Human-readable task description")
    expected_outcome: str = Field(..., description="What a successful output looks like")
    expected_failure: bool = Field(
        default=False,
        description="If True, the agent is expected to fail. Grader rewards correct failures.",
    )
    failure_criteria: str | None = Field(
        default=None,
        description="Required if expected_failure=True. Describes what the 'correct' failure looks like.",
    )
    stages: list[Stage] = Field(..., min_length=1)
    pass_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Composite score at or above this value counts as a pass. "
            "If None, the Grader's default threshold is used."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _failure_criteria_when_expected(self) -> Task:
        if self.expected_failure and not self.failure_criteria:
            raise ValueError(
                "failure_criteria must be provided when expected_failure=True. "
                "An agent cannot claim credit for intentional failure retroactively."
            )
        return self


class StageResult(BaseModel):
    """The grader's output for a single stage."""

    stage: StageName
    scores: dict[str, float] = Field(..., description="Per-rubric scores, 0-100")
    weighted_score: float = Field(..., ge=0.0, le=100.0)
    notes: str = Field(default="", description="Grader's rationale")
    grader_model: str
    graded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _scores_in_range(self) -> StageResult:
        for name, score in self.scores.items():
            if not 0.0 <= score <= 100.0:
                raise ValueError(f"Rubric {name!r} score {score} outside [0, 100]")
        return self


class GradeResult(BaseModel):
    """Final output of grading a task attempt.

    `passed` is True if the composite score meets the task's passing threshold.
    For expected-failure tasks, `passed` reflects whether the agent failed correctly.
    """

    task_id: str
    stage_results: list[StageResult]
    composite_score: float = Field(..., ge=0.0, le=100.0)
    passed: bool
    expected_failure: bool = False
    notes: str = ""
    graded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        """Human-readable one-line summary."""
        verdict = "PASS" if self.passed else "FAIL"
        ef_tag = " (expected-failure)" if self.expected_failure else ""
        return f"{self.task_id}: {self.composite_score:.1f} → {verdict}{ef_tag}"
