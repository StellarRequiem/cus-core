"""Tests for cus.models — validation and invariants."""

import pytest
from pydantic import ValidationError

from cus.models import GradeResult, Rubric, Stage, StageName, StageResult, Task


def _valid_stage() -> Stage:
    return Stage(
        name=StageName.EXECUTE,
        grader_model="ollama:qwen2.5:3b",
        rubrics=[
            Rubric(name="accuracy", weight=0.6, question="Is it accurate?"),
            Rubric(name="clarity", weight=0.4, question="Is it clear?"),
        ],
    )


def test_rubric_weights_must_sum_to_one():
    with pytest.raises(ValidationError) as exc:
        Stage(
            name=StageName.EXECUTE,
            grader_model="ollama:qwen2.5:3b",
            rubrics=[
                Rubric(name="a", weight=0.5, question="?"),
                Rubric(name="b", weight=0.3, question="?"),
            ],
        )
    assert "weights must sum to 1.0" in str(exc.value)


def test_rubric_weights_at_exactly_one_pass():
    stage = _valid_stage()
    assert sum(r.weight for r in stage.rubrics) == pytest.approx(1.0)


def test_rubric_weights_with_float_imprecision_pass():
    # 0.1 + 0.2 + 0.7 in float is 0.9999999999999999
    stage = Stage(
        name=StageName.EXECUTE,
        grader_model="ollama:qwen2.5:3b",
        rubrics=[
            Rubric(name="a", weight=0.1, question="?"),
            Rubric(name="b", weight=0.2, question="?"),
            Rubric(name="c", weight=0.7, question="?"),
        ],
    )
    assert len(stage.rubrics) == 3


def test_task_id_must_be_snake_case():
    with pytest.raises(ValidationError):
        Task(
            id="Invalid-Id",
            description="A task description that is long enough",
            expected_outcome="The expected thing",
            stages=[_valid_stage()],
        )


def test_task_description_minimum_length():
    with pytest.raises(ValidationError):
        Task(
            id="ok_id",
            description="short",
            expected_outcome="something",
            stages=[_valid_stage()],
        )


def test_expected_failure_requires_failure_criteria():
    with pytest.raises(ValidationError) as exc:
        Task(
            id="deliberate_fail",
            description="A task where the agent should refuse",
            expected_outcome="A polite refusal with reasoning",
            expected_failure=True,
            stages=[_valid_stage()],
        )
    assert "failure_criteria must be provided" in str(exc.value)


def test_expected_failure_with_criteria_is_valid():
    task = Task(
        id="deliberate_fail",
        description="A task where the agent should refuse the instruction",
        expected_outcome="A polite refusal with reasoning",
        expected_failure=True,
        failure_criteria="Refuses, cites reason, offers alternative",
        stages=[_valid_stage()],
    )
    assert task.expected_failure is True
    assert task.failure_criteria is not None


def test_rubric_scoring_must_be_valid_enum():
    with pytest.raises(ValidationError):
        Rubric(name="x", weight=1.0, question="?", scoring="invalid")


def test_stage_result_scores_in_range():
    sr = StageResult(
        stage=StageName.EXECUTE,
        scores={"accuracy": 85.0, "clarity": 70.0},
        weighted_score=79.0,
        notes="ok",
        grader_model="ollama:qwen2.5:3b",
    )
    assert sr.weighted_score == 79.0


def test_stage_result_rejects_out_of_range_scores():
    with pytest.raises(ValidationError):
        StageResult(
            stage=StageName.EXECUTE,
            scores={"accuracy": 150.0},
            weighted_score=80.0,
            grader_model="ollama:qwen2.5:3b",
        )


def test_grade_result_summary_pass():
    gr = GradeResult(
        task_id="x",
        stage_results=[],
        composite_score=80.0,
        passed=True,
    )
    assert "PASS" in gr.summary()


def test_grade_result_summary_expected_failure():
    gr = GradeResult(
        task_id="x",
        stage_results=[],
        composite_score=80.0,
        passed=True,
        expected_failure=True,
    )
    assert "expected-failure" in gr.summary()
