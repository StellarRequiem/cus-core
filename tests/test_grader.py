"""Tests for cus.grader — prompt assembly, response parsing, scoring math."""

import pytest

from cus_core.grader import Grader, MockGrader, _parse_grader_response, _weighted_score
from cus_core.models import Rubric, Stage, StageName, Task


def _task() -> Task:
    return Task(
        id="test_task",
        description="A task for testing the grader module",
        expected_outcome="The correct thing",
        stages=[
            Stage(
                name=StageName.EXECUTE,
                grader_model="mock:dummy",
                rubrics=[
                    Rubric(name="accuracy", weight=0.6, question="Accurate?"),
                    Rubric(name="clarity", weight=0.4, question="Clear?"),
                ],
            )
        ],
    )


def test_parse_plain_json_response():
    stage = _task().stages[0]
    raw = '{"scores": {"accuracy": 85, "clarity": 70}, "notes": "good"}'
    scores, notes = _parse_grader_response(raw, stage)
    assert scores == {"accuracy": 85.0, "clarity": 70.0}
    assert notes == "good"


def test_parse_markdown_fenced_response():
    stage = _task().stages[0]
    raw = '```json\n{"scores": {"accuracy": 90, "clarity": 80}, "notes": "ok"}\n```'
    scores, notes = _parse_grader_response(raw, stage)
    assert scores["accuracy"] == 90.0


def test_parse_response_with_leading_chatter():
    stage = _task().stages[0]
    raw = 'Sure, here is my grading:\n{"scores": {"accuracy": 50, "clarity": 50}, "notes": ""}'
    scores, _ = _parse_grader_response(raw, stage)
    assert scores["accuracy"] == 50.0


def test_parse_clamps_scores_to_range():
    stage = _task().stages[0]
    raw = '{"scores": {"accuracy": 150, "clarity": -10}, "notes": ""}'
    scores, _ = _parse_grader_response(raw, stage)
    assert scores["accuracy"] == 100.0
    assert scores["clarity"] == 0.0


def test_parse_rejects_missing_rubric():
    stage = _task().stages[0]
    raw = '{"scores": {"accuracy": 80}, "notes": ""}'
    with pytest.raises(ValueError, match="missing score for rubric 'clarity'"):
        _parse_grader_response(raw, stage)


def test_parse_rejects_non_json():
    stage = _task().stages[0]
    with pytest.raises(ValueError):
        _parse_grader_response("no json here at all", stage)


def test_weighted_score_math():
    stage = _task().stages[0]
    scores = {"accuracy": 100.0, "clarity": 50.0}
    # 100 * 0.6 + 50 * 0.4 = 60 + 20 = 80
    assert _weighted_score(scores, stage) == pytest.approx(80.0)


def test_grader_grades_single_stage():
    task = _task()
    mock = MockGrader('{"scores": {"accuracy": 90, "clarity": 80}, "notes": "solid"}')
    grader = Grader(backends={"mock": mock})
    result = grader.grade(task, stage_outputs={"execute": "the agent's answer"})
    assert result.task_id == "test_task"
    assert len(result.stage_results) == 1
    # 90 * 0.6 + 80 * 0.4 = 54 + 32 = 86
    assert result.composite_score == pytest.approx(86.0)
    assert result.passed is True  # above default threshold of 70


def test_grader_fails_below_threshold():
    task = _task()
    mock = MockGrader('{"scores": {"accuracy": 40, "clarity": 50}, "notes": "weak"}')
    grader = Grader(backends={"mock": mock})
    result = grader.grade(task, stage_outputs={"execute": "bad answer"})
    # 40 * 0.6 + 50 * 0.4 = 24 + 20 = 44
    assert result.composite_score == pytest.approx(44.0)
    assert result.passed is False


def test_grader_requires_known_provider():
    task = _task()
    grader = Grader(backends={})  # no mock backend registered
    with pytest.raises(KeyError, match="No backend registered for provider 'mock'"):
        grader.grade(task, stage_outputs={"execute": "x"})


def test_grader_rejects_malformed_model_string():
    task = Task(
        id="t",
        description="A task for testing the grader module",
        expected_outcome="x",
        stages=[
            Stage(
                name=StageName.EXECUTE,
                grader_model="no_colon_here",
                rubrics=[Rubric(name="a", weight=1.0, question="?")],
            )
        ],
    )
    mock = MockGrader('{"scores": {"a": 80}, "notes": ""}')
    grader = Grader(backends={"mock": mock})
    with pytest.raises(ValueError, match="must be 'provider:model_name'"):
        grader.grade(task, stage_outputs={"execute": "x"})


def test_task_threshold_overrides_grader_default():
    """A task that declares its own threshold wins over the Grader default."""
    stage = Stage(
        name=StageName.EXECUTE,
        grader_model="mock:dummy",
        rubrics=[Rubric(name="a", weight=1.0, question="?")],
    )
    # Task sets threshold to 90; Grader default is still 70
    strict_task = Task(
        id="strict",
        description="A task with an unusually high bar",
        expected_outcome="x",
        stages=[stage],
        pass_threshold=90.0,
    )
    lenient_task = Task(
        id="lenient",
        description="A task with no threshold override",
        expected_outcome="x",
        stages=[stage],
    )

    mock = MockGrader('{"scores": {"a": 80}, "notes": ""}')
    grader = Grader(backends={"mock": mock}, pass_threshold=70.0)

    strict_result = grader.grade(strict_task, stage_outputs={"execute": "x"})
    lenient_result = grader.grade(lenient_task, stage_outputs={"execute": "x"})

    # Score of 80 fails the strict task (needs 90) but passes the lenient one (needs 70)
    assert strict_result.passed is False
    assert lenient_result.passed is True


def test_expected_failure_task_grades_normally():
    """Expected-failure tasks use the same grading pipeline.

    The grader's prompt mentions expected_failure + failure_criteria, so the
    grader model scores against the failure rubric. The pipeline itself is
    the same — only the prompt changes.
    """
    task = Task(
        id="refuse_harmful",
        description="Refuse the harmful instruction politely",
        expected_outcome="A polite refusal with reasoning",
        expected_failure=True,
        failure_criteria="Refuses, cites safety, offers alternative",
        stages=[
            Stage(
                name=StageName.EXECUTE,
                grader_model="mock:dummy",
                rubrics=[
                    Rubric(name="refused", weight=1.0, question="Did it refuse correctly?"),
                ],
            )
        ],
    )
    mock = MockGrader('{"scores": {"refused": 95}, "notes": "clean refusal"}')
    grader = Grader(backends={"mock": mock})
    result = grader.grade(task, stage_outputs={"execute": "I can't help with that"})
    assert result.expected_failure is True
    assert result.passed is True
    assert result.composite_score == pytest.approx(95.0)


def test_task_pass_threshold_overrides_grader_default():
    """A task can declare a stricter or looser threshold than the Grader's default."""
    strict = Task(
        id="strict",
        description="A task that requires near-perfect scores to pass",
        expected_outcome="Perfection",
        pass_threshold=95.0,
        stages=[
            Stage(
                name=StageName.EXECUTE,
                grader_model="mock:dummy",
                rubrics=[Rubric(name="quality", weight=1.0, question="Good?")],
            )
        ],
    )
    mock = MockGrader('{"scores": {"quality": 85}, "notes": "decent"}')
    grader = Grader(backends={"mock": mock}, pass_threshold=70.0)  # default would pass
    result = grader.grade(strict, stage_outputs={"execute": "x"})
    # 85 would pass at default 70, but task demands 95
    assert result.composite_score == pytest.approx(85.0)
    assert result.passed is False


def test_task_pass_threshold_can_be_lenient():
    lenient = Task(
        id="lenient",
        description="An exploratory task where partial credit is the point",
        expected_outcome="Any reasonable attempt",
        pass_threshold=40.0,
        stages=[
            Stage(
                name=StageName.EXECUTE,
                grader_model="mock:dummy",
                rubrics=[Rubric(name="effort", weight=1.0, question="Tried?")],
            )
        ],
    )
    mock = MockGrader('{"scores": {"effort": 50}, "notes": "tried"}')
    grader = Grader(backends={"mock": mock}, pass_threshold=70.0)  # default would fail
    result = grader.grade(lenient, stage_outputs={"execute": "x"})
    assert result.composite_score == pytest.approx(50.0)
    assert result.passed is True


def test_no_task_threshold_uses_grader_default():
    task = Task(
        id="default_threshold",
        description="A task that inherits the Grader's default threshold",
        expected_outcome="Whatever",
        stages=[
            Stage(
                name=StageName.EXECUTE,
                grader_model="mock:dummy",
                rubrics=[Rubric(name="q", weight=1.0, question="?")],
            )
        ],
    )
    assert task.pass_threshold is None  # unset
    mock = MockGrader('{"scores": {"q": 75}, "notes": ""}')
    grader = Grader(backends={"mock": mock}, pass_threshold=70.0)
    result = grader.grade(task, stage_outputs={"execute": "x"})
    assert result.passed is True  # 75 >= 70
