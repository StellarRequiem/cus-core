"""
Grader backends.

A Grader takes a Task, a stage, and the agent's output, and returns a StageResult.
We define an abstract protocol so forest, buddy, or any future service can plug
in their own model backend. Two reference implementations are provided:

    - OllamaGrader:   calls a local Ollama model (e.g. qwen2.5:3b)
    - AnthropicGrader: calls the Anthropic API (e.g. claude-haiku-4-5)

Both produce the same StageResult shape, so the caller doesn't care which is used.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from cus_core.models import GradeResult, Stage, StageResult, Task


GRADER_PROMPT_TEMPLATE = """You are a grader. Evaluate the agent's output against the task contract.

TASK
====
ID: {task_id}
Description: {task_description}
Expected outcome: {task_expected_outcome}
Expected to fail: {task_expected_failure}
Failure criteria (if expected to fail): {task_failure_criteria}

STAGE
=====
Name: {stage_name}

AGENT OUTPUT
============
{agent_output}

RUBRICS
=======
Answer each question with a score from 0 to 100 and a one-sentence rationale.
{rubric_block}

RESPONSE FORMAT
===============
Return ONLY a JSON object of the shape:
{{
  "scores": {{ "rubric_name_1": 85, "rubric_name_2": 70, ... }},
  "notes": "One or two sentences of overall rationale."
}}

Do not include any text outside the JSON object.
"""


def _build_prompt(task: Task, stage: Stage, agent_output: str) -> str:
    rubric_lines = []
    for r in stage.rubrics:
        rubric_lines.append(f"- {r.name} (weight {r.weight}, {r.scoring}): {r.question}")
    return GRADER_PROMPT_TEMPLATE.format(
        task_id=task.id,
        task_description=task.description,
        task_expected_outcome=task.expected_outcome,
        task_expected_failure=task.expected_failure,
        task_failure_criteria=task.failure_criteria or "N/A",
        stage_name=stage.name.value,
        agent_output=agent_output,
        rubric_block="\n".join(rubric_lines),
    )


def _parse_grader_response(raw: str, stage: Stage) -> tuple[dict[str, float], str]:
    """Extract scores and notes from the grader's JSON response.

    Robust to models that wrap JSON in markdown fences or add chatter.
    Raises ValueError if no valid JSON object can be extracted.
    """
    # Strip common markdown fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find the first JSON object in the string
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in grader response: {raw[:200]}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in grader response: {e}\nRaw: {raw[:200]}") from e

    scores_raw = data.get("scores", {})
    notes = str(data.get("notes", ""))

    # Coerce and validate
    scores: dict[str, float] = {}
    rubric_names = {r.name for r in stage.rubrics}
    for name in rubric_names:
        if name not in scores_raw:
            raise ValueError(f"Grader response missing score for rubric {name!r}")
        try:
            val = float(scores_raw[name])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Score for {name!r} is not numeric: {scores_raw[name]!r}") from e
        scores[name] = max(0.0, min(100.0, val))
    return scores, notes


def _weighted_score(scores: dict[str, float], stage: Stage) -> float:
    return sum(scores[r.name] * r.weight for r in stage.rubrics)


class GraderBackend(ABC):
    """Minimal interface for a grading model backend."""

    @abstractmethod
    def complete(self, prompt: str, *, model: str, temperature: float) -> str:
        """Run the grader model and return its raw text response."""


class OllamaGrader(GraderBackend):
    """Local Ollama backend.

    Model identifier: bare model name, e.g. 'qwen2.5:3b'.
    (The 'ollama:' prefix in Stage.grader_model is stripped by the Grader
    orchestrator before being passed here.)
    """

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host

    def complete(self, prompt: str, *, model: str, temperature: float) -> str:
        try:
            import ollama  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "OllamaGrader requires the 'ollama' extra. "
                "Install with: uv add 'cus-core[ollama]'"
            ) from e
        from ollama import Client

        client = Client(host=self.host)
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response["response"]


class AnthropicGrader(GraderBackend):
    """Anthropic API backend.

    Model identifier: e.g. 'claude-haiku-4-5', 'claude-sonnet-4-6'.
    API key is read from ANTHROPIC_API_KEY env var by the SDK.
    """

    def __init__(self, max_tokens: int = 1024) -> None:
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, model: str, temperature: float) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicGrader requires the 'anthropic' extra. "
                "Install with: uv add 'cus-core[anthropic]'"
            ) from e
        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Take the first text block
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise RuntimeError("Anthropic response contained no text blocks")


class MockGrader(GraderBackend):
    """Deterministic mock for tests. Returns whatever you hand it."""

    def __init__(self, canned_response: str) -> None:
        self.canned_response = canned_response
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, model: str, temperature: float) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        return self.canned_response


class Grader:
    """Orchestrates grading of an agent output against a task's stages.

    The Grader is the main entry point for users. It resolves stage.grader_model
    strings like 'ollama:qwen2.5:3b' or 'anthropic:claude-haiku-4-5' to the
    appropriate backend.

    Usage:
        grader = Grader(backends={'ollama': OllamaGrader(), 'anthropic': AnthropicGrader()})
        result = grader.grade(task, stage_outputs={'execute': 'the agent said...'})
    """

    def __init__(
        self,
        backends: dict[str, GraderBackend] | None = None,
        pass_threshold: float = 70.0,
    ) -> None:
        self.backends = backends or {}
        self.pass_threshold = pass_threshold

    def _resolve_backend(self, grader_model: str) -> tuple[GraderBackend, str]:
        """Parse 'provider:model_name' into (backend, model_name)."""
        if ":" not in grader_model:
            raise ValueError(
                f"grader_model must be 'provider:model_name', got {grader_model!r}"
            )
        provider, _, model_name = grader_model.partition(":")
        if provider not in self.backends:
            raise KeyError(
                f"No backend registered for provider {provider!r}. "
                f"Available: {list(self.backends)}"
            )
        return self.backends[provider], model_name

    def grade_stage(self, task: Task, stage: Stage, agent_output: str) -> StageResult:
        backend, model_name = self._resolve_backend(stage.grader_model)
        prompt = _build_prompt(task, stage, agent_output)
        raw = backend.complete(prompt, model=model_name, temperature=stage.temperature)
        scores, notes = _parse_grader_response(raw, stage)
        return StageResult(
            stage=stage.name,
            scores=scores,
            weighted_score=_weighted_score(scores, stage),
            notes=notes,
            grader_model=stage.grader_model,
        )

    def grade(self, task: Task, stage_outputs: dict[str, str]) -> GradeResult:
        """Grade an agent attempt across all task stages.

        `stage_outputs` maps stage name (e.g. 'execute') to the agent's output
        at that stage. Stages without an output are skipped with a note.
        """
        stage_results: list[StageResult] = []
        for stage in task.stages:
            output = stage_outputs.get(stage.name.value)
            if output is None:
                continue
            stage_results.append(self.grade_stage(task, stage, output))

        if not stage_results:
            raise ValueError("No stage outputs provided; nothing to grade")

        composite = sum(r.weighted_score for r in stage_results) / len(stage_results)
        threshold = task.pass_threshold if task.pass_threshold is not None else self.pass_threshold
        passed = composite >= threshold

        # Expected-failure semantics: if the task declared expected_failure,
        # a HIGH score means the agent correctly failed in the expected way.
        # The `passed` flag already reflects this, because the grader is
        # scoring against `failure_criteria` when expected_failure=True.
        return GradeResult(
            task_id=task.id,
            stage_results=stage_results,
            composite_score=composite,
            passed=passed,
            expected_failure=task.expected_failure,
        )
