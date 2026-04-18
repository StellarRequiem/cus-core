"""
YAML task loader.

Tasks can be defined in YAML for readability and sharing. Example:

    id: summarize_for_student
    description: "Summarize the given file at a high-school reading level"
    expected_outcome: "A 200-word plain-language summary, no jargon"
    expected_failure: false
    stages:
      - name: execute
        grader_model: "ollama:qwen2.5:3b"
        temperature: 0.0
        rubrics:
          - name: accuracy
            weight: 0.4
            question: "Does the summary faithfully represent the source?"
          - name: readability
            weight: 0.4
            question: "Would a high-school student understand this?"
          - name: brevity
            weight: 0.2
            question: "Is it roughly 200 words?"

The loader is deliberately thin. Pydantic does the heavy lifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cus_core.models import Task


def load_task(path: str | Path) -> Task:
    """Load a Task from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Task file not found: {p}")
    with p.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Task file must be a YAML mapping, got {type(data).__name__}")
    return Task.model_validate(data)


def load_task_from_string(yaml_text: str) -> Task:
    """Load a Task from a YAML string (useful for tests)."""
    data: Any = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError(f"Task YAML must be a mapping, got {type(data).__name__}")
    return Task.model_validate(data)
