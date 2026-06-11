# cus-core

**cus-core — Contract Under Scrutiny: a grading framework for agent outputs.**

cus-core scores whether an agent adhered to the operator's declared intent,
not just whether it produced something that looks right. Tasks are declared
up front with rubrics, stages, and (optionally) failure criteria. An agent
that fails in the correct, pre-declared way earns credit for it; an agent
that succeeds for the wrong reasons does not.

This is a small, standalone Python package. It has no dependency on any
particular agent framework. It is used by
[`forest`](https://github.com/StellarRequiem/forest) for worker grading
and by other projects that want the same evaluation discipline.

---

## Install

Not yet on PyPI — install from GitHub:

```bash
pip install "git+https://github.com/StellarRequiem/cus-core"

# with extras — local Ollama grading / Anthropic grading / dev tools:
pip install "cus-core[ollama] @ git+https://github.com/StellarRequiem/cus-core"
pip install "cus-core[anthropic] @ git+https://github.com/StellarRequiem/cus-core"

# or with uv:
uv add "git+https://github.com/StellarRequiem/cus-core"
```

Requires Python 3.11+. (PyPI release planned.)

---

## Quickstart

```python
from cus import Grader, load_task
from cus.grader import OllamaGrader, AnthropicGrader

task = load_task("examples/summarize_for_student.yaml")

grader = Grader(
    backends={
        "ollama": OllamaGrader(),
        "anthropic": AnthropicGrader(),
    },
)

stage_outputs = {
    "plan":    "I'll extract the main argument, pick 3 supporting points, rewrite at grade-10 level.",
    "execute": "...the agent's 200-word summary...",
    "assess":  "This summary captures the thesis but skips the second counter-argument.",
}

result = grader.grade(task, stage_outputs)
print(result.summary())
# summarize_for_student: 82.4 → PASS
```

---

## Concepts

### Task

The contract between operator and agent. Declared up front in YAML or Python.

```yaml
id: summarize_for_student
description: "Summarize a file at high-school reading level"
expected_outcome: "A ~200-word plain-language summary"
expected_failure: false
pass_threshold: 75.0        # optional; falls back to Grader default if omitted
stages:
  - name: execute
    grader_model: "ollama:qwen2.5:3b"
    rubrics:
      - {name: accuracy,    weight: 0.5, question: "Faithful to the source?"}
      - {name: readability, weight: 0.5, question: "Would a student follow it?"}
```

### Stage

One step of the grading pipeline. Canonical stages: `queue`, `instruct`,
`plan`, `execute`, `assess`. A task can use any subset. Each stage has its
own grader model — you can grade `plan` with a cheap local model and
`assess` with a stronger one.

### Rubric

A weighted question the grader answers on a 0–100 scale. Weights within a
stage must sum to 1.0 (validated at construction).

### Expected failure

A task can declare `expected_failure: true` with a `failure_criteria`
string. The grader then evaluates whether the agent failed in the declared
way. **The criteria must be declared before the attempt** — agents cannot
claim credit for intentional failure retroactively.

### Pass threshold

Composite score at or above which the task counts as passed. Set per-task
(`pass_threshold` in YAML) or fall back to the `Grader(pass_threshold=...)`
default (70.0 out of the box).

---

## Backends

| Backend              | Model string example            | Requires              |
| -------------------- | ------------------------------- | --------------------- |
| `OllamaGrader`       | `"ollama:qwen2.5:3b"`           | local Ollama running  |
| `AnthropicGrader`    | `"anthropic:claude-haiku-4-5"`  | `ANTHROPIC_API_KEY`   |
| `MockGrader`         | `"mock:anything"`               | nothing (for tests)   |

Bring your own backend by subclassing `GraderBackend` and implementing
`complete(prompt, *, model, temperature) -> str`.

---

## Audit chain

Every non-trivial operation should be logged to an append-only SHA-256
hash-chained event log. cus-core ships this as a standalone utility:

```python
from pathlib import Path
from cus import AuditChain, AuditEvent

chain = AuditChain(Path("~/BuddyVault/audit.jsonl").expanduser())

chain.append(AuditEvent(
    event_type="TASK_GRADED",
    actor="buddy",
    payload={"task_id": "summarize_for_student", "score": 82.4, "passed": True},
))

ok, bad_line = chain.verify()
assert ok, f"Chain corrupted at line {bad_line}"
```

Tampering with any entry invalidates every subsequent hash.

---

## How `forest` could adopt this (not yet wired)

`forest` does **not** import `cus-core` today — its `core/grading_engine.py`
implements its own flat 4-factor rubric (compliance / usefulness / efficiency /
novelty). If adopted, the migration plan would be:

1. Add `cus-core` as a dependency of forest.
2. Define each worker's grading as a cus Task (one stage, four rubrics).
3. Replace `grading_engine.grade()` calls with `cus.Grader.grade()`.
4. Keep forest's audit chain format — cus-core's `AuditChain` is a
   drop-in reimplementation of the same SHA-256 pattern.

forest will continue to run unchanged during the migration. Tests in both
repos pin the grading output so the swap is observable.

---

## Development

```bash
git clone https://github.com/StellarRequiem/cus-core
cd cus-core
uv sync --extra dev
uv run pytest                      # all tests
uv run ruff check .                # lint
uv run mypy src                    # type-check
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE)
