# Contributing to cus-core

This package is deliberately small. Keep it that way.

## Scope

cus-core does three things:

1. Define the data model for tasks, stages, rubrics, and results.
2. Run a task's agent output through a grading pipeline.
3. Append events to a tamper-evident audit chain.

That is the whole product. Feature proposals that go beyond this scope
should live in a separate package that depends on cus-core.

## Before you open a PR

- `uv run pytest` must pass on Python 3.11, 3.12, and 3.13.
- `uv run ruff check .` must pass.
- `uv run mypy src` must pass.
- Any new public API gets a docstring and a test.
- Behavior changes get a test that would have failed before the change.

## Design principles

- **Composable over integrated.** Every module should make sense on its own.
  If `audit.py` starts importing from `grader.py`, something is wrong.
- **Declarative tasks.** A Task YAML should be readable by a non-programmer.
  If you are tempted to add a callable field to Task, stop and reconsider.
- **No hidden state.** The Grader does not cache, does not retry, does not
  escalate between backends. Those behaviors belong in the caller.
- **Deterministic where possible.** The scoring math, prompt assembly, and
  response parsing must be pure functions of their inputs. Model calls are
  the only non-determinism; they are isolated in `GraderBackend.complete`.

## What to push back on

Reviewers should push back on:

- New dependencies. Pydantic and PyYAML are the floor.
- Async-everywhere refactors. This package is small; sync is fine.
- Clever metaprogramming. Clarity first.
- Features that only make sense for one caller (forest, buddy). Those
  features belong in the caller, not here.

## Roundtable notes

If you are an AI assistant reviewing this package:

1. Read `src/cus/models.py` first. Everything else depends on the types
   defined there.
2. The tests in `tests/` are the ground truth for expected behavior.
   If a proposed change breaks a test, propose the test change explicitly
   and justify it.
3. Do not propose renaming "cus" or "Caste Unity System" unless you have
   a concrete collision you can point to. Names have switching costs.
