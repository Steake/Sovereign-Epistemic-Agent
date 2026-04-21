# AGENTS.md

## Project Overview

This repository contains the **Sovereign Epistemic Agent** project. The current implemented subsystem is the **Epistemic Tribunal**, a Python research codebase for adjudicating between competing candidate solutions on ARC-like grid tasks.

The core pipeline is:

1. Load a task.
2. Generate multiple `CandidateTrace` objects with different generators.
3. Extract structural invariants from training pairs.
4. Score candidates with the trace critic.
5. Compute uncertainty signals across the candidate pool.
6. Aggregate those signals into a `SELECT`, `RESAMPLE`, or `ABSTAIN` decision.
7. Persist task, trace, decision, and failure data to a SQLite ledger.

The code is organized as a single Python package under `src/epistemic_tribunal`, with experiment configs in `configs/`, sample and generated data in `data/`, operational scripts in `scripts/`, and tests in `tests/`.

## Repository Map

- `src/epistemic_tribunal/cli.py`: Typer CLI entrypoint exposed as `tribunal`
- `src/epistemic_tribunal/orchestrator.py`: end-to-end pipeline wiring
- `src/epistemic_tribunal/tribunal_types.py`: shared Pydantic models used across the pipeline
- `src/epistemic_tribunal/config.py`: YAML + env-driven settings loader
- `src/epistemic_tribunal/generators/`: generator implementations and registry
- `src/epistemic_tribunal/invariants/`: invariant extraction
- `src/epistemic_tribunal/critics/`: candidate critique logic
- `src/epistemic_tribunal/uncertainty/`: entropy / coalition / disagreement analysis
- `src/epistemic_tribunal/tribunal/`: final aggregation and scoring
- `src/epistemic_tribunal/ledger/`: SQLite persistence layer
- `src/epistemic_tribunal/evaluation/`: benchmark, calibration, and summary metrics
- `src/epistemic_tribunal/tasks/`: ARC-like task loading and task helpers
- `configs/default.yaml`: baseline runtime configuration
- `configs/validation/`: validation-arm configs used by sweep scripts
- `data/examples/`: small local benchmark/sample tasks
- `scripts/`: operational utilities, GPU/pod setup, validation runners, debug helpers
- `docs/`: experiment reports and roadmap

## Setup

Use Python 3.10+; the project metadata targets Python 3.11. On this machine, prefer `python3` rather than `python`.

### Editable install with `uv`

```bash
uv pip install -e ".[dev]"
```

### Editable install with venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[llm]"
```

## Common Commands

After editable install, the main CLI is `tribunal`.

### Run one task

```bash
tribunal run data/examples/colour_swap_001.json
tribunal run data/examples/colour_swap_001.json --json
tribunal run data/examples/colour_swap_001.json --config configs/default.yaml
tribunal run data/examples/colour_swap_001.json --ledger data/my_ledger.db
```

### Run a benchmark over a directory

```bash
tribunal benchmark data/examples/
tribunal benchmark data/examples/ --json
tribunal benchmark data/examples/ --resume
tribunal benchmark data/examples/ --ledger data/benchmark_ledger.db
```

### Calibration and ledger inspection

```bash
tribunal calibrate --ledger data/tribunal_ledger.db
tribunal ledger stats --ledger data/tribunal_ledger.db
tribunal ledger inspect --task-id colour_swap_001 --ledger data/tribunal_ledger.db
```

### Test commands

```bash
pytest
pytest -v
pytest tests/test_orchestrator.py
pytest --cov=epistemic_tribunal --cov-report=term-missing
```

### Linting

```bash
ruff check .
```

## Configuration

Runtime settings are YAML-backed Pydantic models defined in `src/epistemic_tribunal/config.py`.

Important config behavior:

- Default config path is `configs/default.yaml`.
- `TRIBUNAL_CONFIG_PATH` can override the config file.
- `TRIBUNAL_LEDGER_PATH` can override the SQLite ledger path.
- `LOG_LEVEL` can override logging verbosity.
- Several LLM configs interpolate env vars such as `DEEPSEEK_API_KEY`.

Example:

```bash
export TRIBUNAL_CONFIG_PATH=configs/default.yaml
export TRIBUNAL_LEDGER_PATH=data/tribunal_ledger.db
export LOG_LEVEL=DEBUG
```

For dataset-driven validation scripts, `ARC_DATASET_PATH` is the main dataset override.

## Development Guidance

### Where to make changes

- Add or modify generator implementations in `src/epistemic_tribunal/generators/`, and register new generator names in `build_generators()` in `src/epistemic_tribunal/generators/base.py`.
- Change shared pipeline data structures in `src/epistemic_tribunal/tribunal_types.py`.
- Change scoring, abstention, or decision behavior in `src/epistemic_tribunal/tribunal/` and `src/epistemic_tribunal/critics/`.
- Change benchmark/resume behavior in `src/epistemic_tribunal/evaluation/benchmark.py`.
- Change persistence behavior in `src/epistemic_tribunal/ledger/`.

### Invariants to preserve

- Keep generator names stable if they are referenced from YAML configs, experiments, or saved metrics.
- Preserve the shape and semantics of `ExperimentRun`, `TribunalDecision`, `CandidateTrace`, and failure-ledger records unless you are deliberately migrating downstream consumers.
- Keep CLI commands stable where practical; they are documented in `README.md` and used by scripts.
- If you change config model fields, update both YAML configs and any logic that depends on those fields.
- If you change benchmark checkpointing, keep `run_progress.json` and checkpoint ledger behavior coherent.

### Testing expectations

When touching core pipeline code, run the most relevant focused tests first, then the full suite if the change is broad.

Examples:

```bash
pytest tests/test_generators.py
pytest tests/test_trace_critic.py
pytest tests/test_uncertainty.py
pytest tests/test_orchestrator.py tests/test_integration.py
```

Add or update tests for:

- new generators
- new config branches
- score aggregation changes
- ledger schema/persistence behavior
- benchmark resume/checkpoint behavior

## Code Style

The repository is conventional Python with:

- `src/` layout
- `from __future__ import annotations`
- Pydantic models for shared types
- typed function signatures
- line length 100 via Ruff

Follow existing patterns:

- Prefer small, explicit functions over dense abstractions.
- Keep docstrings on public modules/classes/functions when the surrounding code already uses them.
- Use plain data models and dictionaries for experiment metadata rather than introducing heavy framework layers.
- Prefer adding focused helpers in the relevant subsystem instead of cross-cutting utility sprawl.

## Data, Artifacts, and Scripts

This repository contains many experimental artifacts:

- SQLite ledgers under `data/`
- validation DBs and result JSON
- benchmark logs in the repo root
- pod/server scripts under `scripts/`

Do not casually rewrite or delete generated artifacts unless the task explicitly calls for it.

Many scripts in `scripts/` are operational and environment-specific:

- some assume GPU/pod environments
- some assume external ARC datasets
- some contain hardcoded absolute paths or service endpoints

Treat those scripts as operational tooling, not as general-purpose local dev defaults.

## Security and Secrets

- Never commit fresh secrets into configs, `.env`, or scripts.
- Prefer environment variables for API keys and tokens.
- If you need model or dataset credentials, use env vars such as `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, or `HF_TOKEN` rather than inlining them.

## Recommended Agent Workflow

When making code changes in this repository:

1. Read the relevant module plus its corresponding tests.
2. Check for YAML config usage if the change affects runtime behavior.
3. Make the smallest coherent change in the owning subsystem.
4. Run focused tests first.
5. Run broader tests if the change crosses subsystem boundaries.
6. Update `README.md` or `docs/` only when behavior or operator workflows materially changed.
