This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
uv run pytest tests/policies/test_policies.py -k act  # One file / filtered tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
make DEVICE=cpu test-act-ete-train                    # One E2E target (see Makefile)
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
make check-docstrings check-doctest-list doctest      # Docstring checks run by quality.yml
make fix-docstrings                                   # Auto-fix docstring stubs + sort doctest list
```

CI's `fast_tests.yml` runs the whole suite once per extras tier (base `test`, then `dataset`, `hardware`, `viz`), so a test needing an extra must skip cleanly when it's absent (`pytest.importorskip` or `skip_if_package_missing` from `tests/utils.py`).

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`rollout/`, `inference/`, `remote_inference/`, `transport/`** — Deployment path: `rollout/` drives the robot control loop (strategies: episodic, DAgger, sentry, …); `inference/` picks a backend (`sync`, local `rtc`, or `remote`); `remote_inference/` is the policy server/client over Zenoh. Design reference: `src/lerobot/remote_inference/README.md`.
- **`streaming/`** — Training-time dataset streaming primitives (episode cache/pool, MP4 sidecars, range fetch).
- **`rewards/`, `rl/`, `distributed/`** — Reward models (own factory/`pretrained.py`, mirroring policies), RL training, and multi-GPU parallelism/checkpointing.
- **`lerobot_types.py`** and **`configs/types.py`** — Core type aliases (`TransitionKey`, `EnvTransition`) and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit + docstring/doctest checks), `fast_tests.yml` (per-extras tiers, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `benchmark_tests.yml` (sim benchmarks in `docker/Dockerfile.benchmark.*`), `docker_publish.yml` (also runs `multigpu`-marked tests), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`), CI (`Dockerfile.internal`), Jetson, and per-benchmark sim images.
- **`utils/`** — Docstring/doctest checkers used by `make check-docstrings`; `utils/documentation_tests.txt` is the allowlist of files whose docstring examples `make doctest` runs.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the paths in its `files` setting, except `lerobot.rl`, the vendored `molmoact2_hf_model` and the generated `*_pb2` modules. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **Docstrings**: the API reference is generated from `src/lerobot/` docstrings; public additions must follow the [docstring standard](https://huggingface.co/docs/lerobot/writing_docstrings), which CI parses and checks.
- **PRs**: per `AI_POLICY.md`, disclose significant AI assistance in the PR description and follow `.github/PULL_REQUEST_TEMPLATE.md`.
