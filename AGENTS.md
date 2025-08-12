# Repository Guidelines

## Project Structure & Modules
- Root: `vision_action_agent/` (library) and `pdfcollect/` (example app).
- Source: `vision_action_agent/src/` with `core/`, `vision/`, `actions/`, `config/`, `utils/`.
- Tests: `vision_action_agent/tests/` (unit tests), `pdfcollect/test_screenshot.py` (manual/system check).
- Assets: `vision_action_agent/templates/` (image templates used by detectors).
- Docs & Examples: `vision_action_agent/docs/`, `vision_action_agent/examples/`.

## Architecture Overview
- Orchestrator: `core/agent.py` wires config, vision, actions, workflows, and error handling.
- Vision: `vision/` provides screen capture, OCR, and element detection (icons, buttons, text fields).
- Actions & Workflows: `actions/` controls mouse/keyboard; workflows in YAML via `WorkflowExecutor`, with custom actions supported.

## Build, Test, Run
- Install deps: `pip install -r vision_action_agent/requirements.txt`
- Editable install (dev): `pip install -e vision_action_agent` (uses `setup.py`).
- Tests (unittest): `python -m unittest discover -s vision_action_agent/tests -v`
- Tests (pytest): `pytest` (uses repo `pytest.ini` defaults)
- Type check (optional): `mypy --config-file mypy.ini`
- Run example app: `python pdfcollect/main.py <DOI>`

## Coding Style & Naming
- Python 3.11; follow PEP 8 with 4-space indents.
- Names: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Typing: add type hints for public functions; keep stubs minimal in `vision/` modules if heavy deps complicate typing.
- Logging: use `loguru` (already wired in `core/agent.py`).

## Testing Guidelines
- Frameworks: `unittest` primary; `pytest` supported for discovery and running.
- Conventions: name tests `test_*.py`; group by component (e.g., `tests/test_vision_*.py`).
- Isolation: mock OS/UI interactions (`pyautogui`, OCR, screen capture). Avoid real screen I/O in unit tests.
- Commands: `python -m unittest -v` or `pytest` (verbose and log output preconfigured). Keep tests fast and deterministic; system checks may live under `pdfcollect/`.

## Commit & PR Guidelines
- Commits: history is minimal; use clear, imperative messages. Conventional Commits encouraged (e.g., `feat:`, `fix:`, `refactor:`).
- PRs: include summary, rationale, before/after behavior, and test notes. Link issues when applicable. Attach screenshots of workflows or logs if relevant.
- CI readiness: ensure tests pass locally and code installs from `setup.py`.

## Configuration Tips
- Workflows: edit `pdfcollect/config.yaml` to adapt steps; template images live in `vision_action_agent/templates/`.
- Platform paths: `pdfcollect/main.py` moves downloads from `/mnt/c/Users/bchoi/Downloads`; adjust for your OS.
- Heavy deps (OCR/TF): install system requirements for `pytesseract`/`easyocr` and TensorFlow if enabling those paths.
