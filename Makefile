# CORA developer tasks. Windows equivalent: .\tasks.ps1 <target>
.PHONY: setup lint format test data eval demo

setup:
	uv sync --locked --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

data:
	@echo "not implemented yet - tasks 1.1-1.4" && exit 1

# Full evaluation chain (Phase 6). Each data/LLM-dependent stage FAILS CLOSED with a clear message
# when its inputs are absent (never fabricates - REQ-47); the leading '-' lets a partial run on a
# bare host continue to make_report, which always renders the report with pending-run placeholders.
# Run from the MAIN repo root where the landing lives (see EVALUATION.md "Reproduction").
eval:
	-uv run python scripts/eval/build_scenarios.py
	-uv run python scripts/eval/rederive_thresholds.py
	-uv run python scripts/eval/run_eval.py
	-uv run python scripts/eval/compute_metrics.py
	-uv run python scripts/eval/fairness_report.py
	-uv run python scripts/eval/historical_baseline.py
	uv run python scripts/eval/make_report.py

demo:
	@echo "not implemented yet - task 5.4" && exit 1
