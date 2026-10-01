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

eval:
	@echo "not implemented yet - phase 6" && exit 1

demo:
	@echo "not implemented yet - task 5.4" && exit 1
