.PHONY: install test lint typecheck run-scenarios grade-local diagram history demo-resume clean

install:
	pip install -e '.[dev,google,sqlite]'

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

run-scenarios:
	python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json

grade-local:
	python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

# ── extensions ────────────────────────────────────────────────────────
diagram:
	python -m langgraph_agent_lab.cli diagram --output docs/graph.mmd

history:
	python -m langgraph_agent_lab.cli history --scenario-id S05_error

demo-resume:
	python -m langgraph_agent_lab.cli demo-resume --scenario-id S04_risky --approve

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json outputs/*.sqlite*
