.PHONY: install run check-output cpu-venv runner test-runner debug clean fclean lint lint-strict test test-fast

install:
	sh goinfre.sh uv sync

run:
	sh goinfre.sh uv run python -m src

# Lighter install: separate venv with CPU-only torch (no CUDA libraries).
CPU_VENV = .venv-cpu
TORCH_CPU_INDEX = https://download.pytorch.org/whl/cpu

cpu-venv:
	sh goinfre.sh uv venv --allow-existing $(CPU_VENV)
	sh goinfre.sh uv pip install --python $(CPU_VENV) \
		--index-url $(TORCH_CPU_INDEX) "torch>=2.0.0"
	sh goinfre.sh uv pip install --python $(CPU_VENV) ./llm_sdk numpy pydantic

runner: cpu-venv
	sh goinfre.sh $(CPU_VENV)/bin/python -m src

# Fast tests (no real model) inside the CPU-only venv.
test-runner: cpu-venv
	sh goinfre.sh uv pip install --python $(CPU_VENV) pytest flake8 mypy
	SKIP_SLOW=1 sh goinfre.sh $(CPU_VENV)/bin/python -m pytest tests -q

debug:
	sh goinfre.sh uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/output/*.json 2>/dev/null || true
	rm -f .coverage .coverage.* 2>/dev/null || true

# Everything clean removes, plus both venvs, data/output and goinfre caches.
fclean: clean
	sh clean.sh

lint:
	sh goinfre.sh uv run flake8 .
	sh goinfre.sh uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	sh goinfre.sh uv run flake8 .
	sh goinfre.sh uv run mypy . --strict

test:
	sh goinfre.sh uv run --with pytest --with pytest-cov python -m pytest tests -v \
		--cov=src --cov-report=term-missing

test-fast:
	SKIP_SLOW=1 sh goinfre.sh uv run --with pytest --with pytest-cov python -m pytest tests \
		--cov=src --cov-report=term-missing

check-output:
	sh goinfre.sh uv run python tests/check_output.py
