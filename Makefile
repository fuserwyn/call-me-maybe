.PHONY: install run debug clean lint lint-strict bonus test test-fast

install:
	uv sync --extra dev

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/output/*.json 2>/dev/null || true
	rm -f .coverage .coverage.* 2>/dev/null || true

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

bonus:
	uv run python -m unittest discover -s tests -p 'test_*.py' -v
	uv run python -m src.bonus --visualize

test:
	uv run --with pytest --with pytest-cov python -m pytest tests -v \
		--cov=src --cov-report=term-missing

test-fast:
	SKIP_SLOW=1 uv run --with pytest --with pytest-cov python -m pytest tests \
		--deselect tests/test_bonus_tokenizer.py --cov=src --cov-report=term-missing
