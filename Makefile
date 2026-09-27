.PHONY: install run debug clean lint lint-strict bonus

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
