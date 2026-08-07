.PHONY: install lint test

install:
	pip install -e .[dev]

lint:
	ruff check src/ tests/

test:
	pytest
