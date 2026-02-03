.PHONY: lint format check

# Variables
PYTHON := python3
RUFF := ruff

# Default target
all: lint format

# Linting
lint:
	uv run $(RUFF) check . --fix

# Formatting
format:
	uv run $(RUFF) format .

# Check without fixing (CI style)
check:
	uv run $(RUFF) check .
	uv run $(RUFF) format --check .
