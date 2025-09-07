# Makefile to mirror GitHub Actions workflow locally
# Requires: uv (https://docs.astral.sh/uv/)

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

UV ?= uv
ARTIFACTS_DIR ?= artifacts

.PHONY: help install test lint format format-check type-check docs build ci artifacts artifact-docs artifact-build clean

help:
	@echo "Common targets:"
	@echo "  make install       # Sync project deps (all extras + dev)"
	@echo "  make test          # Run pytest"
	@echo "  make lint          # Run ruff lint (check)"
	@echo "  make format-check  # Check formatting with ruff"
	@echo "  make type-check    # Run pyright"
	@echo "  make docs          # Build MkDocs site into ./site"
	@echo "  make build         # Build the package into ./dist"
	@echo "  make artifacts     # Create local tar.gz artifacts for site/ and dist/"
	@echo "  make ci            # Run the full CI suite (like GitHub Actions)"
	@echo "  make clean         # Remove build/test artifacts"

install:
	$(UV) sync --all-extras --dev

# Mirrors: uv run pytest tests
# If tests/ doesn't exist, fall back to plain pytest
test:
	@if [ -d tests ]; then \
		$(UV) run pytest tests; \
	else \
		$(UV) run pytest; \
	fi

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format

format-check:
	$(UV) run ruff format --check

type-check:
	$(UV) run pyright

# Mirrors: uv run mkdocs build
docs:
	$(UV) run mkdocs build

# Mirrors: uv build
build:
	$(UV) build

# Run all steps in the same order as the GitHub workflow
ci: install test lint format-check type-check docs build

# Create local artifacts mirroring the GitHub upload-artifact steps
artifact-docs: docs
	@mkdir -p $(ARTIFACTS_DIR)
	@if [ -d site ]; then \
		tar -czf $(ARTIFACTS_DIR)/docs-site.tar.gz -C site .; \
		echo "Created $(ARTIFACTS_DIR)/docs-site.tar.gz"; \
	else \
		echo "No site/ directory found. Run 'make docs' first."; \
		exit 1; \
	fi

artifact-build: build
	@mkdir -p $(ARTIFACTS_DIR)
	@if [ -d dist ]; then \
		tar -czf $(ARTIFACTS_DIR)/build.tar.gz -C dist .; \
		echo "Created $(ARTIFACTS_DIR)/build.tar.gz"; \
	else \
		echo "No dist/ directory found. Run 'make build' first."; \
		exit 1; \
	fi

artifacts: artifact-docs artifact-build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache site dist build *.egg-info $(ARTIFACTS_DIR)

