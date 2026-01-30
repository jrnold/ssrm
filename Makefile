SRC   ?= ssrm_test
SHELL ?= bash
RUN   ?= uv run
DOCS  ?= docs

NOTEBOOKS ?= $(wildcard notebooks/*.ipynb)

## Meta #######################################################################

.PHONY: help

# note: keep this as first target
help:  ## displays available make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

## Environment ################################################################

.PHONY: install install-dev clean clean-pyc clean-notebooks

install:  ## installs dependencies for external users.
	uv sync --frozen

install-dev:  ## installs dev dependencies for local development.
	uv sync --frozen --all-extras
	$(RUN) pre-commit install

clean: clean-pyc  ## cleans all generated files.
	-@rm -rf dist build out
	-@find . -name '*.ipynb_checkpoints' -exec rm -rf {} +

clean-pyc:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-notebooks:
	$(foreach f,$(NOTEBOOKS),jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $(f);)

## Build/Release ##############################################################

.PHONY: docs fmt fmt-notebooks lint typecheck release

docs:  ## builds docs.
	$(MAKE) -C $(DOCS) clean html
	# Copy logo image to fix Sphinx not groking relative paths in the README.
	@mkdir -p $(DOCS)/build/html/logos && cp logos/*.png $(DOCS)/build/html/logos

fmt:  ## runs code auto-formatters and linter (ruff).
	$(RUN) ruff check --fix $(SRC)
	$(RUN) ruff format $(SRC)

fmt-notebooks:  ## runs notebook auto-formatters (black_nbconvert).
	$(RUN) black_nbconvert $(NOTEBOOKS)

lint:  ## runs code linter (ruff).
	$(RUN) ruff check $(SRC)
	$(RUN) ruff format --check $(SRC)

typecheck:  ## runs type checker (pyright).
	$(RUN) pyright $(SRC)

release: clean  ## builds release artifacts into dist directory.
	uv build

## Testing ####################################################################

.PHONY: check test

test: PYTEST_ARGS ?= --color=yes --cov-report term --cov=$(SRC)
test:  ## runs the unit tests.
	$(RUN) pytest $(PYTEST_ARGS)

check: lint typecheck test  ## runs all checks (lint, typecheck, test).
