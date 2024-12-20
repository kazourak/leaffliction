#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = leaffliction
PYTHON_INTERPRETER = python3
VENV_DIR = .venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: setup
setup:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) setup.py develop

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf ./data/external/images
	rm -rf .venv

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 leaffliction  --disable-noqa
	isort --check --diff --profile black leaffliction
	black --check --config pyproject.toml leaffliction

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml leaffliction
	isort leaffliction --settings-path pyproject.toml

## Download data from 42 intranet
.PHONY: data
data:
	wget "https://cdn.intra.42.fr/document/document/17060/leaves.zip"
	[ -d ./data/external ] || mkdir -p ./data/external
	unzip leaves.zip -d ./data/external/
	rm -rf leaves.zip

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> New virtualenv created. Activate with:source $(VENV_DIR)/bin/activate"

## Build the html documentation
.PHONY: build_docs
build_docs:
	pydoctor \
    --project-name=${PROJECT_NAME} \
    --project-version=0.1 \
    --project-url=https://github.com/kazourak/${PROJECT_NAME}/ \
    --docformat=numpy \
    ./leaffliction

## Serve the html documentation
.PHONY: serve_docs
serve_docs:
	python3 -m http.server --directory ./apidocs 8000

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
