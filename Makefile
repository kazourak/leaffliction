#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = leaffliction
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
VENV_DIR = .venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 leaffliction
	isort --check --diff --profile black leaffliction
	black --check --config pyproject.toml leaffliction

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml leaffliction




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> New virtualenv created. Activate with:source $(VENV_DIR)/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
