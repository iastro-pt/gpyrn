.DEFAULT_GOAL := help

.PHONY: help
help:
		@echo "Please use 'make <target>' where <target> is one of:"
		@echo ""
		@echo "	format-check		Formatting tools:  only checks for errors"
		@echo "	format-fix			Formatting tools: fixes errors where possible"
		@echo "	lint-check			Linting tools: only checks for errors"
		@echo "	type-check			Static type checker: only checks for errors"
		@echo "	docstring-check		Docstring checker: checks missing doctrings"
		@echo "	static-check		Runs format-check, lint and typing"
		@echo "	static-fix			Runs format-fix, lint and typing"
		@echo "	unit-test			Runs the unit tests"
		@echo "	run-examples		runs the examples in the example folder."

.PHONY: format-check
format-check:
	black --check examples/ src/ tests/
	isort --check-only --profile black examples/ src/ tests/

.PHONY: format-fix
format-fix:
	black examples/ src/ tests/
	isort --profile black examples/ src/ tests/

.PHONY: lint-check
lint-check:
	flake8 examples/ src/ tests/

.PHONY: type-check
type-check:
	mypy examples/ src/ tests/

.PHONY: docstring-check
docstring-check:
	pydocstyle examples/ src/ tests/

.PHONY: static-check
static-check: format-check lint-check type-check

.PHONY: static-fix
static-fix: format-fix lint-check type-check

.PHONY: unit-test
unit-test:
	pytest --cov-report term-missing --cov=src/ -vv -W ignore::DeprecationWarning

.PHONY: run-examples
run-examples:
	examples/run_examples.sh
