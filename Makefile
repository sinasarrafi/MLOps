# Define Python interpreter
PYTHON_INTERPRETER := python

# Define phony targets (not actual files)
.PHONY: lint
lint: fmt-check flake8  # Run both formatting and linting checks

.PHONY: fmt
fmt:  # Auto-format code
	@echo "Auto-formatting code with black and isort..."
	$(PYTHON_INTERPRETER) -m black src tests
	$(PYTHON_INTERPRETER) -m isort src tests

.PHONY: fmt-check
fmt-check:  # Check code formatting without modifying files
	@echo "Running black+isort fmt check..."
	$(PYTHON_INTERPRETER) -m black --check --diff src tests
	$(PYTHON_INTERPRETER) -m isort --check --diff src tests

.PHONY: flake8
flake8:  # Check code quality and style violations
	@echo "Running flake8 lint..."
	$(PYTHON_INTERPRETER) -m flake8 src tests

.PHONY: test
test:  # Run all tests with verbose output
	$(PYTHON_INTERPRETER) -m pytest -v --capture=no --disablewarnings tests