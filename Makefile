.PHONY: style quality test

include_dirs = mini_torch examples tests

# Check that source code meets quality standards
quality:
	git ls-files $(include_dirs) | grep -E "\.py$\" | xargs isort --diff --check-only
	git ls-files $(include_dirs) | grep -E "\.py$\" | xargs black --check --diff --color
	git ls-files $(include_dirs) | grep -E "\.py$\" | xargs flake8 --count --statistics

# Format source code automatically
style:
	git ls-files $(include_dirs) | grep -E "\.py$\" | xargs isort
	git ls-files $(include_dirs) | grep -E "\.py$\" | xargs black

# Run tests for the library
test:
	python3 -m pytest ./tests/
