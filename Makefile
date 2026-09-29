.PHONY: install dev lint format test test-data validate clean

install:
	pip install -e ".[bench]"

dev:
	pip install -e ".[dev,realtime,gemini]"

lint:
	ruff check apex_voice tests scripts
	ruff format --check apex_voice tests scripts

format:
	ruff check --fix apex_voice tests scripts
	ruff format apex_voice tests scripts

test:
	pytest -q

test-data:
	@test -n "$(APEX_VOICE_DATA)" || (echo "set APEX_VOICE_DATA to the dataset root" && exit 1)
	pytest -q -m dataset

validate:
	apex-voice validate

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
