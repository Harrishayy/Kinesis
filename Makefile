.PHONY: setup smoke train eval test lint clean

setup:
	uv venv --python 3.11
	. .venv/bin/activate && uv pip install -e ".[dev]"
	git submodule update --init --recursive
	pre-commit install

smoke:
	python scripts/quick_smoke.py

train:
	python -m kinesis.train --config src/kinesis/configs/default.yaml

eval:
	python -m kinesis.evaluate --checkpoint checkpoints/latest.zip

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -rf logs/* checkpoints/* results/raw/* results/scratch/*
