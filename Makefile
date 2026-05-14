.PHONY: setup smoke train eval test lint clean

setup:
	uv venv --python 3.11
	. .venv/bin/activate && uv pip install -e ".[dev]"
	git submodule update --init --recursive
	pre-commit install

smoke:
	python scripts/train.py --smoke

train:
	python scripts/train.py

eval:
	python scripts/eval.py

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -rf logs/* checkpoints/* results/raw/* results/scratch/*
