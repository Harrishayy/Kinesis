.PHONY: setup smoke train eval play test lint clean train-fig8 eval-fig8 play-fig8

DYLD_LIBRARY_PATH ?= $(HOME)/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/lib

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

play:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) . .venv/bin/activate && \
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) mjpython scripts/play.py $(ARGS)

# Figure-8 (3D Lissajous) targets — engages all 7 arm joints.
train-fig8:
	python scripts/train.py --config figure8_3d

eval-fig8:
	python scripts/eval.py --config figure8_3d

play-fig8:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) . .venv/bin/activate && \
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) mjpython scripts/play.py --config figure8_3d $(ARGS)

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -rf logs/* checkpoints/* results/raw/* results/scratch/*
