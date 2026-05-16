.PHONY: setup smoke train eval play test lint clean train-fig8 eval-fig8 play-fig8

# Derive the dynamic-loader path from the venv's actual Python at invocation time
# so the value tracks whatever uv-installed patch version is in use (3.11.x).
# Required only by `make play` on macOS, where mjpython needs libpython on DYLD_LIBRARY_PATH.
DYLD_LIBRARY_PATH ?= $(shell test -x .venv/bin/python && .venv/bin/python -c \
  "import sys, pathlib; print(pathlib.Path(sys.executable).resolve().parent.parent / 'lib')")

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev]"
	git submodule update --init --recursive
	uv run pre-commit install

smoke:
	uv run python scripts/train.py --smoke

# `train` stays on the circle baseline as a fast smoke (~7 min CPU) so a fresh
# clone can verify the training loop works end-to-end without a 25-min commitment.
train:
	uv run python scripts/train.py

# `eval` points at the headline residual experiment. The viviani_residual best
# checkpoint is committed (~2 MB), so this reproduces the RESULTS.md headline
# numbers (6.43 mm steady RMS) on a fresh clone in ~10 s with no training needed.
eval:
	uv run python scripts/eval.py --config viviani_residual

play:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) \
	  .venv/bin/python .venv/bin/mjpython scripts/play.py $(ARGS)

# Figure-8 (3D Lissajous) targets - engages all 7 arm joints.
train-fig8:
	uv run python scripts/train.py --config figure8_3d

eval-fig8:
	uv run python scripts/eval.py --config figure8_3d

play-fig8:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) \
	  .venv/bin/python .venv/bin/mjpython scripts/play.py --config figure8_3d $(ARGS)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

# Wipe training artifacts but preserve the committed headline checkpoints
# (checkpoints/viviani_residual/best/best_model.zip + ppo_panda_final.zip)
# so `uv run python scripts/eval.py --config viviani_residual` keeps working.
clean:
	rm -rf logs/* results/raw/* results/scratch/*
	find checkpoints -name 'ppo_panda_*_steps.zip' -delete 2>/dev/null || true
