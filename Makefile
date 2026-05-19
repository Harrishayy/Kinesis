.PHONY: setup smoke train eval play test lint clean

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

# `eval` points at the headline residual experiment with 6-DoF tracking
# (position + orientation). The viviani_residual_orient best checkpoint is
# committed (~2 MB), so this reproduces the RESULTS.md headline numbers
# (0.46 mm steady RMS, 19.0° orientation RMS) on a fresh clone in ~10 s with
# no training needed.
eval:
	uv run python scripts/eval.py --config viviani_residual_orient

play:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) \
	  .venv/bin/python .venv/bin/mjpython scripts/play.py $(ARGS)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

# Wipe training artifacts but preserve the committed headline checkpoints
# (checkpoints/viviani_residual_orient/best/best_model.zip — the 0.46 mm /
# 19° headline; also viviani_residual/best/best_model.zip — preserved as the
# earlier position-only baseline for historical reference).
clean:
	rm -rf logs/* results/raw/* results/scratch/*
	find checkpoints -name 'ppo_panda_*_steps.zip' -delete 2>/dev/null || true
