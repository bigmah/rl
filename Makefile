.PHONY: setup build play train eval

# One-time: fetch the PufferLib submodule and Python deps
setup:
	git submodule update --init
	uv sync

build:
	./build.sh

# A/D move, W/Space jump, S drop through platforms, R restart, ESC quit
play: build
	./build/platformer

# Extra flags pass through, e.g. make train ARGS="--train.total-timesteps 10_000_000"
train: build
	uv run puffer train platformer --slowly $(ARGS)

# Watch the most recent checkpoint play (ESC to quit)
eval: build
	uv run puffer eval platformer --slowly --load-model-path latest --vec.total-agents 1 $(ARGS)
