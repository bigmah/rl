.PHONY: setup build play train eval sm64 sm64-state sm64-watch sm64-bench sm64-probe sm64-explore sm64-robustify

# Which env is built into PufferLib's extension. One at a time, which is
# PufferLib's own shape: make ENV=sm64 train builds sm64 and trains it.
ENV ?= platformer

# One-time: fetch the PufferLib submodule and Python deps
setup:
	git submodule update --init
	uv sync

build:
	./build.sh $(ENV)

# A/D move, W/Space jump, S drop through platforms, R restart, ESC quit
play: build
	./build/platformer

# Extra flags pass through, e.g. make train ARGS="--train.total-timesteps 10_000_000"
train: build
	uv run python mlx_pufferl.py train $(ENV) $(ARGS)

# Watch the most recent checkpoint play (ESC to quit)
eval: build
	uv run python mlx_pufferl.py eval $(ENV) --load-model-path latest --vec.total-agents 1 $(ARGS)

# --- Super Mario 64 ----------------------------------------------------------
# The game itself comes from N64Bundler: drop your own sm64.z64 on it once so it
# is recompiled, and set N64BUNDLER if the checkout is not ../static_recomp/n64bundler.

sm64:
	./build.sh sm64

# Play through the title, the file select and Peach's letter once, and save the
# castle grounds. Every episode starts from this. Made automatically on the
# first run; this remakes it.
sm64-state: sm64
	./build/sm64_tool state

sm64-train: sm64 sm64-state
	uv run python mlx_pufferl.py train sm64 $(ARGS)

# Go-Explore's first phase with no policy: random play from the archive until it
# has opened the door, keeping the fastest run in build/sm64/door.demo
sm64-explore: sm64
	./build/sm64_tool explore $(or $(GAMES),8) $(or $(SECONDS),600)

# Go-Explore's second phase: train a policy to open the door from the real
# start, working back along that demo. Only the door and the clock pay.
# Entropy is bought at a fixed price rather than held at sm64.ini's target,
# and the learning rate is a third of PufferLib's 0.015: see the README
sm64-robustify: sm64
	uv run python mlx_pufferl.py train sm64 --env.backward 0.8 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 \
		--train.target-entropy 0 --train.ent-coef 0.01 --train.learning-rate 0.005 $(ARGS)

# Watch the most recent checkpoint play, in a window, at the speed a console ran
sm64-watch: sm64
	uv run python mlx_pufferl.py eval sm64 --load-model-path latest \
		--vec.total-agents 1 --env.window 1 --env.go-explore 0 $(ARGS)

# The game with no policy at all: hold forward and jump, in a window
sm64-demo: sm64
	./build/sm64_tool watch forward

# How many agent steps a second this machine manages, with N games at once
sm64-bench: sm64
	./build/sm64_tool bench $(or $(GAMES),8) 400
