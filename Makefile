.PHONY: setup trainer test bench build play train eval sm64 sm64-state sm64-train sm64-explore sm64-robustify sm64-picture sm64-watch sm64-replay sm64-demo sm64-bench

# Which env to build and train. Each builds into a library of its own
# (build/vecenv_<env>.dylib, or .so), so building one leaves the others as they are.
ENV ?= platformer

# The trainer (src/) runs on wgpu: Metal, Vulkan or DX12, whatever the machine has
PUFFERL = ./target/release/pufferl

# One-time, and optional: fetch the submodules and build the trainer
setup:
	git submodule update --init
	cargo build --release

trainer:
	cargo build --release

# The kernels against the CPU, acting against training, and the whole of a training
# step against golden numbers (tests/golden)
test:
	cargo test --release

# Where the time goes on this GPU, with no env
bench:
	cargo run --release --example bench_matmul
	cargo run --release --example bench_train
	cargo run --release --example bench_latency

build: trainer
	./build.sh $(ENV)

# A/D move, W/Space jump, S drop through platforms, R restart, ESC quit
play: build
	./build/platformer

# Extra flags pass through, e.g. make train ARGS="--train.total-timesteps 10_000_000"
# On the GPU: checkpoints go to checkpoints/<env>/
train: build
	$(PUFFERL) train $(ENV) $(ARGS)

# Watch the most trained checkpoint play one env (ESC to quit)
eval: build
	$(PUFFERL) eval $(ENV) latest $(ARGS)

# --- Super Mario 64 ----------------------------------------------------------
# The game is your own sm64.z64, Super Mario 64 (USA), at the top of this checkout
# or wherever SM64_ROM says. The first build fetches N64Bundler (vendor/n64bundler),
# builds it and recompiles the game with it; N64BUNDLER names another checkout.
#
# Every target is about one star, STAR: <course>-<number> as the game numbers a
# course's stars (pss-2, wf-1, bob-7), or a course alone for any star in it.
# `./build/sm64_tool courses` lists the courses; without STAR it is the star
# sm64.ini names. A star with settings of its own -- a longer clock, respawning --
# has them in envs/sm64/stars/$(STAR).ini, which goes over sm64.ini; any other
# star needs no file. A star keeps its savestate, demos and seeds in
# build/sm64/$(STAR)/, and its checkpoints and logs in checkpoints/$(STAR)/ and
# logs/$(STAR)/, so `latest` is that star's most trained policy.
STAR ?= $(shell sed -n 's/^star *= *\([a-z0-9-]*\).*/\1/p' envs/sm64/sm64.ini)
STAR_INI = envs/sm64/stars/$(STAR).ini
STAR_CONFIG = $(if $(wildcard $(STAR_INI)),--config $(STAR_INI)) --env.star $(STAR)
STAR_RUN = $(STAR_CONFIG) --base.checkpoint-dir checkpoints/$(STAR) --base.log-dir logs/$(STAR)

sm64: trainer
	./build.sh sm64

# Get into the star's course and save the game there. Every episode starts from
# this. Made automatically the first time a star is trained; this remakes it.
sm64-state: sm64
	./build/sm64_tool state $(STAR_CONFIG)

# The star, the clock and novelty, explored with Go-Explore
sm64-train: sm64
	$(PUFFERL) train sm64 $(STAR_RUN) $(ARGS)

# Go-Explore's first phase with no policy: random play from the archive until it
# has the star, keeping the fastest run in build/sm64/$(STAR)/fastest.demo
sm64-explore: sm64
	./build/sm64_tool explore $(or $(GAMES),8) $(or $(SECONDS),600) $(STAR_CONFIG)

# Go-Explore's second phase: train a policy to get the star from the real start,
# working back along that run, with only the star and the clock paying. Every
# star it gets faster than the run becomes the run for the next time.
# Advantages are normalized (norm_adv), which is 4.0's and not 5.0's: without it
# the slide learned nothing in 1.5M steps. The discount is 0.999 rather than
# 5.0's 0.995, at which a star 400 frames off is worth a third. Entropy is bought
# at a fixed price rather than held at sm64.ini's target, and the learning rate
# is a third of PufferLib's 0.015: see the README.
sm64-robustify: sm64
	$(PUFFERL) train sm64 $(STAR_RUN) --env.backward 0.8 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 \
		--train.target-entropy 0 --train.ent-coef 0.005 --train.learning-rate 0.005 \
		--train.norm-adv 1 --train.gamma 0.999 --train.gae-lambda 0.95 --train.minibatch-size 1024 $(ARGS)

# Train with the policy looking at the game: an 80 by 60 picture of it in the
# observation, through a small convolutional net. About a quarter of the speed.
# Watch one with: make sm64-watch ARGS="--env.picture-width 80 --env.picture-height 60"
sm64-picture: sm64
	$(PUFFERL) train sm64 $(STAR_RUN) --env.picture-width 80 --env.picture-height 60 $(ARGS)

# Watch the star's most trained checkpoint start where the task does and go for
# it, in a window, at the speed a console ran
sm64-watch: sm64
	$(PUFFERL) eval sm64 latest $(STAR_RUN) --env.window 1 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 $(ARGS)

# Watch the fastest run to the star on file
sm64-replay: sm64
	./build/sm64_tool replay watch $(STAR_CONFIG)

# The game with no policy at all: hold forward and jump, in a window
sm64-demo: sm64
	./build/sm64_tool watch forward $(STAR_CONFIG)

# How many agent steps a second this machine manages, with N games at once
sm64-bench: sm64
	./build/sm64_tool bench $(or $(GAMES),8) 400 $(STAR_CONFIG)
