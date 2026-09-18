.PHONY: setup trainer test bench build play train eval sm64 sm64-state sm64-train sm64-watch sm64-demo sm64-bench sm64-explore sm64-robustify sm64-slide sm64-slide-explore sm64-slide-robustify sm64-slide-watch sm64-picture

# Which env to build and train. Each builds into a library of its own
# (build/vecenv_<env>.dylib, or .so), so building one leaves the others as they are.
ENV ?= platformer

# The trainer (src/) runs on wgpu: Metal, Vulkan or DX12, whatever the machine has
PUFFERL = ./target/release/pufferl

# One-time: fetch the PufferLib submodule and build the trainer
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
# The game itself comes from N64Bundler: drop your own sm64.z64 on it once so it
# is recompiled, and set N64BUNDLER if the checkout is not ../static_recomp/n64bundler.

sm64: trainer
	./build.sh sm64

# Play through the title, the file select and Peach's letter once, and save the
# castle grounds. Every episode starts from this. Made automatically on the
# first run; this remakes it.
sm64-state: sm64
	./build/sm64_tool state

sm64-train: sm64 sm64-state
	$(PUFFERL) train sm64 $(ARGS)

# Go-Explore's first phase with no policy: random play from the archive until it
# has opened the door, keeping the fastest run in build/sm64/door.demo
sm64-explore: sm64
	./build/sm64_tool explore $(or $(GAMES),8) $(or $(SECONDS),600) $(or $(FRAMES),900)

# Go-Explore's second phase: train a policy to open the door from the real
# start, working back along that demo. Only the door and the clock pay.
# Entropy is bought at a fixed price rather than held at sm64.ini's target,
# and the learning rate is a third of PufferLib's 0.015: see the README
sm64-robustify: sm64
	$(PUFFERL) train sm64 --env.backward 0.8 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 \
		--train.target-entropy 0 --train.ent-coef 0.01 --train.learning-rate 0.005 $(ARGS)

# Race to the star at the bottom of Peach's Secret Slide, where gravity does
# most of the work and the way to lose is to fall off. A death keeps the clock
# and puts the game back rather than ending the episode. Everything else is the
# usual reward: the star, the clock and novelty, explored with Go-Explore.
#
# The slide is about 1100 frames long at the pace an explorer slides it, so no
# episode ever fit the star into the 900-frame clock: this course gets 1500, and
# every frame of it still costs. Its checkpoints and logs are kept apart from the
# door's, so `latest` here is the slide's most trained policy and not the door's.
SLIDE = SM64_GOAL=star SM64_LEVEL=27
SLIDE_ARGS = --env.respawn 1 --env.max-ticks 1500 --base.checkpoint-dir checkpoints/slide --base.log-dir logs/slide
sm64-slide: sm64
	$(SLIDE) $(PUFFERL) train sm64 $(SLIDE_ARGS) $(ARGS)

# Go-Explore's first phase on the slide, with no policy: random sliding from the
# archive until it has a star, keeping the fastest in build/sm64/star-level27-act0.demo
sm64-slide-explore: sm64
	$(SLIDE) ./build/sm64_tool explore $(or $(GAMES),8) $(or $(SECONDS),600) $(or $(FRAMES),1500)

# Go-Explore's second phase on the slide: train a policy to reach the star from
# the top, working back along that demo, with only the star and the clock
# paying. Every star it gets faster than the demo becomes the demo for the next
# run. Advantages are normalized (norm_adv), which is 4.0's and not 5.0's:
# without it this learned nothing in 1.5M steps. The star is 750 steps from the
# top at most, so the discount is 0.999 rather than 5.0's 0.995, at which a star
# 400 frames off is worth a third. See the README.
sm64-slide-robustify: sm64
	$(SLIDE) $(PUFFERL) train sm64 $(SLIDE_ARGS) --env.backward 0.8 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 \
		--train.target-entropy 0 --train.ent-coef 0.005 --train.learning-rate 0.005 \
		--train.norm-adv 1 --train.gamma 0.999 --train.gae-lambda 0.95 --train.minibatch-size 1024 $(ARGS)

# Watch the slide's most trained checkpoint start at the top and go for the star
sm64-slide-watch: sm64
	$(SLIDE) $(PUFFERL) eval sm64 latest $(SLIDE_ARGS) --env.window 1 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 $(ARGS)

# Train with the policy looking at the game: an 80 by 60 picture of it in the
# observation, through a small convolutional net. About a quarter of the speed.
# Watch one with: make sm64-watch ARGS="--env.picture-width 80 --env.picture-height 60"
sm64-picture: sm64
	$(PUFFERL) train sm64 --env.picture-width 80 --env.picture-height 60 $(ARGS)

# Watch the most trained checkpoint play, in a window, at the speed a console ran
sm64-watch: sm64
	$(PUFFERL) eval sm64 latest --env.window 1 --env.go-explore 0 $(ARGS)

# The game with no policy at all: hold forward and jump, in a window
sm64-demo: sm64
	./build/sm64_tool watch forward

# How many agent steps a second this machine manages, with N games at once
sm64-bench: sm64
	./build/sm64_tool bench $(or $(GAMES),8) 400
