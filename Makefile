.PHONY: setup build check bench play train eval

# Which env: sm64 (the default) or platformer, e.g. make train ENV=platformer
ENV ?= sm64

# Fetch the submodules: PufferLib 5.0 (the trainer) and N64Bundler (the game)
setup:
	git submodule update --init

# ./puffer_$(ENV): PufferLib's CUDA trainer with envs/$(ENV)/$(ENV).h compiled
# in. For sm64 also the game from your sm64.z64 and build/sm64_check; for the
# platformer, build/platformer to play.
build:
	./build.sh $(ENV)

# Train. Flags pass through: make train ARGS="--train.total-timesteps=100_000_000"
# Checkpoints go to checkpoints/$(ENV)/<run id>/, the run's log to logs/$(ENV)/.
train: build
	./puffer_$(ENV) train $(ARGS)

# Score the most trained checkpoint, headless. Without --headless it opens a
# window and plays one env, which needs a display.
eval: build
	./puffer_$(ENV) eval latest --headless $(ARGS)

# The platformer yourself: A/D move, W/Space jump, S drop, R restart, ESC quit
play:
	./build.sh platformer
	./build/platformer

# sm64 without a GPU: the fastest run on file played through the reward.
# It should end "the star" at frame 674.
check:
	./build.sh sm64
	./build/sm64_check replay demos/peach-slide/star-674-touch.demo

# How many agent steps a second this machine's games manage, on random actions
bench:
	./build.sh sm64
	./build/sm64_check bench $(or $(GAMES),32) 1000
