.PHONY: setup build check bench train eval

# Fetch the submodules: PufferLib 5.0 (the trainer) and N64Bundler (the game)
setup:
	git submodule update --init

# The game from your sm64.z64, sm64_check, and ./puffer (PufferLib's CUDA
# trainer with envs/sm64/sm64.h compiled in)
build:
	./build.sh

# The env without a GPU: the fastest run on file played through the reward.
# It should end "the star" at frame 674.
check: build
	./build/sm64_check replay demos/peach-slide/star-674-touch.demo

# How many agent steps a second this machine's games manage, on random actions
bench: build
	./build/sm64_check bench $(or $(GAMES),32) 1000

# Train. Flags pass through: make train ARGS="--train.total-timesteps=100_000_000"
# Checkpoints go to checkpoints/sm64/<run id>/, the run's log to logs/sm64/.
train: build
	./puffer train $(ARGS)

# Score the most trained checkpoint (no window: the box has no display)
eval: build
	./puffer eval latest --headless $(ARGS)
