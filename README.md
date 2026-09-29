# rl — core-cuda

Super Mario 64's Peach's Secret Slide, trained with [PufferLib](https://github.com/PufferAI/PufferLib) 5.0's own CUDA trainer, on arm64 Linux with an NVIDIA GPU (a GH200 or any other Grace or arm64 box).

There's also a small 2D platformer (`envs/platformer/`) built the same way. It needs no ROM, builds in a minute and trains in a few, which makes it a quick way to check that a new GPU box works.

This branch is not `main`. There's no Rust or wgpu trainer here, and no novelty, Go-Explore or demos in the reward. The trainer is PufferLib's `pufferl.cu`, unmodified, with the env compiled in. The env's observation is everything about Mario that can be read out of memory, and its reward is the star's location and nothing else.

The game is the real cartridge. [N64Bundler](https://github.com/bigmah/n64bundler) statically recompiles your ROM into native arm64, and each agent runs its own copy of the game in a process of its own. That's why the box has to be arm64: the recompiled game is arm64 code.

## On the box

[`vps_kickoff.md`](vps_kickoff.md) is the step-by-step for a fresh box, with what each step should print and what to do when it doesn't. In short:

You need an arm64 Linux machine with an NVIDIA driver, the CUDA toolkit (12.x, with `nvcc`) and NCCL, plus your own dump of Super Mario 64 (USA).

```sh
sudo apt install build-essential clang cmake ninja-build git curl python3 ccache jq \
    libsdl2-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev
# NCCL, if the image doesn't have it (NVIDIA's CUDA apt repo), or: pip install nvidia-nccl-cu12
sudo apt install libnccl2 libnccl-dev

git clone -b core-cuda --recurse-submodules https://github.com/bigmah/rl.git rl && cd rl
export PATH=/usr/local/cuda/bin:$PATH
```

Check the box first with the platformer. It needs no ROM and no N64Bundler.

```sh
make train ENV=platformer   # builds ./puffer_platformer, then 20M steps
make eval ENV=platformer    # score the most trained checkpoint, headless
```

Its `perf` is the fraction of episodes that reach the flag. On `main`'s wgpu port of this same trainer, 20M steps reached 99.7%.

Then the slide:

```sh
cp /path/to/sm64.z64 .      # or: export SM64_ROM=/path/to/sm64.z64
make build     # N64Bundler, the game recompiled from your ROM, ./puffer_sm64, build/sm64_check
make check     # the record run played through the reward: should end "the star" at frame 674
make bench     # agent steps a second with 32 games on random actions
make train     # ./puffer_sm64 train, checkpoints in checkpoints/sm64/<run id>/
make eval      # score the most trained checkpoint, headless
```

`ENV` defaults to `sm64`. Each env builds its own binary, `./puffer_<env>`, so building one leaves the other as it was.

The first sm64 build takes a few minutes: it builds N64Bundler and raylib (which has no arm64 release, so it's built from source), recompiles the game in about twenty seconds, and then compiles the trainer with nvcc. Later builds find the first three done.

Flags go to the trainer as `--section.key=value`, which is PufferLib's syntax:

```sh
make train ARGS="--train.total-timesteps=200_000_000 --vec.total-agents=64 --vec.num-threads=64 --train.minibatch-size=4096"
./puffer_sm64 train --train.learning-rate=0.005
./puffer_sm64 eval checkpoints/sm64/<run id>/<step>.bin --headless
```

The build's own knobs are environment variables: `NVCC_ARCH` (default `native`; set `sm_90` to build on a machine without the GPU), `PRECISION=float` for fp32 instead of bf16, `CUDA_HOME`, `SM64_ROM` and `N64BUNDLER`.

### How many games

`[vec] total_agents` in `config/sm64.ini` is how many copies of the game run at once. Each one keeps a bit more than a core busy, so throughput peaks at about half the cores: the config says 32, for a 72-core Grace. Keep `num_threads` equal to `total_agents`, so every game gets a thread to wait on it. `make bench GAMES=48` tells you what the box actually manages.

The trainer adds its own constraints. `total_agents × horizon` has to be at least `minibatch_size`, and `total_agents` has to divide evenly by `minibatch_size / horizon`. With the config's horizon of 64 and minibatch of 1024, any multiple of 16 works.

## The env (`envs/sm64/sm64.h`)

- **Start.** Every episode starts where the slide's painting drops Mario, from a savestate that the first game makes by playing through the intro and in through the castle door. That happens once, taking about a minute, and the state is kept in `build/sm64/pss-2/start.state`. After loading it, up to `random_start` frames of a random stick direction make the starts differ.
- **Actions.** Four discrete heads: a stick direction (none, or one of 16 relative to the way Mario faces; the env measures the camera and corrects for it), and A, B and Z held or not. One decision is `frameskip` = 2 frames.
- **Observation.** 259 floats, all read out of the console's memory each step:
  - position, velocity, forward speed and ground speed
  - facing (yaw, pitch, roll) and angular velocity
  - slide yaw and slide velocity, and twirl yaw
  - the stick as the game read it (magnitude and direction), and the camera, both relative to his facing
  - height above the floor, the floor's height, normal, angle and surface type
  - the wall's normal, whether there's a ceiling, and headroom
  - water level, health and peak height
  - the A/B, wall-kick, double-jump and invincibility timers
  - his action as group, id and 23 flag bits, one-hot; his previous action's group and id
  - his action's timer, state and argument
  - the 16 input bits and 16 Mario flag bits
  - the star's offset, distance and direction from him, the closest he has been this episode, and how much of the clock is left
- **Reward.** It uses the star's location and nothing else about the course:
  - **Getting closer.** Each time Mario gets nearer the star than he has been this episode, he's paid `progress` × (distance gained) / (distance at the start), so the whole way down is worth `progress` (1.0). Only the closest approach pays, so going back and forth earns nothing. It only pays while there's floor under him, so falling out of the world toward the star earns nothing either.
  - **The star.** It pays 1.0 the frame Mario touches it (its bit in the save file) and ends the episode.
  - **The clock.** Every frame costs `time_penalty / max_ticks` (0.5 over the 1,500-frame clock). A death or a warp out of the course charges the rest of the clock at once and ends the episode.
- **Star location.** `star_x/y/z` = (-6358, -4300, 4700), where the slide's timer star comes to rest. Every recorded run touches it within 150 units of there.

`pss-2` is the slide's *second* star, the one that spawns only if Mario reaches the bottom within 21 seconds of the slide's own timer. A slower run reaches the spot and finds no star, so the distance reward gets a policy to the bottom, and only a fast one gets the star. The distance is a straight line, and the slide winds: from about 8,500 units out there's a stretch of track (frames 240 to 420 of the record run) that doesn't get any closer, so it pays nothing until the lower track.

The log (PufferLib's dashboard, and `logs/sm64/<run id>.ini` at the end):

- `perf`: the fraction of episodes that took the star
- `score`: the clock left when he took it
- `frames`: frames to the star, or the whole clock without one
- `progress`: the share of the way covered at his closest
- `closest`: that closest distance, in units
- `progress_reward`, `top_speed`, `airborne` and `ended_early`

## `sm64_check`

This is the env with no trainer and no GPU. It reads the same config from the same place (`config/default.ini`, `config/sm64.ini`, then flags).

```sh
./build/sm64_check replay [file.demo]      # a run of pad inputs through puf_step's reward
./build/sm64_check bench [games] [steps]   # random actions: steps a second, what episodes did
```

`demos/peach-slide/` holds the fastest runs found on `main`. `star-674-touch.demo` touches the star at frame 674 (22.5 s), for a return of 1.77 under this reward.

## Layout

```
envs/sm64/sm64.h        the env: PufferLib 5.0's env API, compiled into pufferl.cu
envs/sm64/n64b_gym.h    one recompiled game in a process of its own, driven from here
envs/sm64/sm64_check.c  the env without the trainer
envs/platformer/        platformer.h (the env), platformer.c (play it: make play, needs a display)
config/<env>.ini        each env's and trainer's settings, over config/default.ini
config/default.ini      -> vendor/PufferLib/config/default.ini
build.sh [env]          ./puffer_<env>, and for sm64 the game and sm64_check
demos/peach-slide/      the record runs from main
vendor/PufferLib        submodule, branch 5.0: the trainer
vendor/n64bundler       submodule: the recompiler and the host the game runs in
```

`./puffer_<env>` reads its config from `config/` in the directory it's run in, so run it from the top of the checkout. `build/`, `puffer_*`, `checkpoints/`, `logs/` and ROMs are gitignored.
