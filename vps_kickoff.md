# VPS kickoff

Steps for a rented GPU box, from a fresh machine to a run of the slide. Do them in order. Each step says what you should see, so you know where it broke if it did.

## 0. Is it the right box?

```sh
uname -m            # aarch64: sm64 needs arm64. On x86_64 only the platformer will run
nvidia-smi          # the GPU and driver are there
nvcc --version || ls /usr/local/cuda/bin/nvcc   # the CUDA toolkit, 12.x
nproc               # cores: about half this many games at once is the sm64 sweet spot
df -h ~             # a few GB free: N64Bundler's build, the game, checkpoints
```

If `nvcc` isn't on PATH but `/usr/local/cuda/bin/nvcc` exists, run `export PATH=/usr/local/cuda/bin:$PATH` and add it to `~/.bashrc`. With no toolkit at all, install `cuda-toolkit-12-8` or similar from NVIDIA's CUDA apt repo, for `sbsa` on arm64.

## 1. Packages

```sh
sudo apt update
sudo apt install -y build-essential clang cmake ninja-build git curl python3 ccache jq tmux \
    libsdl2-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev
```

NCCL. Check whether the image already has it:

```sh
ls /usr/include/nccl.h /usr/lib/$(uname -m)-linux-gnu/libnccl.so* 2>/dev/null
```

If it doesn't, install one of these:

```sh
sudo apt install -y libnccl2 libnccl-dev     # needs NVIDIA's CUDA apt repo
pip install nvidia-nccl-cu12                 # build.sh finds the wheel too
```

## 2. Clone

```sh
git clone -b core-cuda --recurse-submodules https://github.com/bigmah/rl.git rl
cd rl
git log --oneline -1                          # the core-cuda commit you expect
```

If the submodules didn't come down, run `git submodule update --init`.

## 3. Smoke test: the platformer (no ROM, a few minutes)

This checks CUDA, NCCL, the build and the trainer, without the game.

```sh
tmux new -s train
make train ENV=platformer
```

- The first build compiles raylib from source (arm64 has no release of it) and then the trainer with nvcc. That takes a minute or two, and ends with `Built: ./puffer_platformer`.
- The dashboard should come up and count steps. `perf`, the share of episodes that reach the flag, should climb toward 1 by 20M steps. On main's Rust/wgpu port of this same trainer it reached 99.7%.
- Then run `make eval ENV=platformer` to score the latest checkpoint, headless.

Note the steps per second on the dashboard. That's the trainer with a cheap env, a ceiling for sm64.

If this fails, fix it before going on. See the troubleshooting section at the end.

## 4. The ROM

Copy your own dump of Super Mario 64 (USA) up from the Mac, into the top of the checkout:

```sh
# on the Mac
scp /Users/tonyradtke/dev/static_recomp/n64bundler/sm64.z64 <user>@<box>:rl/
```

`build.sh` checks that it is exactly that dump (game id `NSME`, ROM hash `8a90daa33e09a265`). Alternatively, put it anywhere and set `export SM64_ROM=/path/to/sm64.z64`.

## 5. Build and check sm64 (5-10 minutes the first time)

```sh
make build        # N64Bundler, the game recompiled from the ROM, ./puffer_sm64, build/sm64_check
make check
```

`make check` should print:

```
sm64: playing through the intro and into Peach's Secret Slide once, to make .../build/sm64/pss-2/start.state
...
the episode ended at frame 674 with a reward of 0.999 on its last step: the star
return 1.767, of which getting closer paid 0.991; closest 167 units
```

That means the game runs, the savestate got made, and the reward pays the star. If it says anything other than "the star" at frame 674, stop: something in the game or the env is off.

## 6. How many games

```sh
make bench GAMES=16
make bench GAMES=32
make bench GAMES=48
make bench GAMES=64
```

Pick the count where steps a second stops going up. It's usually around half of `nproc`. Then set it in `config/sm64.ini` under `[vec]`, or pass it as flags:

- `total_agents` = the game count
- `num_threads` = the same number, a thread per game
- `minibatch_size`: the trainer asserts that `total_agents * horizon >= minibatch_size` and that `total_agents` divides evenly by `minibatch_size / horizon`. With `horizon = 64`:

| games | minibatch_size |
|---|---|
| 16, 32, 48 | 1024 |
| 64 | 1024, 2048 or 4096 |
| 128 | up to 8192 |

## 7. Train

In tmux, so it survives the SSH session:

```sh
make train
# or with flags, e.g. 64 games and a longer run:
make train ARGS="--vec.total-agents=64 --vec.num-threads=64 --train.minibatch-size=2048 --train.total-timesteps=200_000_000"
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t train`.

On the dashboard, these tell you if it's learning:

- `progress`: the share of the way to the star covered at Mario's closest. It should rise over the first few million steps.
- `closest`: the same thing in units. The start is about 18,900 from the star.
- `perf`: the star rate. It stays at 0 until runs get to the bottom inside 21 seconds of the slide's timer, since slower runs find no star.
- `ended_early`: deaths. If this sits near 1, Mario is going over the side.
- `frames`: frames to the star, with the whole clock (1500) counted for runs that don't get it.

Output goes to:

- checkpoints: `checkpoints/sm64/<run id>/<step>.bin`, every 50 epochs
- the run's config and metrics: `logs/sm64/<run id>.ini`, written when the run ends

To stop a run, press `Ctrl-C` in its tmux window. If any games outlive it, `pkill -f n64b-run`.

To score the latest checkpoint: `make eval` (headless, 64 episodes).

To continue from a checkpoint: `make train ARGS="--base.load-model-path=checkpoints/sm64/<run id>/<step>.bin"`.

## 8. Bring results home

```sh
# on the Mac, from the repo
rsync -a <user>@<box>:rl/checkpoints/ checkpoints/
rsync -a <user>@<box>:rl/logs/ logs/
```

Before giving the box back, copy off anything you want to keep. Checkpoints are small, but the box is not coming back.

## Troubleshooting

| symptom | fix |
|---|---|
| `No nvcc` | `export PATH=/usr/local/cuda/bin:$PATH`, or `export CUDA_HOME=/usr/local/cuda` |
| `No NCCL` | step 1: `libnccl-dev` or `pip install nvidia-nccl-cu12` |
| link errors naming `cusparse`, `nvJitLink` and so on | the toolkit is partial: `sudo apt install cuda-toolkit-12-X` (match the installed version) |
| `libnvidia-ml.so.1: cannot open shared object file` | no NVIDIA driver loaded; `nvidia-smi` should work first |
| nvcc rejects `-arch=native` | name the GPU's architecture: `NVCC_ARCH=sm_90 make build` (GH200/H100), `sm_80` (A100) |
| `assertion ... minibatch_size ...` | step 6's table |
| `No Super Mario 64 ROM` or `is not Super Mario 64 (USA)` | step 4 |
| `the game never reached the castle grounds` | delete `build/sm64/pss-2/start.state` and run `make check` again; if it repeats, look at `build/sm64/game.log` |
| raylib didn't build | the apt line in step 1 (the X11 `-dev` packages) |
| bf16 trouble on an older GPU | `PRECISION=float make build` |
| an x86_64 box | the platformer works; sm64 doesn't, because the recompiled game is arm64 code |
