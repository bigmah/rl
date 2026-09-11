# rl

RL experiments with [PufferLib](https://github.com/PufferAI/PufferLib) 4.0, vendored as a git submodule at `vendor/PufferLib` (branch `4.0`).

## Platformer

`envs/platformer/` is a single-level 2D platformer written as a PufferLib C env: run right, clear the pits and spikes, touch the flag.

- `platformer.h`: level, physics, observations, rewards, rendering
- `platformer.c`: human play
- `binding.c`: PufferLib vec-env binding
- `platformer.ini`: env and training config

Rewards: progress toward the flag (1.0 for the whole run), +1 for the flag, and -0.1 for dying or running out of time. Episodes end at the flag, on death, or after 30 s. Keep the failure penalty small: at -0.5, early deaths taught some seeds to stand still.

## Training on the Mac GPU with MLX

`mlx_pufferl.py` ports PufferLib's CUDA trainer to [MLX](https://github.com/ml-explore/mlx), so training runs on the Apple Silicon GPU. It uses PufferLib's regular policy and PPO variant:
- **Policy:** Linear encoder, then a 4-layer MinGRU (hidden size 128), then a Linear decoder.
- **PPO:** V-trace-clipped advantages, prioritized minibatches and the Muon optimizer.

It follows `pufferlib/torch_pufferl.py` but matches the CUDA kernels where the two differ:
- bias-free layers
- one decoder matrix whose last row is the value head
- √2 encoder init
- the CUDA advantage formula
- entropy-coefficient annealing

It installs itself as PufferLib's backend, so PufferLib's CLI, configs, dashboard, checkpoints and eval loop all work unchanged:

```sh
uv run python mlx_pufferl.py train platformer [--train.total-timesteps 20_000_000 ...]
uv run python mlx_pufferl.py eval platformer --load-model-path latest --vec.total-agents 1
```

The env still steps on the CPU (C with OpenMP). Advantages and minibatch sampling run in numpy; the policy and PPO updates run on the GPU. Limits: only the default MinGRU policy and discrete actions are implemented. Checkpoints hold MLX weights (numpy `.npz` data under puffer's `.bin` names), so PufferLib's torch and CUDA backends can't load them.

## Setup (macOS, Apple Silicon)

```sh
brew install libomp uv
make setup   # git submodule update --init && uv sync
```

## Play, train, watch

```sh
make play    # A/D move, W/Space jump, S drop through platforms, R restart, ESC quit
make train   # MLX on the GPU; checkpoints go to checkpoints/platformer/
make eval    # watch the latest checkpoint
```

Pass extra `puffer` flags with `ARGS`, e.g. `make train ARGS="--train.total-timesteps 10_000_000"`.

## Mac build notes

- `build.sh` stands in for PufferLib's Linux/CUDA build script. It compiles against Homebrew's OpenMP headers and links torch's bundled `libomp`, because two OpenMP runtimes in one process abort.
- `puffer` only reads configs from inside the PufferLib checkout, so `build.sh` symlinks `platformer.ini` into `vendor/PufferLib/config/`. The link is excluded from the submodule's git status.
