# rl

RL experiments with [PufferLib](https://github.com/PufferAI/PufferLib) 4.0, vendored as a git submodule at `vendor/PufferLib` (branch `4.0`).

## Platformer

`envs/platformer/` is a single-level 2D platformer written as a PufferLib C env: run right, clear the pits and spikes, touch the flag.

- `platformer.h`: level, physics, observations, rewards, rendering
- `platformer.c`: human play
- `binding.c`: PufferLib vec-env binding
- `platformer.ini`: env and training config

Rewards: progress toward the flag (1.0 for the whole run), +1 for the flag, -0.5 for dying. Episodes end at the flag, on death, or after 30 s.

### Setup (macOS, Apple Silicon)

```sh
brew install libomp uv
make setup   # git submodule update --init && uv sync
```

### Play, train, watch

```sh
make play    # A/D move, W/Space jump, S drop through platforms, R restart, ESC quit
make train   # checkpoints go to checkpoints/platformer/
make eval    # watch the latest checkpoint
```

Pass extra `puffer` flags with `ARGS`, e.g. `make train ARGS="--train.total-timesteps 10_000_000"`.

### Mac notes

- There's no CUDA, so training uses PufferLib's PyTorch backend (`--slowly`) on CPU with a stateless MLP policy.
- `build.sh` stands in for PufferLib's Linux/CUDA build script. It compiles against Homebrew's OpenMP headers and links torch's bundled `libomp`.
- `puffer` only reads configs from inside the PufferLib checkout, so `build.sh` symlinks `platformer.ini` into `vendor/PufferLib/config/`. The link is excluded from the submodule's git status.
