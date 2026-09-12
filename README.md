# rl

RL experiments with [PufferLib](https://github.com/PufferAI/PufferLib) 4.0, vendored as a git submodule at `vendor/PufferLib` (branch `4.0`).

Two envs live here. `make ENV=<name> train` builds one into PufferLib's extension and trains it; PufferLib compiles one env at a time, so building another replaces it.

## Platformer

`envs/platformer/` is a single-level 2D platformer written as a PufferLib C env: run right, clear the pits and spikes, touch the flag.

- `platformer.h`: level, physics, observations, rewards, rendering
- `platformer.c`: human play
- `binding.c`: PufferLib vec-env binding
- `platformer.ini`: env and training config

Rewards: progress toward the flag (1.0 for the whole run), +1 for the flag, and -0.1 for dying or running out of time. Episodes end at the flag, on death, or after 30 s. Keep the failure penalty small: at -0.5, early deaths taught some seeds to stand still.

## Super Mario 64

`envs/sm64/` is the real cartridge as an environment, and the whole goal is **speed**: the reward is how far Mario actually moves, and nothing else.

The game is not emulated and not reimplemented. [N64Bundler](https://github.com/bigmah/n64bundler) statically recompiles `sm64.z64` into native arm64 and runs it in `n64b-run`; this env starts one of those per agent, in a process of its own, and reads Mario out of the console's memory — which is mapped into both processes, so an observation is a load rather than a request.

- `sm64.h`: where the game keeps Mario, what an action does to the controller, the reward
- `n64b_gym.h`: starting a game, stepping it, saving and loading its state
- `sm64.c`: the env without the trainer — make the savestate, watch it, benchmark it
- `sm64.ini`: env and training config

```sh
make sm64-state       # play through the intro once and save the castle grounds
make ENV=sm64 train   # 8 copies of the game, ~9000 agent steps a second
make sm64-watch       # watch the latest checkpoint play, in a window
make sm64-demo        # no policy at all: hold forward and jump
```

It needs your own dump of the cartridge, recompiled once: drop `sm64.z64` on N64Bundler so it appears in its library as `NSME`. `build.sh` finds the recompiled game there. Set `N64BUNDLER` if that checkout is not at `../static_recomp/n64bundler`. No game data is copied into this repository, and everything derived from it lands in the gitignored `build/`.

### Measuring Mario's speed

Everything the env reads was found in the C that N64Bundler generates from the cartridge, then checked against the running game. `mario_set_forward_vel` is the function that pins the whole structure down, and in the generated code at `0x80251708` it is unmistakable:

```c
MEM_W(0X54, ctx->r4) = ctx->f4.u32l;      // forwardVel
ctx->r14 = MEM_HU(ctx->r4, 0X2E);         // faceAngle[1], his yaw
ctx->f6.u32l = MEM_W(ctx->r1, 0X6000);    // the sine table at 0x80386000, indexed by yaw >> 4
ctx->f10.fl = MUL_S(ctx->f6.fl, ctx->f8.fl);
MEM_W(0X58, ctx->r4) = ctx->f10.u32l;     // slideVelX
// ... the cosine table at 0x80387000 ...
MEM_W(0X48, ctx->r4) = ctx->f6.u32l;      // vel[0] = slideVelX
MEM_W(0X50, ctx->r4) = ctx->f8.u32l;      // vel[2] = slideVelZ
```

That gives `forwardVel` at `0x54`, the velocity vector at `0x48` and the facing angle at `0x2E` in one read, and position sits just before them at `0x3C`. `gMarioState` is the pointer at `0x8032D93C`, which points at `0x8033B170`. Each of them was then confirmed by watching the game: the stick arrives in `gControllers[0]`, Mario's action becomes `ACT_WALKING` (`0x04000440`) and then `ACT_JUMP` (`0x03000880`), and `gGlobalTimer` at `0x8032D5D4` ticks exactly once per frame the game draws.

**The reward is distance covered, not `forwardVel`.** Forward velocity is what the game believes about Mario, and it can be enormous while he grinds into a wall going nowhere — rewarding it would pay for the belief rather than the motion. Distance per frame counts a long jump (about 48 units a frame), a slope he slides down, and whatever exploit turns out to beat both. `forward_vel` is logged next to it so the two can be compared.

### The savestate

A new file opens with Peach's letter, Lakitu's arrival and Mario climbing out of the pipe: about 1500 frames of cutscene that waits on A for its text boxes. No episode should watch that, so it is played through **once** — two presses of Start through the menus, then A until Mario stands in the castle grounds in `ACT_IDLE` — and saved. It takes 0.4 seconds, because the game runs at about 5000 frames a second with nothing drawing it.

Every episode then begins by putting that state back, which is 8 MB of console memory plus the registers of every thread of the game. `make sm64-state` remakes it; the first run makes it on its own.

### Actions and observations

Four discrete heads: a stick direction (none, or one of sixteen **relative to the way Mario is facing**), and A, B and Z held or not. Aiming relative to Mario rather than to the pad matters because the game turns a stick direction into a heading by adding the camera's yaw to it — a policy pushing "up" would run wherever the camera drifted. The offset between the two is measured rather than looked up: whatever angle we asked for last frame, the game wrote down what it made of it in `intendedYaw`, and the difference is the camera.

The observation is 40 floats: position, velocity, forward velocity, the speed just achieved, his facing and the camera relative to it, height above the floor, the floor's steepness, headroom, whether he is on a wall, in water or in the air, his action's group and flags, and the clock.

An episode ends after `max_ticks` frames, or the moment Mario warps, dies or enters the castle — a level change is a teleport, and a teleport is not running.

### What this needed from the runtime

A game that plays at sixty frames a second for a person is useless to something learning from it, so three things were added to N64Bundler (as patches `0018`-`0020` in its own series, plus a `--gym` mode in `n64b-run`):

- **The retrace comes off the clock.** The host asks for each interrupt of the video interface and the game advances only then — as fast as the machine manages rather than as fast as a television.
- **The console says when it has finished a frame.** A frame is over not when the picture is handed over but when nothing is left: every thread parked waiting for the hardware, every message delivered, every task the game gave the signal processor answered. That is checked rather than waited out, which is what makes a step deterministic.
- **It can be put down and picked up again.** A savestate is the console's memory plus the registers of each of the game's threads, taken while it is quiet and put back into a game that is quiet in the same shape — including a game that has only just booted, which is how an episode restarts.

Drawing is optional: headless swaps the renderer for one that counts the game's frames and discards them, which is most of the speed. The game still builds its display lists, because that is its own code.

### How fast

On an M4 Pro (10 performance cores), headless, one agent step being two frames of the game:

| games | agent steps/s | game frames/s |
|---|---|---|
| 1 | 2 900 | 5 700 |
| 4 | 6 400 | 12 800 |
| 8 | 9 200 | 18 300 |
| 16 | 8 500 | 17 100 |

Each game keeps more than a core busy — its own threads, and the renderer's — so throughput peaks around eight and falls off above it. A console runs this game at 30 frames a second; eight of these run it at 18 000.

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

Pass extra `puffer` flags with `ARGS`, e.g. `make train ARGS="--train.total-timesteps 10_000_000"`, and pick the env with `ENV=sm64`.

## Mac build notes

- `build.sh` stands in for PufferLib's Linux/CUDA build script. It compiles against Homebrew's OpenMP headers and links torch's bundled `libomp`, because two OpenMP runtimes in one process abort. The standalone `sm64_tool` links Homebrew's copy instead, since nothing in it loads torch.
- `puffer` only reads configs from inside the PufferLib checkout, so `build.sh` symlinks the env's `.ini` into `vendor/PufferLib/config/`. The link is excluded from the submodule's git status.
- The sm64 env gives each game one OpenMP thread rather than one per core: a thread stepping a game spends its time waiting on a socket, so the usual rule would run the games one after another.
