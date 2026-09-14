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

`envs/sm64/` is the real cartridge as an environment, and the goal is to **open the castle's front door as soon as possible**, starting from the castle grounds.

The game is not emulated and not reimplemented. [N64Bundler](https://github.com/bigmah/n64bundler) statically recompiles `sm64.z64` into native arm64 and runs it in `n64b-run`; this env starts one of those per agent, in a process of its own, and reads Mario out of the console's memory — which is mapped into both processes, so an observation is a load rather than a request.

- `sm64.h`: where the game keeps Mario, what an action does to the controller, the reward
- `n64b_gym.h`: starting a game, stepping it, saving and loading its state
- `sm64.c`: the env without the trainer — make the savestate, watch it, benchmark it, explore without a policy, replay the demo
- `sm64.ini`: env and training config

```sh
make sm64-state       # play through the intro once and save the castle grounds
make ENV=sm64 train   # 8 copies of the game, ~9000 agent steps a second
make sm64-watch       # watch the latest checkpoint play, in a window
make sm64-demo        # no policy at all: hold forward and jump
./build/sm64_tool probe door   # a scripted walk to the door, printing the reward
make sm64-explore     # Go-Explore phase 1 with no policy: find the door, keep the fastest run
./build/sm64_tool replay       # check that run still opens the door (add `watch` to see it)
make sm64-robustify   # Go-Explore phase 2: train a policy to open it from the real start
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

### The reward: time to the castle door

The door is two objects, its halves, at x −76 and 77, y 803, z −3155. They were found by searching memory for positions in front of the castle, then confirmed by walking into each one. Walking into the left half starts `ACT_PUSHING_DOOR` (`0x1321`) and the right half starts `ACT_PULLING_DOOR` (`0x1320`). Either way the game warps to the castle's inside (level 6) about forty frames later. **The episode ends the frame the door starts to open**, because the warp takes the same forty frames on every run.

**The reward is the door, the clock, and novelty.**

- **The door** pays 1, the most a step can pay once rewards are clipped.
- **The clock.** Every frame costs `time_penalty / max_ticks`, so running out of time costs `time_penalty` in all. The sooner the door opens, the less of the clock was spent.
- **Novelty.** The grounds are cut into cubes `novelty_cell` (500) units a side. The first time in an episode that Mario enters a cube, he is paid `novelty_episode + novelty / sqrt(n)`, where `novelty_episode` is 0.02, `novelty` is 0.05, and `n` counts the episodes, in all eight games, that have entered that cube (this one included).
- **Deaths and warps** out of the castle grounds charge whatever is left of the clock in one step, so neither is a way to stop it early. A death doesn't change the level (Mario respawns in the castle grounds), so deaths are caught by health dropping under `0x100`.

Nothing tells the policy where the door is or pays it for getting nearer. It sees what Mario is doing and where he is, and learns about the door by opening it.

**Why novelty.** The door is 7,600 units away in a straight line, past a moat, over a bridge and through Lakitu, and it opens only for Mario walking into it. Here is how the door and the clock did on their own:
- **Random play:** in 400 episodes of random buttons nobody opened the door, and only one got as far as Lakitu.
- **Training:** 12M steps never opened it. Every episode was worth exactly −1, entropy stayed at its maximum, and the typical episode ended as far from the door as it started.

Nothing in that setup remembers where Mario has been, so exploring was just jittering the stick.

**What novelty does.** Novelty is that memory, and it knows nothing about the door. It has two parts.

- **The fading part** (`novelty / sqrt(n)`) is for places nobody has been. The start, which every episode sees, is worth almost nothing within minutes, while a cube nobody has reached pays the full 0.05. On its own it found the door within 100K steps and then lost it. By 500K, the cubes on the way had been entered thousands of times and novelty had fallen from 0.7 an episode to 0.05. Episodes drifted back to the start, and in 5M steps the door opened only about five times, too rarely to learn the bridge, Lakitu and the door from.
- **The per-episode part** (`novelty_episode`) never fades. An episode that covers ground is always worth more than one that doesn't, so the far side of the grounds keeps being reached, and the door with it. 0.02 a cube is about what running costs in clock. A run to the door (15 cubes, the door, and the clock left over) is still worth more than wandering all episode.

Only the first entry in each episode counts, so pacing back and forth across a cube's edge earns nothing.

The scripted walk to the door enters 22 cubes. In a fresh process it returns 1.96: 0.42 for the door and the clock, 0.44 from the per-episode part, and 1.10 from the fading part. Repeated in the same process, it returns 1.64 and then 1.50.

**Entropy.** No fixed `ent_coef` worked with this reward. At 0.05 the policy sat at exactly uniform, learning nothing. At 0.005 it went out onto the bridge within 350K steps, then collapsed by 1M into a routine that barely moved. So `mlx_pufferl.py` can hold entropy at a `target_entropy` (2.5 here), raising the coefficient while the policy is more certain than that and lowering it while it is less. The setting is off by default.

**Keeping the trainer stable.** Before a door is found, the reward carries almost no signal. Muon's updates are the same size however weak the gradient is, so in that stretch the weights random-walk. The first door-and-clock run blew up at 3.4M steps: value loss went to 567,829 and entropy to zero. `weight_decay = 0.01` in `sm64.ini` keeps the weights bounded. The setting defaults to 0 in `mlx_pufferl.py`, which is what PufferLib uses.

An earlier version also paid for closing the straight-line distance to the door. It ended the episode in water, so that the distance couldn't lure Mario into the moat. That version opened the door in 92% of episodes after 3M steps.

Two things in the way were found by driving the game with a script. The door opens only for Mario walking into it: a dive bonks off, and a punch does nothing. And the first time Mario steps onto the bridge, Lakitu stops him to explain the camera, which is about 250 frames of dialog even with A mashed. The episode keeps going through the dialog (the policy has A), and the clock keeps running.

`./build/sm64_tool probe door` walks to the foot of the bridge, then to the door, mashing A through Lakitu. It opens the door after 518 frames with a return of 1.52 (0.42 without the novelty of a fresh process), and 49% of those frames are dialog.

The log reports:
- `perf`: the fraction of episodes that open the door.
- `score`: the fraction of the clock left when the door opened, or 0 if it didn't.
- `frames`: frames to the door, or the whole clock if it stayed shut. This is the number being minimized.
- `closest`: the nearest an episode without the door came to it. It is logged only; the policy never sees it. The bridge starts about 2,980 units from the door.
- `novelty`, `cells` and `explored`: what novelty paid in the episode, how many cubes the episode entered, and how many cubes any game has ever entered.
- `dialog`, path length, `top_speed`, `forward_vel` and `airborne`.
- `from_start` and `start_perf`: the share of episodes that began where the task does, and the share that began there and opened the door. `start_perf / from_start` is the door rate that counts.
- `archive`, `replayed`, `door_cubes` and `frontier`: see Go-Explore, below.

### Go-Explore

Novelty alone swam the moat for 10M steps. So `sm64.h` also runs [Go-Explore](https://arxiv.org/abs/1901.10995). Its first phase explores: it remembers places and goes back to them before exploring further. Its second phase robustifies: it trains a policy to do reliably, from the real start, what exploring only managed once.

**Going back is replaying inputs.** The game is deterministic, so the pad inputs from the savestate *are* the place, at a few kilobytes against a savestate's 8 MB. The archive keeps, for every cube, the shortest run of inputs that reached it. With `go_explore` on, that share of episodes picks a cube, weighted to the ones fewest episodes have entered, and replays its run before the clock starts.

**Phase 1 without a policy.** `make sm64-explore` is the first phase as the paper ran it: every episode starts from the archive and plays at random. Any exploring run that opens the door writes its inputs to `build/sm64/door.demo`, if nothing on file was faster. It found the door in 200 seconds, and in ten minutes it had opened it 132 times, the fastest in 690 frames (the scripted walk takes 518).

It found nothing until episodes were long enough to get through Lakitu. His speech starts and ends in the same cube on the bridge, so the archive can't keep a place partway through it. An episode has to get from the cube before the bridge to the one after it, speech and all. With 300-frame episodes and A held a few steps at a time, 2,300 episodes never got closer to the door than 1,088 units. With 900 frames and A drawn every step, speeches started finishing within a minute.

`./build/sm64_tool replay` plays the demo back in a fresh game and checks the door opens on the same frame. It does, on this savestate and on one remade from scratch. The two files differ in bytes but not in how the game plays.

**Phase 2: the backward algorithm.** Robustifying uses the backward algorithm ([Salimans and Chen, 2018](https://arxiv.org/abs/1812.03381)), as Go-Explore did. A `backward` share of episodes replay the demo up to a point, the *frontier*, and play from there. The frontier starts `backward_step` (30) frames before the door. Once half of a window of 32 episodes started there open the door, it moves 30 frames further back, until it reaches the start of the demo. The other episodes start where the task does. `make sm64-robustify` turns this on, and turns novelty and the archive off, so the reward is only the door and the clock.

Driven by the scripted walk instead of a policy, the frontier goes all the way back: 1,369 episodes, 1,369 doors, frontier 690 of 690. A trained policy is another matter. Each of these changes came from a run that failed without it:

- **Rehearsal.** The first run's frontier went from 30 frames to 180 in 165K steps, into Lakitu's speech (348 to 102 frames before the door in this demo), and stopped. Every episode on the demo then started where the policy couldn't yet win, and by 200K steps it played at random (entropy 4.3 of 4.9). Now half the episodes on the demo start anywhere between the frontier and the door, and don't count toward moving it.
- **The clock.** The second run gave every episode on the demo the whole 900 frames. Most ran all of it and paid −1, entropy fell to 0.01, and the frontier stuck at 120. The next runs started the clock at what the demo's read at that point, and the stable ones stuck at 570: that far back, the explorer's early wandering had already used a hundred frames, so those episodes had less time than a real start. Now an episode on the demo gets as long as the demo took from there plus ten seconds (`DEMO_SLACK`), and never more than the whole clock.
- **The camera.** A replay didn't keep the measured camera offset up to date, so the first stick after any replayed start, from the archive or the demo, was aimed wrong. Replays now track it the way steps do.
- **The learning rate.** At PufferLib's 0.015, the third run reached 360, then approximate KL hit 0.95 in one epoch and the door rate went to zero. At 0.005 it stays around 0.001.
- **Entropy.** The first run held entropy at `sm64.ini`'s target of 2.5 and swung between 0.6 and 4.3. `make sm64-robustify` pays a fixed 0.01 instead.

**How far it got.** The number that counts is the door rate from the real start. Checkpoints were measured with the demo and archive off, over 100 to 200 episodes each, sampling actions as training does:
- **Best:** 1.23M steps into the fourth run, a checkpoint opened the door in 68.5% of 200 episodes. The runs that opened it averaged about 680 frames on the clock. The demo took 690, and the scripted walk takes 518.
- **Not kept:** the checkpoints 100K steps before and after it opened the door 3% and 0% of the time.
- **Collapse:** every run that trained stably reached a frontier of 480 to 570, and then its door rate fell everywhere. Entropy either climbed to 4.3 with the fixed coefficient, or fell to 0.5 with a target of 2.0.
- **Bigger batches:** four times the batch (16 games, a 2,048-step minibatch) hadn't collapsed by 2.5M steps, but none of its checkpoints opened the door more than 10% of the time.

So robustifying can learn the whole route from the real start, and doesn't yet hold on to it. Muon's updates are the same size however noisy the gradient is (see *Keeping the trainer stable*), which fits a policy that learns the route and then wanders off it.

### The savestate

A new file opens with Peach's letter, Lakitu's arrival and Mario climbing out of the pipe: about 1500 frames of cutscene that waits on A for its text boxes. No episode should watch that, so it is played through **once** — two presses of Start through the menus, then A until Mario stands in the castle grounds in `ACT_IDLE` — and saved. It takes 0.4 seconds, because the game runs at about 5000 frames a second with nothing drawing it.

Every episode then begins by putting that state back, which is 8 MB of console memory plus the registers of every thread of the game. `make sm64-state` remakes it; the first run makes it on its own.

### Actions and observations

Four discrete heads: a stick direction (none, or one of sixteen **relative to the way Mario is facing**), and A, B and Z held or not. Aiming relative to Mario rather than to the pad matters because the game turns a stick direction into a heading by adding the camera's yaw to it — a policy pushing "up" would run wherever the camera drifted. The offset between the two is measured rather than looked up: whatever angle we asked for last frame, the game wrote down what it made of it in `intendedYaw`, and the difference is the camera. That only works if the stick is sent the way the game reads it: `atan2s(-stickY, stickX)` is an angle whose sine is x and whose cosine is *minus* y, so the stick goes out as `(sin θ, −cos θ)`. Sending `cos θ` hands the game a mirror image, which no measured offset can correct — the "camera" then follows Mario's facing, "ahead" freezes the stick wherever it happened to be, and every other direction thrashes it.

The observation is 40 floats: position, velocity, forward velocity, the speed just achieved, his facing and the camera relative to it, height above the floor, the floor's steepness, headroom, whether he is on a wall, in water or in the air, his action's group and flags, and the clock. Nothing in it is about the door.

An episode ends when the door starts to open, when Mario dies or warps anywhere else, or after `max_ticks` frames (900, thirty seconds).

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
