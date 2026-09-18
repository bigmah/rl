# rl

RL experiments with [PufferLib](https://github.com/PufferAI/PufferLib) 5.0, vendored as a git submodule at `vendor/PufferLib` (branch `5.0`).

Two envs live here, written as PufferLib 5.0 C envs. `make ENV=<name> train` builds one into a vecenv library of its own and trains it on the GPU the machine has, with a port of PufferLib's CUDA trainer to Rust and [wgpu](https://wgpu.rs) (`src/`): Metal on a Mac, Vulkan on Linux, DX12 on Windows.

## Platformer

`envs/platformer/` is a single-level 2D platformer written as a PufferLib C env: run right, clear the pits and spikes, touch the flag.

- `platformer.h`: level, physics, observations, rewards, rendering, and the `puf_*` functions PufferLib calls
- `platformer.c`: human play
- `platformer.ini`: env and training config, on top of PufferLib's `config/default.ini`

Rewards: progress toward the flag (1.0 for the whole run), +1 for the flag, and -0.1 for dying or running out of time. Episodes end at the flag, on death, or after 30 s. Keep the failure penalty small: at -0.5, early deaths taught some seeds to stand still.

## Super Mario 64

`envs/sm64/` is the real cartridge as an environment, and the goal is to **get one star as soon as possible**, starting where the star's course drops Mario. Which star is config.

The game is not emulated and not reimplemented. [N64Bundler](https://github.com/bigmah/n64bundler) statically recompiles `sm64.z64` into native code for the machine it is on and runs it in `n64b-run`; this env starts one of those per agent, in a process of its own, and reads Mario out of the console's memory — which is mapped into both processes, so an observation is a load rather than a request.

- `sm64.h`: the courses, how Mario gets into one, where the game keeps him, what an action does to the controller, the reward, and the `puf_*` functions PufferLib calls
- `n64b_gym.h`: starting a game, stepping it, saving and loading its state
- `sm64.c`: the env without the trainer — make a star's savestate, watch it, benchmark it, explore without a policy, replay a run
- `sm64.ini`: env and training config, on top of PufferLib's `config/default.ini`
- `stars/<star>.ini`: a star's own settings, over `sm64.ini`, for the stars that need any

```sh
cp ~/roms/sm64.z64 .  # your own dump of the cartridge (USA): the only thing this needs from you
make sm64-train               # train on the star sm64.ini names (pss-2); 8 copies of the game
make sm64-train STAR=wf-1     # or any other: Whomp's Fortress's first star
make sm64-watch STAR=wf-1     # watch that star's latest checkpoint play, in a window
make sm64-explore STAR=wf-1   # Go-Explore phase 1 with no policy: find the star, keep the fastest run
make sm64-replay STAR=wf-1    # watch the fastest run to it on file
make sm64-robustify STAR=wf-1 # Go-Explore phase 2: a policy that gets the star from the real start
make sm64-demo STAR=wf-1      # no policy at all: hold forward and jump
./build/sm64_tool courses     # the courses, and the names their stars go by
./build/sm64_tool replay watch build/sm64/pss-2/fastest/00662-25827173.demo --env.star pss-2
                              # any of the ten fastest runs to a star, kept by any run, training or exploring
SM64_TRACE=30 ./build/sm64_tool replay --env.star pss-2   # and where Mario is every thirty frames of the way
```

All it needs from you is your own dump of the cartridge, Super Mario 64 (USA): put it at `sm64.z64` in the top of the checkout, or point `SM64_ROM` at it. The first sm64 build fetches N64Bundler (a submodule at `vendor/n64bundler`), builds it, and recompiles the cartridge with it into `build/sm64/game/`: a few minutes for RT64, the renderer, and seconds for the game, once. Set `N64BUNDLER` to build against another checkout of it. No game data is copied into this repository, and everything derived from it lands in the gitignored `build/`.

Every training result in the sections below came from the PufferLib 4.0 trainer, and `sm64.ini` is still tuned for it. 5.0 trains differently enough that those settings learn far more slowly: see *Against the 4.0 port*.

### Picking a star

A star is `<course>-<number>`, numbered the way the act select and the save file number a course's stars: `wf-1` is Whomp's Fortress's first, `bob-7` Bob-omb Battlefield's hundred coins, `pss-2` Peach's Secret Slide's second. A course alone, `bob`, is any star in it. `./build/sm64_tool courses` lists the 24 courses: the fifteen with an act select, the three Bowser courses, the slide, the three cap courses, Wing Mario Over the Rainbow and the Secret Aquarium.

- **Config.** `star` in `[env]` names it (`sm64.ini` says `pss-2`). `make` targets take `STAR=`, and pass `--config envs/sm64/stars/<star>.ini` when that file exists: the slide's gives it a 1,500-frame clock and respawning, and Whomp's Fortress act 2's respawns. Any other star needs no file. `--config` is the trainer's, and works for any env: a config file over the env's own, which like a flag can only set keys the env's config has. The order is `default.ini`, the env's ini, each `--config`, then every flag.
- **Getting there.** The first time a star is asked for, the env plays through the intro, points the castle's front door at the course's painting entry, walks Mario in, picks the star's act, and saves the game once he answers the pad (`sm64_tool state` remakes it). Stars 1 to 6 of a course with an act select are entered for their own act, and the hundred coins or any star for act 1; `[env] act` enters for another. Every one of the 24 courses makes its savestate from nothing this way, including the three Mario arrives in swimming or flying (Dire, Dire Docks, the Secret Aquarium, the Tower of the Wing Cap).
- **What counts.** That star, the frame he touches it: its bit in the save file's star flags for the course goes on. Any other star isn't the goal, and taking it throws him out of the course, which is a death.
- **What it keeps.** Everything is per star: `build/sm64/<star>/` has its savestate (`start.state`), the fastest run on file (`fastest.demo`), the ten fastest (`fastest/`), runs kept by hand to seed the archive (`seeds/`), and the frontier savestates of robustifying (`states-<hash>/`). `make` puts checkpoints and logs in `checkpoints/<star>/` and `logs/<star>/`, so `latest` is that star's policy. `SM64_DATA` points the per-star folders somewhere else, for a test run that shouldn't touch the real ones.
- **The same config everywhere.** `sm64_tool` reads the config the trainer does and takes the same flags, so `./build/sm64_tool explore 8 600 --env.star wf-1` explores with the clock and settings training would use.

The castle's front door was this env's first goal, and most of what follows was learned on it. It is gone now, along with `probe door`; the sections below keep what it taught.

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

### The reward: the star, the clock and novelty

- **The star** pays 1, the most a step can pay once rewards are clipped, and ends the episode the frame Mario touches it.
- **The clock.** Every frame costs `time_penalty / max_ticks`, so running out of time costs `time_penalty` in all. The sooner the star, the less of the clock was spent.
- **Novelty.** The course is cut into cubes `novelty_cell` (500) units a side. The first time in an episode that Mario enters a cube, he is paid `novelty_episode + novelty / sqrt(n)`, where `novelty_episode` is 0.02, `novelty` is 0.05, and `n` counts the episodes, in all eight games, that have entered that cube (this one included). A cube with no floor under Mario isn't a place and pays nothing (see *Peach's Secret Slide*).
- **Deaths and warps** out of the course charge whatever is left of the clock in one step, so neither is a way to stop it early, unless `respawn` puts the game back instead (see *What a death costs*). A death in a course throws Mario out of it; health dropping under `0x100` catches the frames before that.

Nothing tells the policy where the star is or pays it for getting nearer. It sees what Mario is doing and where he is, and learns about the star by taking it.

The log reports:
- `perf`: the fraction of episodes that take the star.
- `score`: the fraction of the clock left when he took it, or 0 if he didn't.
- `frames`: frames to the star, or the whole clock without it. This is the number being minimized.
- `novelty`, `cells` and `explored`: what novelty paid in the episode, how many cubes the episode entered, and how many cubes any game has ever entered.
- `dialog` (frames in a cutscene), path length, `top_speed`, `forward_vel`, `airborne`, `ended_early` and `deaths`.
- `from_start` and `start_perf`: the share of episodes that began where the task does, and the share that began there and took the star. `start_perf / from_start` is the star rate that counts.
- `archive`, `replayed`, `star_cubes`, `ghost` and `frontier`: see Go-Explore, below.

### History: the castle door

The env's first goal was to open the castle's front door as soon as possible, from the castle grounds. It is gone, and so is its scripted walk (`probe door`), but the reward is shaped the way it is because of it.

The door is two objects, its halves, at x −76 and 77, y 803, z −3155. Walking into the left half starts `ACT_PUSHING_DOOR` (`0x1321`) and the right half `ACT_PULLING_DOOR` (`0x1320`), and either way the game warps to the castle's inside about forty frames later. Its warp nodes are what the env now points at a course's painting entry to get Mario there.

**Why novelty.** The door is 7,600 units away in a straight line, past a moat, over a bridge and through Lakitu, and it opens only for Mario walking into it. Here is how the door and the clock did on their own:
- **Random play:** in 400 episodes of random buttons nobody opened the door, and only one got as far as Lakitu.
- **Training:** 12M steps never opened it. Every episode was worth exactly −1, entropy stayed at its maximum, and the typical episode ended as far from the door as it started.

Nothing in that setup remembers where Mario has been, so exploring was just jittering the stick.

**What novelty does.** Novelty is that memory, and it knows nothing about the goal. It has two parts.

- **The fading part** (`novelty / sqrt(n)`) is for places nobody has been. The start, which every episode sees, is worth almost nothing within minutes, while a cube nobody has reached pays the full 0.05. On its own it found the door within 100K steps and then lost it. By 500K, the cubes on the way had been entered thousands of times and novelty had fallen from 0.7 an episode to 0.05. Episodes drifted back to the start, and in 5M steps the door opened only about five times, too rarely to learn the bridge, Lakitu and the door from.
- **The per-episode part** (`novelty_episode`) never fades. An episode that covers ground is always worth more than one that doesn't, so the far side of the grounds keeps being reached, and the door with it. 0.02 a cube is about what running costs in clock. A run to the door (15 cubes, the door, and the clock left over) is still worth more than wandering all episode.

Only the first entry in each episode counts, so pacing back and forth across a cube's edge earns nothing.

The scripted walk to the door entered 22 cubes. In a fresh process it returned 1.96: 0.42 for the door and the clock, 0.44 from the per-episode part, and 1.10 from the fading part. Repeated in the same process, it returned 1.64 and then 1.50.

**Entropy.** No fixed `ent_coef` worked with this reward. At 0.05 the policy sat at exactly uniform, learning nothing. At 0.005 it went out onto the bridge within 350K steps, then collapsed by 1M into a routine that barely moved. So the trainer can hold entropy at a `target_entropy` (2.5 here), raising the coefficient while the policy is more certain than that and lowering it while it is less. The setting is off by default.

**Keeping the trainer stable.** Before a door is found, the reward carries almost no signal. Muon's updates are the same size however weak the gradient is, so in that stretch the weights random-walk. The first door-and-clock run blew up at 3.4M steps: value loss went to 567,829 and entropy to zero. `weight_decay = 0.01` in `sm64.ini` keeps the weights bounded. The setting defaults to 0, which is what PufferLib uses.

An earlier version also paid for closing the straight-line distance to the door. It ended the episode in water, so that the distance couldn't lure Mario into the moat. That version opened the door in 92% of episodes after 3M steps.

Two things in the way were found by driving the game with a script. The door opens only for Mario walking into it: a dive bonks off, and a punch does nothing. And the first time Mario steps onto the bridge, Lakitu stops him to explain the camera, which is about 250 frames of dialog even with A mashed. The episode keeps going through the dialog (the policy has A), and the clock keeps running.

A scripted walk to the foot of the bridge and then to the door, mashing A through Lakitu, opened it after 518 frames with a return of 1.52 (0.42 without the novelty of a fresh process), and 49% of those frames were dialog.

### When a star is won, and which

A star is his the frame he touches it. A star that a box or a boss lets out, or that the slide awards for a time, is *spawned* first: it stops time, plays its cutscene for about a hundred frames while Mario stands frozen in whatever he was doing, and then lands where he can take it. For a while the spawn was the goal, which took the dead frames out of every success; the runs below up to the 648-frame slide were measured that way. But a star that has spawned is not yet won, and a policy that stops there has not learned to take it, so **the episode ends the frame he touches the star**. `sm64_tool replay` still says the frame a star spawned, from the game's time-stop flags in a halfword at `0x8033D482`, found by snapshotting the console's memory before and during the freeze and looking for the one word that goes from 0 to something and stays there — it goes to `0x4A`, which is time stop enabled (2), Mario and doors (8) and active (0x40), and a spawning star is what sets Mario-and-doors in a course (dialog sets a different bit; the doors that set it are in the castle). `./build/sm64_tool trim [file...]` replays runs, cuts them where the star is now taken, and writes them back — renaming a run in a fastest-runs folder for its new length — and `replay` says when a run wants trimming.

Which star it was used to be anyone's guess: the goal was the star count going up. It is now the star's own bit in the save file, `gSaveBuffer.files[0][0].courseStars` at `0x8020770C`, a byte a course and a bit a star, found the same way: taking the slide's star turns on bit 1 of the byte for course 19, the frame the count goes up. That settled something about the slide. **Every slide run on file takes its second star, the one for reaching the bottom inside 21 seconds** — not the box. The HUD timer starts when the sliding does (about frame 83 from the savestate) and stops when the star spawns: 564 frames, 18.8 seconds, on the 786-frame run. Set to 23 seconds just before the bottom, the same route spawns no star at all. So a run from the top of the slide that is slower than 21 seconds gets nothing, however clean its ending.

### What a death costs

A death, or any warp out of the course, ends the episode and pays whatever is left of the clock at once, so stopping the clock early is never a way to lose less. That's right where Mario has to go out of his way to die, and wrong in a fortress in the sky: **70% of episodes ended in a death**, most inside the first half of the clock, and each one entered 12 new cubes where a full episode enters 60. Dying in a course throws Mario out of it — back to the castle, one life gone — so the episode really is over as far as the game is concerned.

Paying less for a death would have made dying the cheapest way to play, since every frame costs clock. `respawn = 1` keeps the clock instead: the savestate goes back, the episode carries on from where the death left it, and a death costs the time it wasted and the ground it gave up. The archive stays honest because the game is that savestate to the bit again, so the trail it keeps starts empty and still replays. Over 21 minutes in Whomp's Fortress:

| | respawn = 0 | respawn = 1 |
|---|---|---|
| episode length (of 450 steps) | 206 | 450 |
| ended in a death | 69% | 0% |
| cubes entered per episode | 12 | 60 |
| episode return | −0.61 | +0.55 |
| distance per episode | 3 900 | 22 000 |
| agent steps/s | 1 045 | 1 549 |

Returns go positive because novelty earns more than the clock costs, and deaths per episode climb from 0.1 to about 6 as the policy gets bolder: falling becomes a cost to weigh rather than the end. It still hasn't reached a star.

### Peach's Secret Slide

`pss-2` is the easiest star in the game to state: the slide drops Mario 6,000 units, the star appears at the bottom of it, and gravity does most of the work. Nothing has to be climbed or fought, and the only way to lose is to go over the side — or, it turns out, to be slow: it is the star for reaching the bottom inside 21 seconds (see *When a star is won, and which*).

```sh
make sm64-train STAR=pss-2   # stars/pss-2.ini: a 1,500-frame clock and respawn = 1
```

It is a secret course, so there is no act select: the game keeps its act at 0, nothing is written over it, and A is not pressed on the way in. Everything else — the clock, novelty, the archive, the demo — is the same as any other course.

**It is far easier to stumble into.** Go-Explore's first phase, random buttons from the archive, found the star in 60 seconds and four of them in two minutes. The castle door took 200 seconds for its first, and Whomp's Fortress never reached a star at all. `./build/sm64_tool replay` lands the fastest of them on the same frame it was found on.

Eight minutes of it found 40 stars and took the fastest from 1,326 frames to 1,246, so random play from the archive does shorten the way, but slowly. Since then the explorer plays a course differently from the grounds (`make sm64-explore`): A is held and let go like B and Z rather than drawn fresh every step — that was for Lakitu, and a jump every step is the slowest way down a slide — and half the time the stick is redrawn it goes straight ahead, because a course is covered by going somewhere. That version has not been timed against the old one.

**Twenty minutes of training** — the star, the clock and novelty, with the archive on and `respawn = 1` — is 3.81M steps at 3,900 a second:

| steps | cubes entered per episode | distance | episode return | cubes explored | stars |
|---|---|---|---|---|---|
| 0.5M | 33 | 16 800 | +0.03 | 1 959 | 0 |
| 1.4M | 18 | 8 500 | −0.52 | 3 066 | 0 |
| 2.4M | 35 | 17 300 | −0.13 | 3 552 | 0 |
| 3.3M | 54 | 27 300 | +0.29 | 4 040 | 1 |
| 3.8M | 65 | 32 600 | +0.51 | 4 206 | 0 |

One star in 2,653 episodes, still climbing at the end. It started from a cube in the archive rather than from the top of the slide, and `start_perf` never left 0. It is a faster star than the explorer's, though — 1,378 frames from the savestate against 1,940, and it replays.

Both are longer than the 900-frame clock, which the archive's replay doesn't spend, so nothing had gone from the top of the slide to the star inside one episode. **The clock was the problem.** `SM64_TRACE=30 ./build/sm64_tool replay` prints where a run has Mario every thirty frames, and the 1,326-frame star looks like this: 170 frames walking about the top, then 960 frames of sliding from y 6,144 down to y −4,500, the slide winding through six turns on the way, then 200 frames at the bottom before the star. Holding forward does not cover the slide in 600 frames; it covers *half* of it, and flies off the side at the fourth turn. At the pace an explorer slides, the star is 1,100 frames from the top, and the 900-frame clock could never fit it. So this course gets a 1,500-frame clock (`stars/pss-2.ini`), and every frame of it still costs, so faster is still better.

**Novelty was paying for falling out of the world.** A cube is 500 units, so a fall through empty space enters a fresh one every 500 units down, and each pays `novelty_episode` again in every episode. `sm64_tool probe forward` shows it: at frame 660, off the side of the slide at y −1,675 with no floor under him and nothing below but the death plane, the step paid +0.0678, which is a cube nobody had entered. Worse, those cubes went into the archive, and few runs fall down the same column, so they were among the rarest — which is what it draws first, restarting episodes midway through a fall. In a course whose only way to lose is going over the side, that is paying to lose, twice.

`mario->floor` is null exactly there, so **a cube reached with no floor under Mario is not a place**: it pays nothing and is not archived. The same twenty minutes again:

| steps | cubes entered per episode | distance | episode return | cubes explored | stars |
|---|---|---|---|---|---|
| 0.5M | 20 | 10 300 | −0.34 | 1 699 | 0 |
| 1.4M | 23 | 10 600 | −0.37 | 3 026 | 0 |
| 2.4M | 47 | 21 600 | +0.21 | 4 116 | 0 |
| 3.3M | 62 | 29 300 | +0.48 | 4 655 | 0 |
| 3.8M | 65 | 29 300 | +0.51 | 4 789 | 0 |

No star this time, which says nothing on its own: at one star in 2,653 episodes, one and none in 8,000 episodes each are the same rate. What did change is that every cube is now somewhere Mario could stand, and there are 14% more of them than the run before explored counting the empty air, reached sooner — 47 an episode at 2.4M steps where the run before was at 35.

So exploring the slide is not the thing in the way. Going down it inside one clock is.

Showing the policy the game helps here, which it did not in Whomp's Fortress: at equal steps a picture policy covers about twice the ground, enough to pay for the half of the speed it costs. See the picture, below. It has not reached a star either.

### Go-Explore

Novelty alone swam the moat for 10M steps. So `sm64.h` also runs [Go-Explore](https://arxiv.org/abs/1901.10995). Its first phase explores: it remembers places and goes back to them before exploring further. Its second phase robustifies: it trains a policy to do reliably, from the real start, what exploring only managed once.

**Going back is replaying inputs.** The game is deterministic, so the pad inputs from the savestate *are* the place, at a few kilobytes against a savestate's 8 MB. The archive keeps, for every cube, the shortest run of inputs that reached it. With `go_explore` on, that share of episodes picks a cube, weighted to the ones fewest episodes have entered, and replays its run before the clock starts.

**An archive that starts with the runs on file.** The archive is the process's, so every run began with none: thirty minutes on the slide spent the first nine finding the shortcut the run before had found, and what two runs found never met. With `go_explore_seed = 1` the first game to start plays every run on file once — the demo, the folder of the fastest and `seeds/` — and offers each cube on the way to the archive, then credits those cubes as on the way to the goal. A run stops being offered the step it reaches the goal, as an episode does, or the cube after it would start episodes with the star already won. On the slide that is 174 cubes from 11 runs in a few seconds, and an hour begun that way (`--env.go-explore 0.9 --env.go-explore-star 0.75`, so nine episodes in ten start from the archive and three quarters of those on a way to the star) touched the star in 9 to 12% of episodes where the hour before managed 1 to 2%. It moved the record from 770 frames to 768. The archive keeps the shortest way to a *place*, and the splice it was seeded for — one run's fast top, another's better jump — needed the jump made from the first run's speed and line, which the policy tried and did not land (`demos/peach-slide/README.md`).

**Racing the archive.** The clock pays for speed only at the star and little: thirty frames off a 770-frame star is 0.02. What a run gets faster by is the archive, which keeps the shortest way into every cube, and that way is a ghost to race. With `ghost` on, the first time in an episode Mario enters a cube that has been on a way to the star, on the ground and sooner than the archive's way into it, he is paid `ghost` a frame of the difference, 0.1 at most. The archive then keeps his way, so the same again pays nothing and only a faster run is ever paid; a run fifty frames up on the ghost is paid at every cube for as long as it stays ahead. Only on the ground, because a jump on its way over the side passes through the cubes of the track below sooner than anything that slid there. At 0.001 a frame it paid 0.0002 an episode and changed nothing; at 0.02 an hour on the slide (`--env.go-explore-seed 1 --env.ghost 0.02`) touched the star in a fifth of its episodes by the end and took the record from 768 frames to 750, with every run in the fastest folder under the 768 it began with. The 750 is the 786's top and speed down the upper track, a new jump off it, and a faster ending (`demos/peach-slide/README.md`). What it does not do is make a landing happen: started on the 786 a second before its jump, the policy beat the ghost in none of 1,500 episodes, and no pay for beating it can teach a jump that never lands.

Scoring from the first minute broke the entropy target. `ent_coef` follows the sum of the gap between entropy and `target_entropy`, and with stars arriving at once entropy answered it late and with hysteresis: it cycled between 0.3 and 4.6 every 700K steps, at a quarter of the rate as well. `ent_coef_damping = 0.5` adds the gap itself to the coefficient, a factor of e for every two of entropy off target, and the same run held 2.1 to 2.9. It is 0, the sum alone, everywhere else.

**Phase 1 without a policy.** `make sm64-explore` is the first phase as the paper ran it: every episode starts from the archive and plays at random. Any exploring run that takes the star writes its inputs to `build/sm64/<star>/fastest.demo`, if nothing on file was faster. On the castle door it found the door in 200 seconds, and in ten minutes it had opened it 132 times, the fastest in 690 frames (the scripted walk takes 518).

It found nothing until episodes were long enough to get through Lakitu. His speech starts and ends in the same cube on the bridge, so the archive can't keep a place partway through it. An episode has to get from the cube before the bridge to the one after it, speech and all. With 300-frame episodes and A held a few steps at a time, 2,300 episodes never got closer to the door than 1,088 units. With 900 frames and A drawn every step, speeches started finishing within a minute.

`./build/sm64_tool replay` plays the demo back in a fresh game and checks the star is taken on the same frame. The door's did, on its savestate and on one remade from scratch. The two files differ in bytes but not in how the game plays.

**Phase 2: the backward algorithm.** Robustifying uses the backward algorithm ([Salimans and Chen, 2018](https://arxiv.org/abs/1812.03381)), as Go-Explore did. A `backward` share of episodes replay the demo up to a point, the *frontier*, and play from there. The frontier starts `backward_step` (30) frames before the goal. Once half of a window of 32 episodes started there reach it, it moves 30 frames further back, until it reaches the start of the demo. The other episodes start where the task does. `make sm64-robustify` turns this on, and turns novelty and the archive off, so the reward is only the star and the clock. What follows was learned on the door.

Driven by the scripted walk instead of a policy, the frontier goes all the way back: 1,369 episodes, 1,369 doors, frontier 690 of 690. A trained policy is another matter. Each of these changes came from a run that failed without it:

- **Rehearsal.** The first run's frontier went from 30 frames to 180 in 165K steps, into Lakitu's speech (348 to 102 frames before the door in this demo), and stopped. Every episode on the demo then started where the policy couldn't yet win, and by 200K steps it played at random (entropy 4.3 of 4.9). Now half the episodes on the demo start anywhere between the frontier and the door, and don't count toward moving it.
- **The clock.** The second run gave every episode on the demo the whole 900 frames. Most ran all of it and paid −1, entropy fell to 0.01, and the frontier stuck at 120. The next runs started the clock at what the demo's read at that point, and the stable ones stuck at 570: that far back, the explorer's early wandering had already used a hundred frames, so those episodes had less time than a real start. Now an episode on the demo gets as long as the demo took from there plus ten seconds (`DEMO_SLACK`), and never more than the whole clock.
- **The frontier is a savestate.** Getting to the frontier meant replaying the demo up to it, which costs as many frames as the demo has before that point, and the eight games step in lockstep, so one game replaying holds the other seven. On the slide that was 1,200 frames of demo for an episode 30 frames from the star that itself lasts 330, and training ran at 1,000 steps a second instead of 5,000. So the first episode to reach a point on the demo saves the game there, in a folder named for the demo (`build/sm64/pss-2/states-8d5b4e43/01204.state`), and every episode after loads it: 8 MB read in place of a second of play. Starts are on the frontier's grid of `backward_step` frames — the frontier itself, or for a rehearsal any frontier already passed — so there are as many states as frontiers, about 40 for the slide's demo. The game is deterministic, so which game saved a state makes no difference to what is in it.
- **A run can carry on from another.** The frontier is not in a checkpoint, so a run that loads one (`load_model_path`) used to start its frontier at `backward_step` again. `backward_start` puts it where the last run's got to, in frames before the door; 0 is `backward_step`.
- **A robustifying run feeds the demo.** Only exploring runs used to offer a door to the demo file. Now a robustifying run does too, so a policy that gets there faster than the demo it started on — from the real start, or from a frontier with the replayed part counted — leaves a faster demo for the next run to work back along. The run under way keeps the demo it read when it started.
- **The camera.** A replay didn't keep the measured camera offset up to date, so the first stick after any replayed start, from the archive or the demo, was aimed wrong. Replays now track it the way steps do.
- **The learning rate.** At PufferLib's 0.015, the third run reached 360, then approximate KL hit 0.95 in one epoch and the door rate went to zero. At 0.005 it stays around 0.001.
- **Entropy.** The first run held entropy at `sm64.ini`'s target of 2.5 and swung between 0.6 and 4.3. `make sm64-robustify` pays a fixed price instead (0.01 then, 0.005 since the slide).

**How far it got.** The number that counts is the door rate from the real start. Checkpoints were measured with the demo and archive off, over 100 to 200 episodes each, sampling actions as training does:
- **Best:** 1.23M steps into the fourth run, a checkpoint opened the door in 68.5% of 200 episodes. The runs that opened it averaged about 680 frames on the clock. The demo took 690, and the scripted walk takes 518.
- **Not kept:** the checkpoints 100K steps before and after it opened the door 3% and 0% of the time.
- **Collapse:** every run that trained stably reached a frontier of 480 to 570, and then its door rate fell everywhere. Entropy either climbed to 4.3 with the fixed coefficient, or fell to 0.5 with a target of 2.0.
- **Bigger batches:** four times the batch (16 games, a 2,048-step minibatch) hadn't collapsed by 2.5M steps, but none of its checkpoints opened the door more than 10% of the time.

So robustifying can learn the whole route from the real start, and doesn't yet hold on to it. Muon's updates are the same size however noisy the gradient is (see *Keeping the trainer stable*), which fits a policy that learns the route and then wanders off it.

### The savestate

A new file opens with Peach's letter, Lakitu's arrival and Mario climbing out of the pipe: about 1500 frames of cutscene that waits on A for its text boxes. No episode should watch that, so it is played through **once** a star — two presses of Start through the menus, then A until Mario stands in the castle grounds in `ACT_IDLE`, then through the front door into the star's course (see *Picking a star*) — and saved in `build/sm64/<star>/start.state`. It takes 0.4 seconds, because the game runs at about 5000 frames a second with nothing drawing it.

A course holds Mario under its opening camera for a couple of seconds, and a state saved then loads into a game where he never moves at all, so B is pressed every half second until a state, put back, has a Mario who answers the pad: A and the stick change what he is doing or move him. Every episode then begins by putting that state back, which is 8 MB of console memory plus the registers of every thread of the game. `make sm64-state STAR=...` remakes it; the first run makes it on its own.

### Actions and observations

Four discrete heads: a stick direction (none, or one of sixteen **relative to the way Mario is facing**), and A, B and Z held or not. Aiming relative to Mario rather than to the pad matters because the game turns a stick direction into a heading by adding the camera's yaw to it — a policy pushing "up" would run wherever the camera drifted. The offset between the two is measured rather than looked up: whatever angle we asked for last frame, the game wrote down what it made of it in `intendedYaw`, and the difference is the camera. That only works if the stick is sent the way the game reads it: `atan2s(-stickY, stickX)` is an angle whose sine is x and whose cosine is *minus* y, so the stick goes out as `(sin θ, −cos θ)`. Sending `cos θ` hands the game a mirror image, which no measured offset can correct — the "camera" then follows Mario's facing, "ahead" freezes the stick wherever it happened to be, and every other direction thrashes it.

The observation is 40 floats: position, velocity, forward velocity, the speed just achieved, his facing and the camera relative to it, height above the floor, the floor's steepness, headroom, whether he is on a wall, in water or in the air, his action's group and flags, and the clock. Nothing in it is about the door.

### The picture

Those 40 numbers are about Mario and nothing else. They don't show the moat, the bridge, a Bob-omb or a star. A player sees all of that, so the policy can also be shown the game:

```sh
make sm64-picture   # train with an 80x60 picture of the game in the observation
make sm64-watch ARGS="--env.picture-width 80 --env.picture-height 60"
```

`picture_width` and `picture_height` in `sm64.ini` put the game's frame, shrunk to that size by averaging, after the 40 numbers as RGB in [0, 1]. The trainer sees `picture_width` in the env's config and puts the picture through the Nature DQN's three convolutions before the MinGRU, with the 40 numbers alongside. `state = 0` zeroes those numbers but the clock, so the policy sees the picture and how much time is left. The reward still reads the game's memory either way.

Nothing about the picture is specific to Super Mario 64: `n64gym_picture` reads the console's video interface, not the game's variables. With `--picture`, N64Bundler's RT64 renders each frame back into the framebuffer the game drew it into, at the console's own 320×240, as the RDP did. After each step the host reads the frame the video interface will show next (its origin, width, pixel format and gamma) into the shared memory beside the console's RAM. Headless, RT64 renders only that, on a window that's never shown. That's patches `0022`-`0023` in N64Bundler's series.

What it costs, with 8 games on an M4 Pro:

| | agent steps/s |
|---|---|
| headless, no picture | 8 900 |
| picture, every frame drawn | 1 800 |
| picture, last frame of each step drawn | 2 550 |
| training with a picture policy (80×60) | 1 400–2 100 |

Most of a picture step is RT64: turning the display list into GPU work, then waiting for the GPU. Converting the picture costs next to nothing. A step is two frames and only the second is ever seen, so the env draws only the last frame of each step, and nothing at all while it replays a way to a starting cube. That's safe here because Super Mario 64 draws every frame from nothing. It would be wrong for a game that builds a frame out of the one before it.

Drawing doesn't change how the game plays: `./build/sm64_tool replay --env.picture-width 80 --env.picture-height 60` takes the star on the same frame as it does headless (the door's opened on the same frame, 690), and `./build/sm64_tool bench 8 1000 --env.picture-width 80 --env.picture-height 60` measures the speed.

A state loaded from a file holds framebuffers in whatever condition the game that saved it left them. So after a load there's no picture (the observation's picture is black) until a frame has been drawn.

**What it's worth so far: nothing measurable.** Two runs of Whomp's Fortress act 2, same settings but for the picture, compared at 1.9M steps:

| | with the picture | without |
|---|---|---|
| cubes entered per episode | 59.7 | 62.4 |
| episode return | 0.50 | 0.57 |
| cubes explored | 2 479 | 2 646 |
| agent steps/s | 1 549 | 3 203 |
| stars | 0 | 0 |

Half the speed and no gain. Neither run is long — 2M steps is early for a convolutional encoder learning from a reward this sparse — and both had Mario's 40 numbers in the observation as well, so the picture was only ever extra. The sharper test is `state = 0`, pixels against numbers. The deeper point is that novelty and the archive are what explore here, and both are computed from Mario's position: until *they* come from the picture, what the policy sees can't change where episodes go.

**On Peach's Secret Slide it is worth something.** Twenty minutes each, the same settings but for the picture, and the 40 numbers in both:

| | with the picture | without |
|---|---|---|
| steps in twenty minutes | 1.91M | 3.81M |
| agent steps/s | 1 840 | 3 800 |
| cubes entered per episode, at 1.2M steps | 53.9 | 22.7 (at 1.4M) |
| cubes entered per episode, at the end | 73.7 | 65.4 |
| distance per episode, at the end | 32 900 | 29 300 |
| cubes explored | 4 440 | 4 789 |
| stars | 0 | 0 |

At equal steps the picture policy covers about twice the ground, which is enough to pay for half the throughput: it ends ahead on cubes an episode having taken half as many steps, and its entropy is down to 1.7 rather than 2.8, so it is committing to routes rather than jittering. Cubes explored is the one number where it is behind, and that one counts the whole process's exploring, which is half as many episodes.

A slide is where that would be expected to show. The 40 numbers say how fast Mario is going and how steep the floor is, but not that the floor runs out three feet to his left, and on this course that is the only thing there is to know.

An episode ends when the door starts to open, when Mario dies or warps anywhere else, or after `max_ticks` frames (900, thirty seconds).

### Robustifying the slide

`make sm64-robustify STAR=pss-2` is Go-Explore's second phase on the slide, with the 40 numbers and nothing else in the observation: work back along the fastest star on file, with only the star and the clock paying. Getting it to learn at all took five changes, each from a run that did not:

- **The clock**, above: 1,500 frames, because the slide is 1,100 long.
- **Normalized advantages** (`norm_adv`, see *Against the 4.0 port*): under 5.0's update as it is, 1.5M steps learned nothing and entropy drifted to uniform.
- **The star is won when it spawns** (see *When a star is won, and which*; it is the touch again now): every star on file ended with a kick or a ground pound on the box at the bottom, then a hundred frames frozen while the star came down. Ending the episode at the spawn takes the dead frames out of every success and puts the reward on the move that earned it, fifty decisions sooner.
- **The frontier is a savestate** (see *Go-Explore*): replaying 1,200 frames of demo to start a 330-frame episode ran training at 1,000 steps a second; loading the game there runs it at 5,000.
- **A finer frontier.** Thirty frames worked for a door Mario walks into. The slide's ending is a speed trick — into the room at 77 units a frame, then a long jump that lands on the box two decisions later — and fifteen decisions of that, exactly, is too much to ask of a policy's first frontier. With 30-frame steps two runs spent 2.4M steps each stuck one frontier into the room, entropy climbing back toward uniform as the failures piled up; and each time the policy found a faster ending, the demo changed under the next run and the room had to be learned again. `--env.backward-step 10` asks for the last five decisions first, then ten.

**The demo got faster as the policy did.** A robustifying run offers its stars to the demo file too, and the slide's went 1,326 (the explorer's, when this started) → 1,246 (eight minutes more exploring) → 1,184 (a policy beating the demo's ending from a frontier) → 1,080 (the same run, trimmed to the spawn) → 954: the policy's own ending, a long jump straight into the box from the bottom of the slide, where the explorer had flailed for 200 frames. The 780 frames of slide in it are the explorer's dive-slide at about 100 units a frame, which is close to as fast as the game slides.

**How far it got: the ending, not the slide.** Five runs of 1.2M to 5.9M steps, at about 5,000 steps a second, each begun where the one before left off in some way (`backward_start`, `load_model_path`) or fresh:

| run | frontier step | moves at | reached, in frames before the star | from the top, 50 episodes |
|---|---|---|---|---|
| 1, star counted at the touch | 30 | 50% | 180 of 1,226 (the freeze and the room) in 3.9M steps | 0 |
| 2, star at the spawn | 30 | 50% | 210 of 1,080 in 5.9M; 2.4M of them stuck at 180 | 0 |
| 3, carried on from 2 | 30 | 30% | 90 of 954, nothing in 1.9M | 0 |
| 4, fresh | 10 | 40% | 90 of 954 in 2.6M | 0 |
| 5, carried on from 4 | 30 | 20% | 90, nothing in 1.2M | 0 |

Every run stopped where the demo's ending begins. The policy learns the last five, ten, fifteen decisions — the frontier moves through them — and then stalls once the frontier is far enough up the slide that its own sliding changes the state it arrives in, because the ending only works from the state the demo arrived in. At each stall entropy climbed back from 2.3 toward 4 as the failures piled up, which is the door's collapse again. The slide itself was never the problem: from any frontier state, holding the stick ahead slides into the room.

So what the slide wants next is a demo with a robust ending — stop in the room, get under the box, jump — the way the door's demo ends with a walk into a door. The door had a scripted walk; the box wants the same, or an explorer that stops. Given that, the frontier should run up the slide at a window an episode, since sliding generalizes and the box does not.

What the runs leave behind is worth watching anyway: `make sm64-replay STAR=pss-2` plays the fastest on file, and the runs in `demos/peach-slide/` play with `./build/sm64_tool replay watch <file> --env.star pss-2`. The 936-frame star of that time was 31 seconds from the savestate to the bottom, where the explorer's took 1,326 to the touch; watched, a replay plays five seconds more, so the star comes down.

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

## The trainer

PufferLib 5.0 trains with one CUDA program (`src/pufferl.cu` and `src/algo.cu`), which only an NVIDIA card runs. `src/` is that trainer in Rust on [wgpu](https://wgpu.rs), which is Metal, Vulkan or DX12 depending on the machine, with every kernel written once in WGSL (`src/kernels/`):

```sh
cargo build --release
./target/release/pufferl train platformer [--train.total-timesteps 20_000_000 ...]
./target/release/pufferl eval platformer [latest | path/to/checkpoint.bin] [--headless]
```

`WGPU_BACKEND` and `WGPU_ADAPTER_NAME` pick a GPU, as they do for any wgpu program. What it trains is 5.0's:
- **Policy:** PufferLib's default. A bias-free Linear encoder, a 4-layer MinGRU (hidden size 128), and one decoder matrix whose last row is the value head. For an env whose config names a `picture_width` and `picture_height`, the last `width × height × 3` observation values are a picture, and three convolutions go in front of the encoder.
- **PPO as 5.0 has it:** each minibatch computes its advantages from its own values before the update (GAE, or V-trace with `vtrace = 1`) and doesn't normalize them. Minibatches are whole agent rows taken in order, with no prioritized replay. The MinGRU's state carries over from one horizon to the next and is cleared where an episode ends, both when acting and when training.
- **Muon**, on the clipped gradient.

`vecenv.c` is 5.0's CPU vecenv without the CUDA around it. `build.sh` compiles it around an env into `build/vecenv_<env>.dylib`, and the trainer loads that (`src/vecenv.rs`): the env steps on CPU threads, the policy on the GPU.

Configs and flags work as they do in 5.0: `vendor/PufferLib/config/default.ini`, then `envs/<env>/<env>.ini`, then `--section.key=value` flags, where a space works as well as `=`. Only keys a config already has can be set. One addition: `--config FILE` loads another config over the env's, before any flag, with the same rule — a variant of the env, like one of sm64's stars (`envs/sm64/stars/pss-2.ini`).

Training writes checkpoints to `checkpoints/<env>/<run_id>/` and, at the end, `logs/<env>/<run_id>.ini`: the config and a downsampled `[metrics]` section. With `eval_episodes` above 0, it finishes by playing the final policy in fresh episodes until that many have ended. `eval` shows one env in a window, or with `--headless` plays `eval_episodes` episodes and prints the score. `latest` is the env's most trained checkpoint.

Checkpoints are in 5.0's format: flat float32 weights, in the CUDA trainer's order and alignment. PufferLib's own CPU eval (`src/puffercpu.c`, built around `platformer.h`) loads every weight of a platformer checkpoint in this format and reaches the flag in 300 of 300 episodes. Checkpoints from the 4.0 port (numpy `.npz` data under the same `.bin` names) still load, and a platformer one plays to the flag in all of 2,048 episodes.

Not ported: continuous actions, action masks, multiple GPUs, self-play and sweeps. Rollouts are always synchronous: 5.0's default `async = 1` collects the next rollout with a one-epoch-stale copy of the policy while the learner trains, and this ignores it.

**There is no autograd, as there is none in 5.0.** wgpu runs compute shaders and nothing else: no BLAS, no tensors, no gradients. But `algo.cu` has none either (it is about 35 CUDA kernels with their backward passes written out, and cuBLAS for the products), so this is the same shape: a kernel for each of those, and one matmul kernel in place of cuBLAS.
- `matmul.wgsl`: a thread computes a 4 by 4 block of the result, so each step along the inner dimension is eight reads and sixteen products. A GPU thread is slow alone, about a tenth of a microsecond a step, so a long inner dimension is cut into chunks that run side by side and are summed after; that alone made the weight gradients, whose inner dimension is the 8,192 rows of a minibatch, five times faster. Matrices are views (an offset and two strides), so a transpose or a block of the flat weights costs nothing, and matrices of one shape multiply as a batch.
- `scan_forward.wgsl` and `scan_backward.wgsl` are `algo.cu`'s MinGRU kernels: a GPU thread per unit steps through time, and the second does the backward pass. `mingru_step.wgsl` is the one step of it that acting takes.
- `ppo_loss.wgsl` is `ppo_loss_compute`: the loss of each step and, in the same walk, its gradient with respect to the step's logits and value, which is where the backward pass starts. `ppo_pre.wgsl` and `advantage.wgsl` come before it, as `cache_imp_and_v` and `puff_advantage` do.
- Muon is `muon_momentum.wgsl`, five Newton-Schulz steps of three products and a polynomial, and `muon_apply.wgsl`. The MinGRU's layers are one shape, so they are orthogonalized as one batch.
- The picture's three convolutions are the matmul too: `im2col.wgsl` lays every place the kernel goes out as a row, so the convolution is a product with the weights and both its gradients are products, and `col2im.wgsl` gathers the way back. Written as loops over the picture instead, they were five times slower: a minibatch of 512 pictures was 88 ms and is 14.

**Everything is built once.** Every shape is known when training starts, so every dispatch of an epoch is too: its pipeline, its buffers, and its parameters in a uniform buffer of its own. A step of acting is one list of dispatches and a minibatch is another, and running one is encoding the list again, which takes the CPU 0.2 ms for a minibatch's 146. The numbers that do change (the step of the rollout, the learning rate, the entropy coefficient) live in two small uniforms the kernels share. Every weight is in one buffer in a checkpoint's order and padding, so a checkpoint is that buffer written out; gradients and momentum are buffers laid out the same way, so the global gradient norm is one sum. Nothing is read back while training but the actions, once a step, and the eight loss statistics, once an epoch.

**It is tested.** `make test`:
- `tests/parity.rs`: a policy, a minibatch, and golden numbers for them in `tests/golden/`: the loss statistics, every gradient, and the weights after one and two steps of Muon. Three cases: 5.0 as it is; everything a config can turn on (V-trace, `norm_adv`, weight decay, four action heads); and a picture through the convolutions. The numbers came from an MLX port of 5.0's trainer that this repository had before `src/`, whose gradients were autograd's rather than written out by hand, and the trainer agrees with them to four or five figures, which is what two orders of summing float32 leave. That port is gone, so a change to what the trainer computes (a new loss term, a new layer) has nothing left to check its backward pass against.
- `tests/rollout.rs`: acting and training are different kernels over the same policy, so a rollout trained on before the weights move has to come out exactly on-policy (KL 0, importance 1), across episode ends and carried state. And 20,000 agents shown the same observation have to take each action as often as the sampler says it drew it.
- `tests/ops.rs`: the matmul and the sums against the CPU, over every way the trainer uses them.

**How fast**, on an M4 Pro: platformer trains at 445K steps a second with 1,024 agents, and 20M steps take 45 seconds and reach the flag in 99.7% of 2,000 episodes. sm64 with 8 games takes 30.0 s for 120K steps.

Where an env is fast, as platformer is, the matmul is what takes the time: this one does 1.5 to 2 TFLOPS, with no vendor's library under it, and a minibatch of 8,192 is 8.0 ms forward and back and 1.8 ms of Muon. Where the env is slow, as eight copies of a Nintendo 64 are, the trainer is waiting for it: what matters there is the way to the GPU and back, once a step, and a step of acting for eight agents is 0.22 ms against the 0.18 ms that a round trip with nothing to do costs. With an 80 by 60 picture it is 0.8 ms, and the games step 1,500 times a second either way. `make bench` measures all of this on the GPU it is run on.

**What is and isn't portable.** The trainer is: it builds for Linux and Windows as it stands (`cargo check --target`), and nothing in it knows what GPU it has. It has been run on Metal, and on Vulkan on arm64 Linux, where `make test` passes against the same golden numbers. `build.sh` knows macOS and Linux. The platformer is plain C and raylib. The sm64 env is POSIX (`shm_open`, `posix_spawn`, a socket pair), and the game under it is whatever N64Bundler recompiles it to: arm64 on a Mac, and the machine's own architecture on Linux. On arm64 Linux, from a fresh clone, the 674-frame slide record replays to the same frame it does on a Mac.

Observations are uploaded as floats, so an env with `unsigned char` observations is converted on the CPU first; both envs here are float.

### Against the 4.0 port

These were measured with the MLX port of 5.0 that came before `src/`, which `tests/golden` still holds the trainer to. On an M4 Pro, platformer, 10M steps, perf the fraction of episodes reaching the flag:

| | 4.0 | 5.0 |
|---|---|---|
| perf at about 4M steps | 0.98 | 0.98 |
| perf at about 9M steps | 0.998 | 0.993–0.998 |

That needed two config changes, both back to 5.0's defaults:
- **`ent_coef`**: without normalized advantages, platformer's old 0.01 outweighed its advantages. Entropy stayed at 2.1 and perf was 0.92 at 10M steps. At 5.0's 0.001 it learns as fast as 4.0 did.
- **`eval_episodes`**: 5.0's eval restarts every env and counts the first episodes to end. Those skew toward early deaths, so with 1,024 agents, 100 episodes read 0.77 for a policy at 0.93. 5.0's default of 10,000 doesn't skew.

**sm64 has not been retuned for 5.0.** With 8 games, the 4.0 trainer and 5.0 run at the same speed, 3,500–4,200 steps a second, but `sm64.ini` learns far more slowly on 5.0. Entropy after 150K steps, with 2.5 as the target (4.9 is uniform):

| | entropy |
|---|---|
| 4.0 | 1.8 |
| 5.0, `sm64.ini` as it is | 4.1, still, at 300K steps |
| 5.0, `vf_coef` 0.5 or 0.1 | 4.3–4.4 |
| 5.0, fixed `ent_coef` 0.001 | 3.9 |
| 5.0, fixed `ent_coef` 0.001 and `vf_coef` 0.1 | 3.3 |
| 5.0 with advantages normalized as 4.0 did, and nothing else changed | 1.8 |

So the change that matters for sm64 is advantage normalization. Its rewards are small, and Muon's update is the same size whatever the gradient, so without normalization the policy term barely steers it.

Robustifying the slide made that a wall rather than a slope. With only the star and the clock paying, and the star a hundred frames or more off, advantages run a tenth the size of the value error. Three 1.5M-step runs, the same but for these:

| | entropy at 1.5M steps | frontier |
|---|---|---|
| 5.0 as it is (`vf_coef` 2.0) | 4.90, up from 4.83 | 150 the whole run |
| `vf_coef` 0.5 (stopped at 0.7M, the same) | 4.83–4.89 | 150 |
| advantages normalized | 4.4–4.7 | 180 |

The first two never learned anything: entropy drifted *up* toward the 4.91 of a uniform policy, and the frontier sat where random play alone gets it. So `norm_adv = 1` in `[train]` normalizes each minibatch's advantages as 4.0 did. It is off by default and in `sm64.ini`, so the trainer stays 5.0's unless a run asks; `make sm64-robustify` asks.

## Setup

Rust ([rustup](https://rustup.rs)) and a C compiler, then your own dump of Super Mario 64 (USA) for sm64. Nothing else has to be fetched or built by hand: every target builds the trainer, fetches the submodules it needs, and builds N64Bundler and recompiles the game the first time it is run.

On a Mac, OpenMP too, which Apple's clang doesn't ship, and for sm64 cmake, ninja and Xcode, whose Metal toolchain compiles the renderer's shaders (Xcode 26 downloads it separately: `xcodebuild -downloadComponent MetalToolchain`, and the build says so if it is missing):

```sh
brew install libomp cmake ninja
cp ~/roms/sm64.z64 .
make sm64-train      # builds N64Bundler and the game the first time, then trains on pss-2
```

On Linux (Debian and Ubuntu package names; run on arm64, and x86-64 is written for and has not been run):

```sh
sudo apt install build-essential clang cmake ninja-build git curl python3 libsdl2-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev
cp ~/roms/sm64.z64 .
make sm64-train
```

- **The trainer** runs on whatever Vulkan device there is. A machine with no GPU can still train on Mesa's CPU driver (`libvulkan1 mesa-vulkan-drivers`): the network is small and sm64 is bound by the games, so it is slower rather than stuck: 1,600 steps a second in an arm64 Linux VM on an M4 Pro, where macOS on the same machine does about 5,000 on Metal.
- **The game** runs headless with no GPU at all: on Linux N64Bundler builds its host without RT64, the renderer (`--no-renderer`), so nothing needs a graphics driver or a shader compiler. A picture in the observation (`make sm64-picture`) and watching in a window (`make sm64-watch`) need the renderer: build with `SM64_RENDERER=1`, which wants `libvulkan-dev` as well.
- **raylib**, which every env links because PufferLib's `pufferenv.h` includes it, releases no build for arm64 Linux, so `build.sh` builds it from source there: that is what the X11 and OpenGL headers are for.
- **In a container**, give it more shared memory than Docker's default 64MB (`--shm-size=1g`): each game shares about 9MB of it with the env.

`make setup` does the fetching and the trainer's build up front, if you would rather.

## Play, train, watch

```sh
make play    # A/D move, W/Space jump, S drop through platforms, R restart, ESC quit
make train   # on the GPU; checkpoints go to checkpoints/platformer/
make eval    # watch the most trained checkpoint
make test    # the trainer against the CPU, against itself, and against golden numbers
```

Pass extra flags with `ARGS`, e.g. `make train ARGS="--train.total-timesteps 10_000_000"`, and pick the env with `ENV=sm64`.

## Build notes

- `build.sh` stands in for PufferLib's build script, which compiles an env from its `ocean/` into the CUDA trainer or a CPU play binary. This compiles `vecenv.c` around an env from `envs/` into a library of its own instead, so envs build side by side. It links OpenMP (Homebrew's on a Mac), and raylib, which 5.0's `pufferenv.h` includes for every env and which it downloads, or builds from source where raylib releases no build (arm64 Linux). `RAYLIB` names one that is already somewhere.
- Configs stay beside their envs: the trainer reads `envs/<env>/<env>.ini` itself, over PufferLib's `default.ini`, and finds both from the directory it is run in, or from `PUFFERL_ROOT`.
- The sm64 vecenv should step each game on a thread of its own, which is `num_threads` equal to `total_agents` in `sm64.ini`. A thread stepping a game spends its time waiting on a socket, so the usual rule of one thread per core would run the games one after another.
