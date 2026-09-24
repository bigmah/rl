# Plan: a text-driven agent, frames in and actions out

Written 2026-09-24. Nothing here is built yet.

## The goal

An RL agent that plays a game with access to nothing but:

- **the screen** (frames in),
- **the controller** (actions out),
- **a sentence** saying what to do ("go get the star at the top of the mountain").

No game memory anywhere: not in the observation, not in the reward, not in exploration. What
"good" means comes from the sentence, read by a VLM/LLM. Super Mario 64 is the first game.

The one exception is a **scoreboard**. The memory-read star check (and position, for plots) stays,
but only in the logs. Nothing the agent sees or is paid by comes from it. Without it we can't
tell a text reward that steers toward real goals from one that drifts.

## What we already know

From the director hours on 2026-09-20 (branch `scuffed-shit-ignore-this`, `src/director.rs`) and
the picture runs of 2026-09-19 and 2026-09-21:

1. **Gemma 4 E4B cannot compare two frames.** "Better or worse" tracks how much the picture
   changed: sliding backwards up the slide scored +5, the same as going forward. "Which is later"
   answers "the second" even for two copies of one frame. More frames in one prompt makes this
   worse. **Never ask the VLM to compare pictures.**
2. **E4B describes single frames well.** 157 distinct, accurate captions in 248 looks. It picked
   out the star (at 480×360; 80×60 is too small) and "falling through open air".
3. **Naming the game in the prompt makes E4B give the same answer for every frame.** Keep it out.
4. **A reward from the VLM alone is too sparse.** One verdict per ~4.6 s is about one per 9,000
   policy steps. With it as the only reward: zero stars, ground per episode down from 67,000 to
   13,000, and the losses went to zero.
5. **Exploration is what gets a blank-slate agent moving.** Novelty (`novelty_episode 0.02`) plus
   Go-Explore is what made the picture-only policy touch the star (first touch at 89 min). Both
   currently read Mario's position from memory.
6. **The picture-only policy works.** CNN + MinGRU from 80×60 pixels reached the star (782, then
   776 frames). The 4-frame stack (`picture_stack`, uncommitted) showed no edge in two hours.

## The architecture

```
goal text ──► planner LLM ──► current subgoal text ─────────────┐
                                  ▲                             ▼
                                  │ "done?"     frames ──► policy + goal embedding ──► actions
                                  │                │         (every step)
frames ──► VLM captions (async, slow) ─────────────┤
              │                                    ▼
              ▼                         reward net: (frame, goal) ─► score   (every step)
   text judge: captions + goal                     ▲
   ─► "which is closer to the goal?"  ─────────────┘ trained on the judge's preferences
```

Five parts.

### 1. VLM as describer (slow, async)

Gemma captions sampled frames at 480×360 or more. It only ever describes one frame. That's the
thing finding 2 says works. It runs on its own thread like the current director, drops offers
while busy, and never blocks the trainer.

### 2. Text judge (slow, async)

A text LLM gets the goal plus two captions and answers which one is closer to the goal, or says
they're even. It compares language, not pixels, so finding 1 doesn't apply. It can be the same
Gemma in text-only mode, or a bigger text model if E4B's judgments are weak. This is Motif's
method (Klissarov et al. 2023, "Motif: Intrinsic Motivation from Artificial Intelligence
Feedback"); there it judged NetHack's text messages, not captions of frames.

Pairs come from the same episode and from different episodes under the same goal. Store every
(frame, caption, goal, preference) in a label store on disk. It's reused across runs and is the
most valuable thing the system produces.

### 3. Reward net (fast, every step)

A small CNN on the frame, conditioned on the goal embedding, outputs a scalar. It's trained on the
judge's preferences with a Bradley–Terry loss, is retrained periodically as labels arrive, and
runs on the GPU inside the rollout like the policy. This solves finding 4: the reward is dense and
immediate, and it's still defined only by text.

Guard against reward hacking: pay the reward net's *increase* over the episode's best so far
(pay for progress once, not for standing in a good-looking spot), and clip the per-step reward.

### 4. Goal-conditioned policy with hindsight relabeling

The policy is today's picture policy (`src/model.rs` convs → encoder → MinGRU ×4 → heads), plus
the goal embedding added to the encoder input. The director branch already embeds text into the
observation: mean-pooled `token_embd.weight` rows, projected to `embed_dim`. A frozen sentence
embedding from the VLM is the upgrade if mean-pooling loses too much.

**Hindsight relabeling is what makes it drivable from text.** After each episode, caption what
actually happened ("Mario climbed the tree", "Mario fell into the moat") and add the episode to
training again with that caption as its goal, paid as a success. Every episode teaches *some*
goal. That's how the policy learns the text matters without one success per prompt (hindsight
experience replay, Andrychowicz et al. 2017, with a VLM writing the relabeled goal).

On-policy PPO complicates this: relabeled episodes are off-policy. Options, in order:

- relabel only the goal embedding and recompute advantages with V-trace (`vtrace` is already in
  `Hyper`)
- keep a small relabeled replay alongside the rollout
- use relabeled episodes for an auxiliary goal-prediction head only (cheapest, weakest)

### 5. Planner

At episode start, and whenever the current subgoal is judged done, the text LLM gets the top goal,
the latest caption and the subgoals done so far, and writes the next subgoal. The policy sees the
**subgoal** embedding. "Done" is a yes/no question to the VLM on one frame ("Is Mario standing on
the bridge?"), which is detection, a thing it can do.

### Exploration from pixels

Replace the memory reads in novelty and Go-Explore:

- **Novelty:** Random Network Distillation (RND; Burda et al. 2018) on the frame, or on the
  policy's own conv features, as the intrinsic reward.
- **Go-Explore cells:** a hash of a heavily downscaled, quantized frame (the original Go-Explore's
  Atari trick: e.g. 11×8, 8 grey levels), or the caption itself as the cell key. The caption as
  key gives cells like "on the bridge" and "in the water", which is what the slide runs were
  missing (the "finer cell key" note from 2026-09-17).
- **Returning to cells:** savestates, as now. Restoring the emulator state is part of the harness,
  not the agent's observation, so it doesn't break "frames only".

## Build order

Each phase ends in something that runs and a thing to look at in the logs or the mp4.

### Phase 0: port the director to main

- Cherry-pick the working parts of `scuffed-shit-ignore-this` onto main: `vendor/llmoxide` as a
  submodule, the async Gemma thread, the text embedder, `[env] star_reward` (to turn the memory
  reward off), and `examples/director.rs`.
- Leave out the score/verdict reward path (finding 1).
- Commit `picture_stack` / `picture_gap` or drop them; they're uncommitted on main.

### Phase 1: memory-free observation and exploration

- Picture only (`state 0`), 480×360 frames kept for the VLM, 80×60 (or larger) for the policy.
- RND novelty and a pixel-hash (or caption) Go-Explore cell key, both behind ini flags.
- Scoreboard: log memory-read star and position; never feed them in.
- **Done when** a memory-free run still explores (cells/episode and distance in the range of the
  2026-09-19 `img` run), even with no goal reward.

### Phase 2: caption → judge → label store

- Async caption thread (from phase 0) writes (frame, caption) to the label store.
- Judge thread pairs captions under the current goal and writes preferences.
- Dashboard: labels/min, judge agreement with itself on repeated pairs, and a sample of recent
  captions and preferences.
- **Done when** it runs for an hour next to training without slowing SPS by more than ~10%.

### Phase 3: reward net

- Small CNN + goal embedding → scalar, Bradley–Terry on the label store, retrained every N
  labels, run in the rollout.
- Pay its increase over the episode's best, plus the phase 1 novelty.
- Log the reward net's score against the scoreboard (does it rise on runs that reach the star?).
- **Done when** a single-goal run (one star, one sentence) reaches the star with no memory reward.

### Phase 4: many goals and hindsight relabeling

- Goal set: sentences for many stars and landmarks across open courses (Bob-omb Battlefield,
  Whomp's Fortress, Cool Cool Mountain first; not the slide, whose 21 s requirement is a precision
  problem, not a language one). `[env] star = <course>-<n>` and `stars/*.ini` pick the level.
- Each agent gets a goal sentence; episodes are captioned and relabeled.
- **Done when** a policy trained on goal set A follows a *rephrased* goal from A, measured by the
  scoreboard.

### Phase 5: planner

- LLM subgoals from the top-level sentence; VLM yes/no checks move to the next subgoal.
- mp4s with the current subgoal and captions burned in, so you can watch plan and play together.
- **Done when** "go get the star at the top of the mountain" produces a subgoal chain and a policy
  that follows it on Bob-omb Battlefield.

### Phase 6: a held-out goal

- A star or landmark never in the training goals, given only as text.
- This is the big prize. Everything before it is scaffolding.

## Open questions and risks

- **Judge quality.** Can a text LLM rank captions against a goal reliably? If E4B's text judgments
  are weak, use a bigger text model; the judge is off the hot path, so it can be slower.
- **Caption resolution.** Captions blur fine distinctions (the slide's two tracks read as one
  "chute"). The reward net can only be as fine as the captions. This is the likeliest ceiling.
- **Reward-net drift.** Retraining changes the reward under the policy. Retrain on a schedule,
  keep the old net for a warm start, and watch value loss spikes.
- **Compute on one Mac.** Policy, reward net, VLM and judge share one GPU. llmoxide has its own
  wgpu context; measure contention in phase 2 before adding the reward net.
- **Credit for slow labels.** The reward net sidesteps it (it scores every step). The old
  director's `horizon` ~1024 note only matters if anything is paid straight from a verdict.
- **Off-policy relabeling under PPO.** See part 4. Start with V-trace.
- **Precedents.** DeepMind's SIMA and MineDojo's MineCLIP reached text-following in 3D games with
  large human video/demo datasets and far more compute. No demonstrations here: an open research
  problem, not a checklist.

## References

- Motif: Klissarov et al. 2023, LLM preferences over captions distilled into a reward.
- MineDojo / MineCLIP: Fan et al. 2022, a video–text model as a Minecraft reward.
- Hindsight Experience Replay: Andrychowicz et al. 2017.
- Random Network Distillation: Burda et al. 2018.
- Go-Explore: Ecoffet et al. 2019/2021, downscaled-frame cells.
- ELLM: Du et al. 2023, an LLM suggesting goals for exploration.
- SIMA: DeepMind 2024, text-instructed agents across 3D games from pixels.
