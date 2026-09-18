# Keeping the slide going

Where things stand after the evening of 2026-09-17: the fastest touch of the
star is **674 frames** (`star-674-touch.demo`), down from 750 in two back-to-back
hour-long runs. The archive is seeded from the runs on file at the start of every
run and the ghost pays for beating it. Two runs took the record down 76 frames;
the first hour gave 4 then 10, and the second gave 62 in a single stretch once
the long flight landed. This is how to run the next one.

The runs below need `go_explore_seed`, `ghost` and `ent_coef_damping`, which
are in `sm64.h`, `sm64.ini` and the trainer (`src/`).

Found on 2026-09-18, and worth knowing before the next run: the star every run
here takes is **the slide's second, for reaching the bottom inside 21 seconds**
of the HUD timer, which starts when the sliding does. Reach the bottom slower
and no star appears at all. So an episode from the top pays nothing until it
is already fast -- which may be much of why no run has ever taken it from the
top -- and the env calls this star `pss-2`. `pss-1`, the box, is a different
goal that no run has.

## One run

```sh
LAST=$(ls -t checkpoints/pss-2/sm64 | head -1)
CKPT=checkpoints/pss-2/sm64/$LAST/$(ls checkpoints/pss-2/sm64/$LAST | tail -1)
make sm64-train STAR=pss-2 ARGS="--env.novelty-episode 0 --train.norm-adv 1 \
  --train.learning-rate 0.005 --train.ent-coef 0.007 --train.ent-coef-rate 0.0025 \
  --train.ent-coef-damping 0.5 --env.go-explore 0.9 --env.go-explore-star 0.75 \
  --env.go-explore-seed 1 --env.ghost 0.02 --train.total-timesteps 12_000_000 \
  --base.load-model-path $CKPT"
```

- 12M steps is about 60 minutes. The two runs on 2026-09-17 evening held 3.3 to
  3.4K steps a second averaged over the hour, dipping to 2.5K and touching 4.6K;
  at 2.5K, 12M would take 80 minutes, so size `total-timesteps` off the rate you
  see if the run has to fit a window. The run ends by itself and writes
  `logs/pss-2/sm64/<run id>.ini`.
- It loads the last run's last checkpoint. The first line it prints after the
  build should be `sm64: the archive starts with N cubes ... from 13 runs on
  file`: the touch demo, the ten in `build/sm64/pss-2/fastest/`,
  and the 770 and 750 in `build/sm64/pss-2/seeds/`. N fell from 206 to
  169 between the two runs as the lines got shorter; a smaller archive is not a
  worse one.
- Every touch faster than the demo becomes `build/sm64/pss-2/fastest.demo`
  on its own, and the ten fastest stay in the folder, so the next run is
  seeded with this run's best. Nothing has to be copied between runs.

## While it runs

Watch three numbers on the dashboard:

- `perf` is the touch rate, and it is a per-interval sample: it read 0.000 on
  most checks of both runs and 0.250 to 0.333 on others, while the record fell
  the whole time. Do not read one 0.000 as a stall. It averaged about a fifth
  to a quarter by the end.
- `entropy` should hold between 2 and 3. Both runs sat at 2.4 to 2.8, with one
  dip to 1.87 in the middle of the first hour that came back on its own. If it
  swings between 0.3 and 4.6 every few minutes, `ent_coef_damping` is not on.
- `ghost` is what beating the archive paid an episode. It read 0.000 on nearly
  every check of both hours, including the twenty minutes that found 62 frames,
  so it is a poor progress signal at this point; the file names are the signal.

And the file names in `build/sm64/pss-2/fastest/`: they are
the frame counts, fastest first. Copy a new record out to a scratch directory
as soon as it appears -- the folder keeps only ten, and a good line can be
pushed out by a later family.

To stop a run early: `kill -INT -- -$(ps -o pgid= -p $(pgrep -f "target/release/pufferl train sm64" | head -1) | tr -d ' ')`.
That is the trainer's process group, which the eight games are in too. A run
stopped this way writes no log.

## When it beats the record

```sh
./build/sm64_tool replay --env.star pss-2
cp build/sm64/pss-2/fastest.demo demos/peach-slide/star-NNN-touch.demo
SM64_TRACE=30 ./build/sm64_tool replay demos/peach-slide/star-NNN-touch.demo --env.star pss-2 > demos/peach-slide/route-NNN-touch.txt
./build/sm64_tool record demos/peach-slide/star-NNN-touch.demo demos/peach-slide/star-NNN-touch.mp4 --env.star pss-2
```

Do this after the run ends, not during: the tools write into the same
`build/sm64/pss-2` files the run does. Compare the trace with `route-674-touch.txt`
to see where the frames came from, and add a section to `README.md` here.
`SM64_TRACE=5` is the one to use for finding the exact jump and landing frames.

## Where the next frames are

The 674 is the 750's top and upper track, the end of the upper track taken
backwards at 72 to 81 units a frame from frame 240 to 276, a jump at frame 296,
three and a half seconds of falling from height 3,220 to minus 3,529, a landing
at frame 412 at x -3,816, the lower track at the cap, the box at 541, about 110
frames frozen, and the star about twenty after. The ending and the freeze are
fixed, so everything left is getting to the box sooner than 541.

- **The jump is late and slow.** He gives up the cap to swing around at frames
  276 to 296, crossing the jump point at 44 to 61 units a frame where the rest
  of the route runs at 100. Leaving the upper track at the cap, facing forward,
  would both save those frames and throw him further down the lower track. The
  backwards stretch is probably how the policy found the line rather than
  something the line needs.
- **The landing is at x -3,816 with a quarter of the lower track left.** The
  box is at x -6,296, z 3,166. A flight that carries past the next turn is
  worth another sixty or so frames, and the fall is already three and a half
  seconds, so the height for it is there.
- **Keep old lines in play.** The fastest folder keeps only the ten fastest and
  is now all 674-family; the 750 and 736 are already out of it. A run whose line
  is worth keeping goes in `build/sm64/pss-2/seeds/`; seeding
  reads every `.demo` there as well. The 770 is in there now, and its long
  flight has been superseded by the 674's -- it can come out if the seeds folder
  starts costing more than it gives.
- **A bigger drop, hundreds of frames.** The upper track at frame 210 of the
  786 (x -6605, z -1376, height 3792) is 145 units in plan from the lower
  track 48 frames before the box (frame 600: x -6730, z -1300, height -4079),
  7,900 units below. If that edge is open, the star is around 500. Nobody
  knows: it needs the course's floor triangles read out of memory to see
  what is between them, or a scripted probe over the edge. The 674 falls 6,750
  units in one go, so a drop of that size is not out of the question.

## What does not help

- Episodes from the top of the slide. No policy has ever touched the star from
  there; the record is a run the archive assembles, and `go_explore 0.9`
  already spends almost nothing on the top.
- Seeding a line the policy cannot finish. The 786's approach was seeded for
  an hour and its jump never landed from there; a faster way into a cube the
  policy dies from blocks the way it can finish.
- More than one run at once. Eight games is about the machine's throughput.
- Stopping a run early because `perf` or `ghost` reads zero. Both read zero
  through the stretch that found the 674.
