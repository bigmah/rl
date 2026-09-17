# Peach's Secret Slide in 648 frames

`star-648.demo` is the fastest star on record here for Peach's Secret Slide
(`SM64_LEVEL=27`): the pad inputs that take Mario from the course savestate to
the star's spawn in 648 frames, 21.6 seconds at the game's thirty a second. The
game is deterministic, so playing the inputs back lands the star on the same
frame every time.

It is kept here because the copy the tools use, `build/sm64/star-level27-act0.demo`,
is overwritten by any run that finds a faster star. This one stays as it is.

`star-648.mp4` is the same run as a video: every frame the console showed, at
thirty a second, from the savestate to the star's spawn and five seconds beyond
so the star comes down. 640 by 474, twice the size the game renders headless,
scaled by whole pixels. It plays anywhere; nothing below is needed to watch it.

## Playing it

The game has to be set up first as the top-level README describes: N64Bundler
with your own `sm64.z64`, then `make sm64` to build `build/sm64_tool`. The first
star run on this course plays through the intro once to make the course savestate
(`build/sm64/star-level27-act0.state`), which takes a few seconds and happens on
its own.

Watch it in a window, at the speed a console ran it, with five seconds after the
spawn so the star comes down:

```sh
SM64_GOAL=star SM64_LEVEL=27 ./build/sm64_tool replay watch demos/peach-slide/star-648.demo
```

Check it without a window. This replays at about three times speed and confirms
the star spawns on frame 648:

```sh
SM64_GOAL=star SM64_LEVEL=27 ./build/sm64_tool replay demos/peach-slide/star-648.demo
```

Print where Mario is every thirty frames on the way, which is what `route.txt`
is (`SM64_TRACE` works with `watch` too):

```sh
SM64_GOAL=star SM64_LEVEL=27 SM64_TRACE=30 ./build/sm64_tool replay demos/peach-slide/star-648.demo
```

Make the video again, or one of another demo. This plays the demo headless with
the game drawing every frame and hands them to `ffmpeg`, which has to be on the
path (`brew install ffmpeg`):

```sh
SM64_GOAL=star SM64_LEVEL=27 ./build/sm64_tool record demos/peach-slide/star-648.demo demos/peach-slide/star-648.mp4
```

`SM64_GOAL` and `SM64_LEVEL` are what tell the tool to load the course savestate
rather than the castle grounds. Without them the inputs play on the grounds and
mean nothing.

## The route

| frames | seconds | what happens |
|---|---|---|
| 0 to 90 | 0 to 3 | a jump off the start and onto the slide, already at 73 units a frame |
| 120 to 270 | 4 to 9 | the upper track at the game's sliding cap of 100 units a frame |
| 300 to 390 | 10 to 13 | jumps off the upper track at a height of 2,900 and free-falls for three seconds, landing on the lower track at minus 1,500; this skips the middle turns, about 230 frames of sliding |
| 390 to 630 | 13 to 21 | the lower track at the cap |
| 648 | 21.6 | slides into the box at full speed and the star spawns |

The previous record, 936 frames, slid the whole track and finished with a long
jump into the box. The demo before that, 1,326, was the explorer's first star.

## How it was found

On 2026-09-17 by an hour of `make sm64-slide`-style training on the PufferLib 5.0
MLX trainer, with these changes from the target:

```
--env.novelty-episode 0 --train.norm-adv 1 --train.learning-rate 0.005
--train.ent-coef 0.001 --train.total-timesteps 15_000_000
```

and `load_model_path` pointing at a checkpoint from an earlier twenty-minute
slide run. With the per-episode cube payment off, the star and the clock were
the only things worth having, and the archive's starts partway down the slide
gave the policy stars to learn from. The route appeared at 8.5M steps as a
686-frame star, then 652 minutes later, then 648 at 12.3M steps. All of its
stars began from an archive start along the route; none began at the top of
the slide, so this is a run the process found and can replay, not yet one the
policy performs from the start.
