# Peach's Secret Slide in 648 frames

`star-648.demo` is the fastest star on record here for Peach's Secret Slide
(`pss-2`): the pad inputs that take Mario from the course savestate to
the star's spawn in 648 frames, 21.6 seconds at the game's thirty a second. The
game is deterministic, so playing the inputs back spawns the star on the same
frame every time.

It was found when the spawn was the goal, so it ends there: the star comes down
and Mario does not go and take it. The goal has since become the touch, and the
tool reports this demo as "the star spawned at frame 648, as this demo says".

It is kept here because the copy the tools use, `build/sm64/pss-2/fastest.demo`,
is overwritten by any run that finds a faster star. This one stays as it is.

**Every run here takes the slide's second star**, the one for reaching the
bottom inside 21 seconds of the slide's own timer, not the box: the save file's
flag for it is what goes on, the HUD timer stops at 564 frames (18.8 seconds) on
the 786, and the same route with the timer set past 21 seconds spawns no star at
all. The kick or ground pound at the bottom is not what earns it. So the env
calls this star `pss-2`, and a run slower than 21 seconds down the slide gets
nothing. See *When a star is won, and which* in the top-level README.

`star-648.mp4` is the same run as a video: every frame the console showed, at
thirty a second, from the savestate to the star's spawn and five seconds beyond
so the star comes down. 640 by 474, twice the size the game renders headless,
scaled by whole pixels. It plays anywhere; nothing below is needed to watch it.

## The touch: 786 frames

`star-786-touch.demo` goes all the way: the same route to the box, the hundred
frames frozen while the star comes out, and then a jump to it, with the star
his on frame 786, 26.2 seconds from the savestate. `star-786-touch.mp4` is its
video, with the star dance after, and `route-786-touch.txt` its trace. It was
the fastest touch on record until the 770 below, found on 2026-09-17 by an hour of the backward
algorithm along the 648 run: episodes began at savestates ten, twenty, up to
sixty frames before the box, with only the star and the clock paying, and the
first touch from any of them became the demo. All nine of its 786-frame touches
come from the last few frames of the route; the frontier got sixty frames back
up the slide and no further in 15M steps. The 648 is 138 frames shorter only
because it stops at the spawn: about a hundred of those frames are the freeze,
which no run can shorten, and the rest the jump to where the star lands.

## The touch: 770 frames

`star-770-touch.demo` was the fastest touch on record until the 768 below: the star is his on frame
770, 25.7 seconds from the savestate. `star-770-touch.mp4` is its video and
`route-770-touch.txt` its trace. It was found later on 2026-09-17, fifteen
minutes into thirty of `make sm64-slide` (now `make sm64-train STAR=pss-2`) with the touch as the reward, the same
changes from the target as under "How it was found" below but 7M steps, and
`load_model_path` the checkpoint the 786 came from. The archive starts empty in
every run, so nothing of the 786 was handed to it: this is a run put together
from the top of the slide within the half hour.

It is not the 786 with a better ending. It is thirty frames slower over the top
(sliding at frame 120, where the 786 is by 90) and slower down the upper track,
and it wins all of that back and more at the jump: it leaves the upper track
later, between frames 300 and 330, and flies forward at 85 to 95 units a frame instead
of meeting a wall and dropping straight down, so it lands further along the
lower track and is in the box by about frame 632, where the 786 took 648. The
786's top with this jump would be a touch of about 720, which no run has put
together yet.

The same run found touches of 776, 778, 780 and 780; they are in
`build/sm64/pss-2/fastest/`. Like every star before them, all
of its touches began from an archive start partway down: of about 4,600
episodes that began at the top of the slide, none touched the star.

## The touch: 768 frames

`star-768-touch.demo` was the fastest touch on record until the 750 below, with `route-768-touch.txt`
its trace and no video, because it is the 770 to the eye: the same inputs to
frame 450, then a tighter line through the last two turns that has him in the
box four frames sooner, and two of those given back on the way to the star. It
came 23 minutes into an hour of `make sm64-slide` that began with the archive
seeded from the runs on file (`--env.go-explore-seed 1`, see the top-level
README), which touched the star in one episode in ten where the run before did
in one in seventy, and still moved the record two frames.

The run it was after is the 786's top with the 770's jump, about 720 frames: the
786 is fifty frames ahead where the 770 leaves the upper track. It did not come,
and not for want of tries. Started on the 786 a second before its jump, the
hour's last checkpoint played 3,400 episodes with no training: 50 touches, none
under 784, and one and a half deaths an episode. It does jump from there. It
does not land.

## The touch: 750 frames

`star-750-touch.demo` was the fastest touch on record until the 674 below: the star is his on frame
750, 25.0 seconds from the savestate. `star-750-touch.mp4` is its video and
`route-750-touch.txt` its trace. It came 40 minutes into an hour of
`make sm64-slide` with the archive seeded from the runs on file and the ghost
paying (`--env.go-explore-seed 1 --env.ghost 0.02`, see the top-level README):
the first time in an episode Mario enters a cube on a way to the star sooner
than the archive's way into it, on the ground, he is paid 0.02 a frame of the
difference, 0.1 at most, and the archive then keeps his way, so only a faster
run is ever paid.

It is the splice the seeded hour was after, and then some. Its first 142 frames
are the 786's inputs exactly -- one of the ten runs seeded carried that top --
and it holds the cap down the upper track, where the 770 dropped to 70 and 85.
It leaves the upper track at about frame 260, where the 770 does, but on a
shorter flight that lands at frame 360, where the 786's ground pound lands at
385 and the 770's long flight at 412. It lands higher up the lower track than
the 770, so it is in the box at about the same frame, 630. The rest is the
ending: it takes the star 18 frames after the freeze, where the 770 took 30.

The same hour touched the star in a fifth of its episodes by the end, and left
the fastest folder all 750 to 764: every run in it under the 768 it began with.
The 770's long flight from this top, landing at 412 two turns further down,
would be in the box near 580 and worth another forty frames or so. Still none
from the top of the slide.

## The touch: 674 frames

`star-674-touch.demo` is the fastest touch on record: the star is his on frame
674, 22.5 seconds from the savestate, in 337 inputs. `star-674-touch.mp4` is its
video and `route-674-touch.txt` its trace. It came in the second of two
back-to-back hours of `make sm64-slide` on 2026-09-17 evening, both with the
archive seeded from the runs on file and the ghost paying, each loading the last
checkpoint of the one before. The first hour took the record 750 to 746 to 736;
the second found 724, 714, 712, 700, 690, 686, 684 and finally 674 in its last
twenty minutes, the whole pack moving down together.

It is the long flight, and further than anyone had it. Its first 60 frames off
the start are the 750's exactly, and it holds the sliding cap down the upper
track the way the 750 does, arriving at the end of it within a couple of hundred
units and a frame or two of where the 750 is. Then it takes the
turn at the end of the upper track backwards, sliding at 72 to 81 units a frame
with his back to the way he is going from frame 240 to 276, swings around to
face forward, and jumps at frame 296 -- 36 frames later than the 750 and a
little further along. The fall is the whole of the difference: three and a half
seconds in the air, from a height of 3,220 down to minus 3,529, landing on the
lower track at frame 412 at x -3,816. The 750's shorter jump lands at frame 360
but up at x +1,604, with two thirds of the lower track still to run. The 674
lands past all of that and is in the box at frame 541, where the 750 took 630.
The ending is the same as every run's: about 110 frames frozen while the star
comes out, and the star about twenty frames after.

So it is 52 frames slower into the landing and 89 frames faster out of it. The
note's estimate for this line from the 750's approach was a star near 710; it
came in 36 frames under that, because the flight also carried him past the two
turns the 770 still had to slide.

The fastest folder is now 674 to 716, all of this family, and the 750 family is
out of it. Still none from the top of the slide: like every record before it,
this is a run the archive assembled from starts partway down.

Everything below applies to these the same way, with the file name in place of
`star-648.demo`.

## Playing it

The game has to be set up first as the top-level README describes: N64Bundler
with your own `sm64.z64`, then `make sm64` to build `build/sm64_tool`. The first
run on this star plays through the intro once to make the course savestate
(`build/sm64/pss-2/start.state`), which takes a few seconds and happens on its
own.

Watch it in a window, at the speed a console ran it, with five seconds after the
spawn so the star comes down:

```sh
./build/sm64_tool replay watch demos/peach-slide/star-648.demo --env.star pss-2
```

Check it without a window. This replays at about three times speed and confirms
the star spawns on frame 648:

```sh
./build/sm64_tool replay demos/peach-slide/star-648.demo --env.star pss-2
```

Print where Mario is every thirty frames on the way, which is what `route.txt`
is (`SM64_TRACE` works with `watch` too):

```sh
SM64_TRACE=30 ./build/sm64_tool replay demos/peach-slide/star-648.demo --env.star pss-2
```

Make the video again, or one of another demo. This plays the demo headless with
the game drawing every frame and hands them to `ffmpeg`, which has to be on the
path (`brew install ffmpeg`):

```sh
./build/sm64_tool record demos/peach-slide/star-648.demo demos/peach-slide/star-648.mp4 --env.star pss-2
```

`--env.star pss-2` is what tells the tool to load the slide's savestate. It is
the star `sm64.ini` names, so it can be left off while that stays so; with
another star the inputs play in another course and mean nothing.

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
