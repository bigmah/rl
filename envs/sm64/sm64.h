/* Super Mario 64: open the castle's front door as soon as Mario can.
 *
 * The game is the real cartridge, statically recompiled to native arm64 by
 * N64Bundler and run in a process of its own (see n64b_gym.h). This file is the
 * environment around it: what an action does to the controller, what Mario's
 * state is worth, and where all of that lives in the console's memory.
 *
 * Reward is the door, the clock, and novelty. The door pays DOOR_REWARD and
 * ends the episode the frame it starts to open. Every frame costs
 * time_penalty / max_ticks, so an episode that runs out of time has paid
 * time_penalty in all. Nothing tells the policy where the door is or pays for
 * getting nearer to it: it sees what Mario is doing and where he is, and
 * finds out about the door by opening it.
 *
 * The door is a long way to find by accident: 7600 units in a straight line,
 * past a moat, over a bridge and through one of Lakitu's speeches, then walked
 * into rather than dived at. 400 episodes of random buttons never opened it,
 * and only one got as far as Lakitu. With the door and the clock alone, every
 * episode is worth exactly -time_penalty until the first door, and 12M steps
 * of training never found it: entropy stayed at its maximum, and the typical
 * episode ended as far from the door as it began. Nothing in that setup
 * remembers where Mario has been, so exploring is jittering the stick.
 *
 * Novelty is that memory, and it knows nothing about the door. The castle
 * grounds are cut into cubes novelty_cell units a side. The first time in an
 * episode that Mario enters a cube, he is paid
 *
 *     novelty_episode + novelty / sqrt(n)
 *
 * where n is how many episodes -- in every game this process runs -- have
 * entered it, this one included. Only the first entry in an episode counts, so
 * pacing across a boundary earns nothing.
 *
 * The second part is for places nobody has been. The start, which every
 * episode sees, is worth almost nothing within minutes, and a cube nobody has
 * reached pays all of it. On its own it found the door within 100K steps, and
 * then lost it: by 500K the cubes on the way had been entered thousands of
 * times, novelty had fallen from 0.7 an episode to 0.05, episodes drifted back
 * to the start, and in 5M steps the door opened about five times -- too rarely
 * for the policy to learn the bridge, Lakitu and the door from.
 *
 * The first part never fades: an episode that covers ground is always worth
 * more than one that does not, so the far side of the grounds keeps being
 * reached, and with it the door. It is small. 0.02 a cube is about what
 * running costs in clock, so a run to the door -- fifteen cubes, the door, and
 * the clock left over -- is still worth more than wandering all episode.
 *
 * Novelty with both parts did not get there either. By 10M steps its episodes
 * ranged 3000 to 7700 units from the start, and the door opened once in a
 * thousand. They swam: the moat wraps the castle, dozens of new cubes that
 * lead nowhere near the door, while the way to it is a bridge of two cubes
 * and then 250 frames of Lakitu with nothing new at all. Not one of eight
 * traced episodes met Lakitu.
 *
 * So some episodes start further on (Go-Explore). Every cube any episode has
 * entered is kept in an archive with the shortest run of pad inputs, from the
 * savestate, that reached it. The game is deterministic -- 1400 random inputs
 * replayed give the same Mario to the bit, in the same game or another -- so
 * the inputs are the place, at a few kilobytes instead of a savestate's eight
 * megabytes. A go_explore share of episodes pick a cube, weighted to the ones
 * fewest episodes have entered, replay its inputs, and only then start the
 * clock. The rest start where the task does. The reward is the same either
 * way; only where some episodes begin has changed, and the log keeps the door
 * rate from the real start apart (start_perf / from_start).
 *
 * Exploring is Go-Explore's first phase, and what it finds is a way to the
 * door, not a policy that can open it from the start. Its second phase makes
 * one: every exploring run that opens the door offers its inputs to a demo file, which
 * keeps the fastest, and a `backward` share of episodes start on that demo,
 * moving back from the door as the policy learns (see the fastest door).
 *
 * A death, or a warp out of the castle grounds, pays for the rest of the
 * clock at once, so no episode is ever worth less than running it out, and
 * none worth more: stopping the clock early is not a way to lose less.
 *
 * Two things stand between a fast runner and the door, both measured in the
 * running game. The door opens only for a Mario who walks into it: a dive
 * bonks off it and a punch does nothing. And Lakitu stops him the first time
 * he steps onto the bridge, for about 250 frames of dialog even with A mashed.
 * The episode goes on through that and the clock keeps running.
 *
 * An earlier version also paid for closing the straight-line distance to the
 * door, and ended the episode in water so that distance could not lure Mario
 * into the moat. It opened the door in 92% of episodes after 3M steps. This
 * one is for finding out what that help was worth.
 *
 * Episodes start from a savestate of the castle grounds with Mario standing
 * outside the castle, so no episode spends its first thousand frames watching
 * Peach's letter. See sm64_make_state.
 *
 * SM64_GOAL=star swaps the door for a star in Bob-omb Battlefield: episodes
 * start where the painting drops Mario into the course, the star pays what the
 * door did and ends the episode the frame he touches it, and leaving the
 * course is the death. Everything else -- the clock, novelty, the archive, the
 * demo -- is the same code. See the star.
 *
 * The observation is Mario's state read out of memory, and, with picture_width
 * set, the game's picture after it: the frame the console's video interface is
 * about to show, which N64Bundler reads without knowing what game it is (see
 * the picture). The reward reads memory whatever the policy sees.
 *
 * Actions: four discrete heads
 *   stick: 0 none, 1-16 a direction relative to the way Mario is facing
 *          (1 straight ahead, 5 right, 9 back, 13 left)
 *   a, b, z: held or not
 */

#ifndef SM64_H
#define SM64_H

#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef float obs_t;
#include "pufferenv.h"

#include "n64b_gym.h"
#include "sm64_paths.h" /* written by build.sh: where the game and its pieces are */

/* --- where Super Mario 64 (USA) keeps what matters --------------------------
 *
 * Every address here was checked against the recompiled C that N64Bundler
 * generates from the cartridge, and then against the running game.
 *
 * `mario_set_forward_vel` is the one that pins down Mario's speed, and it is
 * unmistakable in the generated code at 0x80251708:
 *
 *     swc1  $f4, 0x54($a0)     <- forwardVel
 *     lhu   $t6, 0x2E($a0)     <- faceAngle[1], his yaw
 *     lwc1  $f6, 0x6000($at)   <- the sine table at 0x80386000, indexed by yaw >> 4
 *     mul.s $f10, $f6, $f8     <- sins(yaw) * forwardVel
 *     swc1  $f10, 0x58($a0)    <- slideVelX
 *     ... the cosine table at 0x80387000 ...
 *     swc1  $f4, 0x5C($a0)     <- slideVelZ
 *     swc1  $f6, 0x48($a0)     <- vel[0] = slideVelX
 *     swc1  $f8, 0x50($a0)     <- vel[2] = slideVelZ
 *
 * which is `mario_set_forward_vel` instruction for instruction, and gives the
 * offsets of forwardVel, the velocity vector, and the facing angle at once.
 * Position sits right before the velocity at 0x3C, and the rest of the offsets
 * below are the same structure's, confirmed by watching them move: the stick
 * arrives in gControllers[0] as stickX, Mario's action becomes ACT_WALKING
 * (0x04000440) and then ACT_JUMP (0x03000880), and gGlobalTimer ticks once per
 * frame the game draws.
 */
#define GLOBAL_TIMER 0x8032D5D4  /* u32, one per frame the game draws */
#define MARIO_STATE_PTR 0x8032D93C
#define CURR_LEVEL_NUM 0x8032DDF8 /* s16; 16 is the castle grounds */
#define CURR_AREA_INDEX 0x8033BACA /* s16 */
#define PLAYER1_CONTROLLER 0x8033AF90

#define LEVEL_CASTLE 6 /* the inside, where the front door leads */
#define LEVEL_CASTLE_GROUNDS 16

/* The front door is two objects, its halves, at x -76 and 77, y 803, z -3155:
 * found by searching memory for positions in front of the castle, then
 * confirmed by walking into each. The left half is pushed and the right half
 * pulled, and either way the game warps to LEVEL_CASTLE about forty frames
 * later. */
#define DOOR_X 0.0f
#define DOOR_Z -3155.0f
#define ACT_PULLING_DOOR 0x00001320
#define ACT_PUSHING_DOOR 0x00001321
/* As much as rewards are clipped to in a step: more would buy nothing. */
#define DOOR_REWARD 1.0f

/* --- the star ------------------------------------------------------------------
 *
 * Getting there. The castle grounds keep a warp node for each half of the
 * front door, {id, level, area, node} in one word: node 0 and 1 lead to the
 * castle's inside. Pointed at a course's node 0x0A instead, which is where its
 * painting puts Mario, the door takes him to that course the way a painting
 * does, act select and all.
 *
 * What counts. Any star: touching one raises M_NUM_STARS that frame. Nothing
 * says which star, where it is, or what makes it appear.
 *
 * Which course, and which act. SM64_LEVEL is the course -- 9 is Bob-omb
 * Battlefield, 24 is Whomp's Fortress, whose stars are out in the open to be
 * climbed to, and 27 is Peach's Secret Slide, where the star is at the bottom
 * of a slide and gravity does most of the work. A course's objects depend on
 * the act it was entered for, and the act select only offers what the save file
 * has, which for a new file is the first and nothing else. The act is a
 * halfword in memory, so SM64_ACT writes it over the selected one while the
 * course loads, before the level script spawns anything -- which is what
 * decides which star is there.
 *
 * A secret course -- the slides, and the ones behind the castle's other doors --
 * has no act select: there is one way in and one course, and the game keeps its
 * act at 0. So for one of those the act is left alone, nothing is written over
 * it, and A is not pressed on the way in.
 *
 * Only the goal moves: the clock, novelty, the archive and the demo are the
 * same, and nothing in the reward or the observation says where a star is.
 * STAR_X and STAR_Z are where Bob-omb Battlefield's act 1 star floats, for the
 * log's `closest` and nothing else; in any other course there is no such place
 * and `closest` is 0.
 */
#define LEVEL_BOB 9
#define LEVEL_WF 24                /* Whomp's Fortress */
#define LEVEL_PSS 27               /* Peach's Secret Slide, which has no act select */
#define CURR_COURSE_NUM 0x8033BAC6 /* s16; Bob-omb Battlefield is course 1 */
#define CURR_ACT_NUM 0x8033BAC8    /* s16 */
/* When the star is won. A star that a box or a boss lets out, or that the slide
 * awards for a time, is spawned: it stops time, plays its cutscene for about a
 * hundred frames while Mario stands frozen, and lands where he can take it. The
 * star is his the frame he touches it, by the star count going up, whether it
 * spawned this episode or was there already. For a while the spawn was the
 * goal, which took the frozen hundred frames out of every star and put the
 * reward on the hit that earned it; but a star that has spawned is not yet
 * won, and a policy that stops there has not taken it. The demos kept then end
 * at the spawn, and sm64_tool says so when it replays one, from
 * gTimeStopState: the halfword at 0x8033D482 -- found by diffing the console's
 * memory before and during the freeze; it is the one word that goes from 0 to
 * 0x4A and stays there -- and a spawning star sets it to ENABLED (2) |
 * MARIO_AND_DOORS (8), with ACTIVE (0x40) coming on as it takes hold. Nothing
 * else in a course sets MARIO_AND_DOORS: dialog sets DIALOG (4) instead, and
 * the doors that set it are in the castle. Touching a star that was already
 * there still counts, by the star count going up. */
#define TIME_STOP_STATE 0x8033D482 /* s16 */
#define TIME_STOP_MARIO_AND_DOORS 0x08
#define DOOR_WARP_NODE_LEFT 0x80196414
#define DOOR_WARP_NODE_RIGHT 0x80196420
#define DOOR_WARP_TO_CASTLE_LEFT 0x00060100 /* node 0 -> level 6, area 1, node 0 */
#define DOOR_WARP_TO_CASTLE_RIGHT 0x01060101
#define PAINTING_NODE 0x0A /* where a course's painting puts Mario, in every course */
#define STAR_X 1550.0f
#define STAR_Z 300.0f

/* struct MarioState */
#define M_INPUT 0x02   /* u16 */
#define M_ACTION 0x0C  /* u32 */
#define M_ACTION_STATE 0x18
#define M_ACTION_TIMER 0x1A
#define M_INTENDED_MAG 0x20 /* f32 */
#define M_INTENDED_YAW 0x24 /* s16, the world direction the stick asked for */
#define M_FACE_ANGLE 0x2C   /* Vec3s: pitch, yaw, roll */
#define M_POS 0x3C          /* Vec3f */
#define M_VEL 0x48          /* Vec3f */
#define M_FORWARD_VEL 0x54  /* f32 */
#define M_WALL 0x60
#define M_FLOOR 0x68
#define M_CEIL_HEIGHT 0x6C
#define M_FLOOR_HEIGHT 0x70
#define M_WATER_LEVEL 0x76 /* s16 */
#define M_NUM_STARS 0xAA   /* s16, goes up the frame he touches a star */
#define M_HEALTH 0xAE      /* s16, 0x880 is full and under 0x100 is dead */
#define SURFACE_NORMAL_Y 0x20 /* f32 inside struct Surface */

/* An action is an index in its low bits, a group in 0x1C0, and flags above. */
#define ACT_GROUP_MASK 0x1C0
#define ACT_GROUP_CUTSCENE 0x100 /* dying, warping, doors, dialog */
#define ACT_FLAG_STATIONARY (1 << 9)
#define ACT_FLAG_MOVING (1 << 10)
#define ACT_FLAG_AIR (1 << 11)
#define ACT_FLAG_SWIMMING (1 << 13)
#define ACT_FLAG_METAL_WATER (1 << 14)
#define ACT_FLAG_BUTT_OR_STOMACH_SLIDE (1 << 18)
#define ACT_FLAG_DIVING (1 << 19)
#define ACT_FLAG_ON_POLE (1 << 20)
#define ACT_FLAG_HANGING (1 << 21)
#define ACT_FLAG_IDLE (1 << 22)
#define ACT_FLAG_ATTACKING (1 << 23)
#define ACT_FLAG_INVULNERABLE (1 << 17)
#define ACT_IDLE 0x0C400201

/* libultra's button bits, as the pad reports them. */
#define BUTTON_A 0x8000
#define BUTTON_B 0x4000
#define BUTTON_Z 0x2000
#define BUTTON_START 0x1000

#define STICK_DIRECTIONS 16
#define ACTION_HEADS 4

/* The observation, in the order it is written. */
enum {
    OBS_POS_X, OBS_POS_Y, OBS_POS_Z,
    OBS_VEL_X, OBS_VEL_Y, OBS_VEL_Z,
    OBS_FORWARD_VEL,
    OBS_SPEED,
    OBS_FACE_SIN, OBS_FACE_COS,
    OBS_PITCH,
    OBS_CAMERA_SIN, OBS_CAMERA_COS,
    OBS_HEIGHT_ABOVE_FLOOR,
    OBS_FLOOR_SLOPE,
    OBS_HEADROOM,
    OBS_TOUCHING_WALL,
    OBS_WATER,
    OBS_ON_GROUND,
    OBS_GROUP_STATIONARY, OBS_GROUP_MOVING, OBS_GROUP_AIRBORNE, OBS_GROUP_SUBMERGED,
    OBS_GROUP_CUTSCENE, OBS_GROUP_AUTOMATIC, OBS_GROUP_OBJECT,
    OBS_ACTION_TIMER, OBS_ACTION_STATE,
    OBS_HEALTH,
    OBS_TIME_LEFT,
    OBS_FLAG_STATIONARY, OBS_FLAG_MOVING, OBS_FLAG_AIR, OBS_FLAG_SWIMMING,
    OBS_FLAG_SLIDING, OBS_FLAG_DIVING, OBS_FLAG_POLE, OBS_FLAG_HANGING,
    OBS_FLAG_IDLE, OBS_FLAG_ATTACKING,
    NUM_OBS
};

/* After those, with picture_width set, the picture: red, green and blue in
 * [0, 1], row by row from the top. */
#define PICTURE_CHANNELS 3

/* How big the observation is depends on the picture, which comes from the env's
 * config, so puf_init sets it -- which is before the vecenv sizes anything by it. */
static int sm64_obs_size = NUM_OBS;
#define OBS_SIZE sm64_obs_size
#define NUM_ATNS ACTION_HEADS
/* stick direction (none, then sixteen ways round), then A, B and Z */
#define ACT_SIZES {STICK_DIRECTIONS + 1, 2, 2, 2}

/* Required struct. Only use floats! */
struct Log {
    float perf;            /* fraction of episodes that opened the door */
    float score;           /* fraction of the clock left when the door opened, 0 if it did not */
    float episode_return;
    float episode_length;  /* agent steps */
    float frames;          /* frames to the door, the whole clock if it stayed shut: what is minimized */
    float closest;         /* the nearest he came to the door without opening it (the bridge starts at 2980). Logged only: the policy never sees it */
    float novelty;         /* what novelty paid this episode */
    float ghost;           /* what beating the archive to cubes on the way to the goal paid this episode */
    float cells;           /* cubes entered this episode */
    float explored;        /* cubes any game has ever entered, as of the end of this episode */
    float from_start;      /* fraction of episodes that began where the task does, not from the archive */
    float start_perf;      /* fraction that began there and opened the door: the door rate is start_perf / from_start */
    float archive;         /* cubes in the archive */
    float replayed;        /* frames replayed to reach the episode's starting cube, 0 from the real start */
    float door_cubes;      /* cubes that have been on the way to a door */
    float frontier;        /* frames before the door that episodes on the demo start at, 0 with no demo */
    float distance;        /* the length of the path he ran */
    float top_speed;       /* the best single frame of it, units per frame */
    float forward_vel;     /* what the game thought his speed was, on average */
    float airborne;        /* fraction of frames off the ground */
    float dialog;          /* fraction of frames in a cutscene, which out here means Lakitu */
    float ended_early;     /* fraction of episodes cut short by a death or a warp that was not the door */
    float deaths;          /* deaths an episode spent; without respawn one ends it, so at most one */
    float n;               /* Required as the last field */
};

/* Required struct */
struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    void* client;
    int num_agents;
    unsigned int rng;     /* vecenv sets this to the env's index; we spawn one game per index */

    int frameskip;        /* frames of the game per agent step */
    int max_ticks;        /* frames of the game per episode */
    int random_start;     /* frames of random stick held after loading the state */
    float time_penalty;   /* reward the whole clock costs, spent a frame at a time */
    float novelty;        /* reward for entering a cube no episode has entered before, fading as 1 / sqrt(episodes) */
    float novelty_episode; /* reward for entering any cube for the first time this episode, which never fades */
    float novelty_cell;   /* the side of a cube, in units */
    int respawn;          /* a death puts the game back and the episode goes on, rather than ending it */
    float go_explore;     /* share of episodes that start from a cube in the archive */
    float go_explore_door; /* share of those that start from a cube on the way to a door, once there is one */
    int go_explore_seed;  /* start the archive with the runs on file rather than with nothing */
    float ghost;          /* reward per frame sooner than the archive's way into a cube on the way to the goal */
    float backward;       /* share of episodes that start on the demo, a little before the frontier */
    int backward_step;    /* frames the frontier moves back at a time */
    float backward_rate;  /* the door rate from the frontier that moves it back */
    int backward_start;   /* frames before the door the frontier begins at; 0 is backward_step */
    int window;           /* draw the game, for watching a policy play */
    int picture_width;    /* the game's picture in the observation, shrunk to this; 0 is none */
    int picture_height;
    int state;            /* Mario's state, read out of memory, in the observation; 0 leaves it zero */

    N64Gym gym;
    int opened;
    int star;             /* the goal is a star in a course, not the door (SM64_GOAL=star) */
    int goal_level;       /* the course that star is in (SM64_LEVEL), and which act of it */
    int goal_act;
    int stars_at_start;   /* the savestate's star count, which a star raises */
    uint32_t mario;       /* where MarioState is, read once from its pointer */
    float last_x, last_z;
    float closest;        /* the nearest he has come to the door, for the log */
    uint32_t episode;     /* counts up from 1, to mark which cubes this episode has entered */
    uint32_t* entered;    /* the episode that last entered each cube */
    float novelty_earned;
    float ghost_earned;
    int cells_entered;
    struct SM64Input* trail; /* every pad input since the savestate, replayed ones first */
    int trail_count, trail_capacity, trail_frames;
    int from_start;
    int start_cube;       /* the archive cube this episode began in, or -1 */
    int start_frontier;   /* the frontier this episode began at on the demo, or -1 */
    int replayed_frames;
    unsigned seed;
    int camera_offset;    /* between the stick and the world, worked out as we go */
    int last_stick_angle;
    int had_stick;
    int tick;             /* frames on the clock */
    int start_tick;       /* where the clock started: 0, or partway for an episode on the demo */
    int steps;
    float episode_return;
    float distance;
    float top_speed;
    float forward_vel_sum;
    int airborne_frames;
    int dialog_frames;
    int deaths;           /* deaths this episode, which only respawn lets past one */
    double next_frame_time; /* for watching it at the speed a television would */
};
typedef Env SM64;

/* --- reading the game -------------------------------------------------------- */

static inline uint32_t sm64_action(SM64* env) { return n64_u32(&env->gym, env->mario + M_ACTION); }
static inline float sm64_pos(SM64* env, int axis) {
    return n64_f32(&env->gym, env->mario + M_POS + 4 * axis);
}
static inline float sm64_vel(SM64* env, int axis) {
    return n64_f32(&env->gym, env->mario + M_VEL + 4 * axis);
}
static inline float sm64_forward_vel(SM64* env) {
    return n64_f32(&env->gym, env->mario + M_FORWARD_VEL);
}
/* Whether there is any floor under him at all. The game keeps the surface he
 * would land on in mario->floor, and it is null only where nothing is below:
 * off the side of a course, falling to the death plane. */
static inline int sm64_in_the_world(SM64* env) {
    return n64_is_ram(n64_u32(&env->gym, env->mario + M_FLOOR));
}
static inline int sm64_face_yaw(SM64* env) {
    return n64_s16(&env->gym, env->mario + M_FACE_ANGLE + 2);
}
static inline int sm64_level(SM64* env) { return n64_s16(&env->gym, CURR_LEVEL_NUM); }

/* An angle the game's way: a signed 16-bit turn of the whole circle. */
static inline float sm64_radians(int angle) { return (float)angle * (2.0f * (float)M_PI / 65536.0f); }

/* How far Mario is from the door, across the ground. For the log, never the policy. */
static inline float sm64_door_distance(float x, float z) {
    return sqrtf((DOOR_X - x) * (DOOR_X - x) + (DOOR_Z - z) * (DOOR_Z - z));
}

/* The same for whichever goal this is. */
static inline float sm64_goal_distance(const SM64* env, float x, float z) {
    if (!env->star) {
        return sm64_door_distance(x, z);
    }
    /* Only Bob-omb Battlefield's act 1 star has a place written down here. */
    if (env->goal_level != LEVEL_BOB || env->goal_act != 1) {
        return 0.0f;
    }
    return sqrtf((STAR_X - x) * (STAR_X - x) + (STAR_Z - z) * (STAR_Z - z));
}

static inline int sm64_home_level(const SM64* env) {
    return env->star ? env->goal_level : LEVEL_CASTLE_GROUNDS;
}

/* The door, the frame it starts to open, or a star, the frame it is his: the
 * star count going up as he touches it. The level check is only a backstop for
 * the door: a step is two frames, so its action is always seen before the warp
 * inside. */
static int sm64_goal_reached(SM64* env) {
    if (env->star) {
        return n64_s16(&env->gym, env->mario + M_NUM_STARS) > env->stars_at_start;
    }
    uint32_t action = sm64_action(env);
    return action == ACT_PUSHING_DOOR || action == ACT_PULLING_DOOR || sm64_level(env) == LEVEL_CASTLE;
}

/* Whether a star is spawning: time stopped for Mario and the doors (see
 * TIME_STOP_STATE). Not the goal, but sm64_tool reports it, because the demos
 * kept while it was the goal end there. */
static int sm64_star_spawning(SM64* env) {
    return env->star && (n64_s16(&env->gym, TIME_STOP_STATE) & TIME_STOP_MARIO_AND_DOORS) != 0;
}

/* --- novelty ------------------------------------------------------------------
 *
 * Cubes over the whole of a level's space: x and z from -8192 to 8192, y from
 * -4096 to 8192. The visit counts are one table for the process, because the
 * games stepping in parallel are exploring the same grounds, and a cube one of
 * them has worn out is not new to the others. Increments race between threads,
 * so they are atomic.
 */
#define NOVELTY_MIN_CELL 250.0f
#define NOVELTY_MAX_XZ 66 /* 16384 / NOVELTY_MIN_CELL, rounded up */
#define NOVELTY_MAX_Y 50  /* 12288 / NOVELTY_MIN_CELL, rounded up */
#define NOVELTY_CUBES (NOVELTY_MAX_XZ * NOVELTY_MAX_XZ * NOVELTY_MAX_Y)

static uint32_t sm64_cube_visits[NOVELTY_CUBES];
static uint32_t sm64_cubes_explored;

/* Which cube a place is in, or -1 outside the level's space. */
static inline int sm64_cube(const SM64* env, float x, float y, float z) {
    float side = fmaxf(env->novelty_cell, NOVELTY_MIN_CELL);
    int i = (int)floorf((x + 8192.0f) / side);
    int j = (int)floorf((y + 4096.0f) / side);
    int k = (int)floorf((z + 8192.0f) / side);
    if (i < 0 || j < 0 || k < 0 || i >= NOVELTY_MAX_XZ || k >= NOVELTY_MAX_XZ || j >= NOVELTY_MAX_Y) {
        return -1;
    }
    return (j * NOVELTY_MAX_XZ + k) * NOVELTY_MAX_XZ + i;
}

/* Novelty's pay for being where Mario is now: something the first time this
 * episode he is in a cube, nothing after. */
/* --- the archive (Go-Explore) ------------------------------------------------
 *
 * For every cube, the shortest run of pad inputs from the savestate that has
 * reached it. One archive for the process, like the visit counts, behind a
 * lock: the games reset and step in parallel. A run longer than
 * ARCHIVE_EPISODES episodes' worth of frames is not kept, so no reset replays
 * more than that: 2700 frames with the door's 900-frame episodes.
 */
typedef struct SM64Input {
    uint16_t buttons;
    uint16_t frames;
    float stick_x, stick_y;
} SM64Input;

typedef struct {
    SM64Input* inputs;
    int count;
    int frames;
} SM64Cell;

#define ARCHIVE_EPISODES 3

static SM64Cell sm64_archive[NOVELTY_CUBES];
static int sm64_archive_cubes[NOVELTY_CUBES]; /* the cubes that have a run, in the order they got one */
static int sm64_archive_size;
static pthread_mutex_t sm64_archive_lock = PTHREAD_MUTEX_INITIALIZER;

static void sm64_trail_push(SM64* env, uint16_t buttons, int frames, float stick_x, float stick_y) {
    if (env->trail_count == env->trail_capacity) {
        env->trail_capacity = env->trail_capacity ? env->trail_capacity * 2 : 1024;
        env->trail = (SM64Input*)realloc(env->trail, (size_t)env->trail_capacity * sizeof(SM64Input));
    }
    env->trail[env->trail_count++] = (SM64Input){buttons, (uint16_t)frames, stick_x, stick_y};
    env->trail_frames += frames;
}

static void sm64_watch_camera(SM64* env);

/* Play the trail into the game, from the savestate: the start of an episode
 * that begins further on. The cubes it passes through count as this episode's,
 * so a door from here credits them too. (The episode number goes up once reset
 * is done, hence + 1.) They pay no novelty: the policy did not walk them. */
static void sm64_replay_trail(SM64* env) {
    for (int k = 0; k < env->trail_count; k++) {
        const SM64Input* input = &env->trail[k];
        /* Nothing is drawn on the way but the frame the episode starts on. */
        n64gym_draw(&env->gym, k + 1 < env->trail_count ? N64B_GYM_DRAW_NOTHING : N64B_GYM_DRAW_LAST_FRAME);
        n64gym_pad(&env->gym, input->buttons, input->stick_x, input->stick_y);
        if (!n64gym_step(&env->gym, input->frames)) {
            fprintf(stderr, "sm64: the game stopped while replaying a way further on: %s\n", env->gym.error);
            exit(1);
        }
        /* Keep up with the camera, as a step does: otherwise the policy's first
         * stick after the replay is aimed off by wherever the camera had turned. */
        env->had_stick = input->stick_x != 0.0f || input->stick_y != 0.0f;
        if (env->had_stick) {
            float angle = atan2f(input->stick_x, -input->stick_y) * (65536.0f / (2.0f * (float)M_PI));
            env->last_stick_angle = (int)lroundf(angle) & 0xFFFF;
            sm64_watch_camera(env);
        }
        int passed = sm64_cube(env, sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2));
        if (passed >= 0) {
            env->entered[passed] = env->episode + 1;
        }
    }
    n64gym_draw(&env->gym, N64B_GYM_DRAW_LAST_FRAME);
}

/* This episode has just reached a cube: keep how, if it is the first way or a shorter one. */
static void sm64_archive_offer(SM64* env, int cube) {
    if (env->trail_frames > ARCHIVE_EPISODES * env->max_ticks) {
        return;
    }
    pthread_mutex_lock(&sm64_archive_lock);
    SM64Cell* cell = &sm64_archive[cube];
    if (cell->inputs == NULL || env->trail_frames < cell->frames) {
        SM64Input* inputs = (SM64Input*)malloc((size_t)env->trail_count * sizeof(SM64Input));
        memcpy(inputs, env->trail, (size_t)env->trail_count * sizeof(SM64Input));
        if (cell->inputs == NULL) {
            sm64_archive_cubes[sm64_archive_size++] = cube;
        }
        free(cell->inputs);
        cell->inputs = inputs;
        cell->count = env->trail_count;
        cell->frames = env->trail_frames;
    }
    pthread_mutex_unlock(&sm64_archive_lock);
}

/* Start from a cube in the archive, weighted to the ones fewest episodes have
 * entered: copy its run onto this episode's trail and replay it. The game is
 * left where that run left it. Returns the cube, or -1 if the archive is empty. */
/* Which cubes have been on the way to the door, and how many episodes have
 * started from each. Rarity alone was not enough: after 8M steps of starting
 * from the cubes fewest episodes had entered -- far corners of the moat and the
 * lake, mostly -- the door opened once in a thousand, from either kind of
 * start. So once an episode has opened the door, every cube it passed through,
 * the replayed part included, is credited, and a go_explore_door share of
 * archive starts are drawn from those cubes, least started-from first: all the
 * way along some path to the door, from near the start to the step before it.
 * That is still only the reward speaking. Nothing says where the door is. */
static uint32_t sm64_cube_doors[NOVELTY_CUBES];
static uint32_t sm64_cube_starts[NOVELTY_CUBES];
static int sm64_door_cubes[NOVELTY_CUBES];
static int sm64_door_cube_count;

/* Draw a cube from a list, each weighted 1 / sqrt(1 + its count). Call with the lock held. */
static int sm64_draw(SM64* env, const int* cubes, int count, const uint32_t* counts) {
    double total = 0.0;
    for (int k = 0; k < count; k++) {
        total += 1.0 / sqrt(1.0 + (double)counts[cubes[k]]);
    }
    double pick = total * ((double)rand_r(&env->seed) / ((double)RAND_MAX + 1.0));
    for (int k = 0; k < count; k++) {
        pick -= 1.0 / sqrt(1.0 + (double)counts[cubes[k]]);
        if (pick < 0.0) {
            return cubes[k];
        }
    }
    return cubes[count - 1];
}

/* The episode has opened the door: credit every cube it was in. */
static void sm64_credit_door(SM64* env, int door_cube) {
    if (door_cube >= 0) {
        env->entered[door_cube] = env->episode;
    }
    pthread_mutex_lock(&sm64_archive_lock);
    for (int cube = 0; cube < NOVELTY_CUBES; cube++) {
        if (env->entered[cube] != env->episode || sm64_archive[cube].inputs == NULL) {
            continue;
        }
        if (sm64_cube_doors[cube]++ == 0) {
            sm64_door_cubes[sm64_door_cube_count++] = cube;
        }
    }
    pthread_mutex_unlock(&sm64_archive_lock);
}

static int sm64_archive_start(SM64* env) {
    pthread_mutex_lock(&sm64_archive_lock);
    if (sm64_archive_size == 0) {
        pthread_mutex_unlock(&sm64_archive_lock);
        return -1;
    }
    int cube;
    if (sm64_door_cube_count > 0 &&
        (double)rand_r(&env->seed) / ((double)RAND_MAX + 1.0) < env->go_explore_door) {
        cube = sm64_draw(env, sm64_door_cubes, sm64_door_cube_count, sm64_cube_starts);
    } else {
        cube = sm64_draw(env, sm64_archive_cubes, sm64_archive_size, sm64_cube_visits);
    }
    sm64_cube_starts[cube]++;
    SM64Cell* cell = &sm64_archive[cube];
    env->trail_count = 0;
    env->trail_frames = 0;
    for (int k = 0; k < cell->count; k++) {
        const SM64Input* input = &cell->inputs[k];
        sm64_trail_push(env, input->buttons, input->frames, input->stick_x, input->stick_y);
    }
    pthread_mutex_unlock(&sm64_archive_lock);
    sm64_replay_trail(env);
    return cube;
}

/* --- the ghost ------------------------------------------------------------------
 *
 * The clock pays for speed only at the goal, and little: thirty frames off a
 * 770-frame star is 0.02 of reward. What a run does get faster by is the archive, which keeps the shortest way into every
 * cube -- so the archive's way into a cube is a ghost to race, as in a racing
 * game, and beating it is paid where it happens rather than 400 frames later.
 *
 * The first time in an episode Mario enters a cube that has been on the way to
 * the goal, sooner than the archive's way into it, he is paid `ghost` a frame
 * of the difference, GHOST_MOST at most. The archive then keeps his way, so the
 * ghost is as fast as he was and doing the same again pays nothing: only a
 * faster run is ever paid, and a run fifty frames up on the ghost is paid at
 * every cube for as long as it stays ahead.
 *
 * Only on the ground. A cube is 500 units high, so a jump that is going over
 * the side passes through the cubes of the track below it on the way, sooner
 * than anything that slid there; being first into a place he is about to fall
 * out of is not beating anything. (The archive keeps those ways all the same,
 * as it always has.)
 */
#define GHOST_MOST 0.1f

static float sm64_ghost(SM64* env, int cube) {
    if (env->ghost <= 0.0f || (sm64_action(env) & ACT_FLAG_AIR)) {
        return 0.0f;
    }
    float pay = 0.0f;
    pthread_mutex_lock(&sm64_archive_lock);
    const SM64Cell* cell = &sm64_archive[cube];
    if (sm64_cube_doors[cube] > 0 && cell->inputs != NULL && env->trail_frames < cell->frames) {
        pay = fminf(env->ghost * (float)(cell->frames - env->trail_frames), GHOST_MOST);
    }
    pthread_mutex_unlock(&sm64_archive_lock);
    return pay;
}

static float sm64_novelty(SM64* env, float x, float y, float z) {
    /* A place is somewhere Mario could be. Falling out of the world is not: a
     * cube is novelty_cell units, so a fall through empty space enters a fresh
     * one every novelty_cell units down, and novelty would be paying for the
     * fall -- in a course whose only way to lose is going over the side, paying
     * to lose. The archive would keep those cubes too, and few runs fall down
     * the same column, so they would be among the rarest -- which is what it
     * draws first, restarting episodes midway through a fall. */
    if (!sm64_in_the_world(env)) {
        return 0.0f;
    }
    int cube = sm64_cube(env, x, y, z);
    if (cube < 0 || env->entered[cube] == env->episode) {
        return 0.0f;
    }
    env->entered[cube] = env->episode;
    env->cells_entered++;
    uint32_t visits = __sync_add_and_fetch(&sm64_cube_visits[cube], 1);
    if (visits == 1) {
        __sync_add_and_fetch(&sm64_cubes_explored, 1);
    }
    float ghost = 0.0f;
    if (env->go_explore > 0.0f) {
        ghost = sm64_ghost(env, cube); /* against the way on file, before his own replaces it */
        sm64_archive_offer(env, cube);
    }
    float bonus = env->novelty_episode + env->novelty / sqrtf((float)visits);
    env->novelty_earned += bonus;
    env->ghost_earned += ghost;
    return bonus + ghost;
}

/* --- the controller ---------------------------------------------------------
 *
 * The stick is aimed in the world, not on the pad. The game turns a stick
 * direction into a direction to run in by adding the camera's yaw to it
 * (`intendedYaw = atan2s(-stickY, stickX) + camera->yaw`), so a policy that
 * pushed the stick "up" would run wherever the camera happened to be pointing
 * and would have to learn the camera as well as the game.
 *
 * So an action names a direction relative to the way Mario is facing, and the
 * offset between that and the stick is measured rather than looked up: whatever
 * angle we asked for last frame, the game wrote down what it understood by it
 * in `intendedYaw`, and the difference between the two is the camera. It costs
 * one frame of lag on a camera that takes a second to swing, and it needs no
 * address for the camera at all.
 */
static void sm64_aim(SM64* env, int direction, float* stick_x, float* stick_y) {
    if (direction <= 0) {
        *stick_x = 0.0f;
        *stick_y = 0.0f;
        env->had_stick = 0;
        return;
    }
    int wanted = sm64_face_yaw(env) + (direction - 1) * (65536 / STICK_DIRECTIONS);
    int angle = (wanted - env->camera_offset) & 0xFFFF;
    env->last_stick_angle = angle;
    env->had_stick = 1;
    /* The game reads the stick as atan2s(-stickY, stickX): an angle whose sine is
     * x and whose cosine is minus y. Sending cos for y hands it the mirror image
     * of the angle, and a mirror is not an offset -- the measured "camera" then
     * follows Mario's facing instead, "ahead" freezes the stick wherever it was,
     * and every other direction thrashes it. */
    *stick_x = sinf(sm64_radians(angle));
    *stick_y = -cosf(sm64_radians(angle));
}

/* What the game made of the stick we sent, which is the camera plus a constant. */
static void sm64_watch_camera(SM64* env) {
    if (!env->had_stick) {
        return;
    }
    if (n64_f32(&env->gym, env->mario + M_INTENDED_MAG) <= 0.0f) {
        return;
    }
    int intended = n64_s16(&env->gym, env->mario + M_INTENDED_YAW);
    env->camera_offset = (int16_t)(intended - env->last_stick_angle);
}

/* Whether the course asks which act it is being entered for. A secret course
 * does not: it drops Mario straight in and leaves the act at 0. */
static inline int sm64_level_selects_act(int level) { return level != LEVEL_PSS; }

/* --- getting to the castle grounds ------------------------------------------
 *
 * Two presses of Start reach the file select and open the first file; from
 * there the game plays Peach's letter and Lakitu's arrival, which is a minute
 * and a half of cutscene that waits on A for its text boxes. Mashing A through
 * it takes about 1500 frames and ends with Mario standing outside the castle in
 * ACT_IDLE, which is where every episode should start -- so it is done once and
 * saved.
 *
 * Start is deliberately not mashed after the menus: in the game it opens the
 * pause screen, and a game paused is a game that never gets anywhere.
 *
 * For the star, the castle's front door is then pointed at Bob-omb Battlefield
 * (see DOOR_WARP_NODE_LEFT) and Mario is put in front of it with the stick
 * pushed forward. He opens it, the game fades to the act select, A picks act
 * 1 -- the only act a new save has -- and he lands at the start of the course.
 * A secret course has no act select, so there A is not pressed and the act is
 * left where the game put it.
 */
static int sm64_make_state(N64Gym* gym, const char* path, int star, int level, int act, char* error,
                           size_t error_size) {
    uint16_t buttons;
    int in_grounds = 0;
    for (int i = 0; i < 4000 && !in_grounds; i++) {
        /* Start twice, a few seconds apart, then A over and over. */
        if (i == 200 || i == 260) {
            buttons = BUTTON_START;
        } else if (i > 300 && (i % 8) < 2) {
            buttons = BUTTON_A;
        } else {
            buttons = 0;
        }
        n64gym_pad(gym, buttons, 0.0f, 0.0f);
        if (!n64gym_step(gym, 1)) {
            snprintf(error, error_size, "the game stopped while it was being started up: %s",
                     gym->error);
            return 0;
        }
        uint32_t mario = n64_u32(gym, MARIO_STATE_PTR);
        in_grounds = i > 400 && n64_is_ram(mario) && n64_s16(gym, CURR_LEVEL_NUM) == LEVEL_CASTLE_GROUNDS &&
                     n64_u32(gym, mario + M_ACTION) == ACT_IDLE;
    }
    if (!in_grounds) {
        snprintf(error, error_size,
                 "the game never reached the castle grounds; it may be waiting on something");
        return 0;
    }
    /* Let him settle, with nothing held. */
    n64gym_pad(gym, 0, 0.0f, 0.0f);
    if (!n64gym_step(gym, 30)) {
        snprintf(error, error_size, "%s", gym->error);
        return 0;
    }
    if (!star) {
        if (!n64gym_save_state(gym, path)) {
            snprintf(error, error_size, "%s", gym->error);
            return 0;
        }
        return 1;
    }

    if (n64_u32(gym, DOOR_WARP_NODE_LEFT) != DOOR_WARP_TO_CASTLE_LEFT ||
        n64_u32(gym, DOOR_WARP_NODE_RIGHT) != DOOR_WARP_TO_CASTLE_RIGHT) {
        snprintf(error, error_size, "the castle door's warp nodes are not at %08x and %08x",
                 DOOR_WARP_NODE_LEFT, DOOR_WARP_NODE_RIGHT);
        return 0;
    }
    n64_set_u32(gym, DOOR_WARP_NODE_LEFT, 0x00000100 | ((uint32_t)level << 16) | PAINTING_NODE);
    n64_set_u32(gym, DOOR_WARP_NODE_RIGHT, 0x01000100 | ((uint32_t)level << 16) | PAINTING_NODE);
    uint32_t mario = n64_u32(gym, MARIO_STATE_PTR);
    n64_set_f32(gym, mario + M_POS, -76.0f); /* in front of the left half, facing it */
    n64_set_f32(gym, mario + M_POS + 4, 803.0f);
    n64_set_f32(gym, mario + M_POS + 8, -2900.0f);

    int selects_act = sm64_level_selects_act(level);
    int spawned = 0, landed = 0;
    for (int i = 0; i < 1500 && !landed; i++) {
        int now_in = n64_s16(gym, CURR_LEVEL_NUM);
        /* Up on the stick until the door has him, then A for the act select --
         * but not once he has appeared in the course, where A is a jump, and
         * not at all for a course that never asks. */
        n64gym_pad(gym, (selects_act && now_in == level && !spawned && (i % 16) < 2) ? BUTTON_A : 0,
                   0.0f, now_in == LEVEL_CASTLE_GROUNDS ? 1.0f : 0.0f);
        if (!n64gym_step(gym, 1)) {
            snprintf(error, error_size, "the game stopped on the way to level %d: %s", level, gym->error);
            return 0;
        }
        /* The act the save file could not offer, written over the one it did,
         * every frame of the load: the level script reads it when it spawns the
         * course's objects, and that is what decides which star is there. */
        if (selects_act && n64_s16(gym, CURR_LEVEL_NUM) == level) {
            n64_set_s16(gym, CURR_ACT_NUM, (int16_t)act);
        }
        mario = n64_u32(gym, MARIO_STATE_PTR);
        uint32_t action = n64_is_ram(mario) ? n64_u32(gym, mario + M_ACTION) : 0;
        spawned = spawned || (n64_s16(gym, CURR_LEVEL_NUM) == level && (action & ACT_FLAG_AIR));
        landed = spawned && action == ACT_IDLE;
    }
    if (!landed || n64_s16(gym, CURR_LEVEL_NUM) != level ||
        (selects_act && n64_s16(gym, CURR_ACT_NUM) != act)) {
        snprintf(error, error_size,
                 "the door never left Mario standing in level %d act %d (level %d course %d act %d)", level,
                 act, n64_s16(gym, CURR_LEVEL_NUM), n64_s16(gym, CURR_COURSE_NUM),
                 n64_s16(gym, CURR_ACT_NUM));
        return 0;
    }
    /* He lands with the course's opening camera still holding the level: Mario
     * does not move, whatever is held, until a button is pressed after about
     * two seconds of it, and that press is spent on the camera. A state saved
     * before then loads into a game where he never moves at all. So B is
     * pressed every half second, and a state is kept only once, put back and
     * dropped from the air, Mario falls. */
    for (int tries = 0; tries < 30; tries++) {
        n64gym_pad(gym, BUTTON_B, 0.0f, 0.0f);
        n64gym_step(gym, 2);
        n64gym_pad(gym, 0, 0.0f, 0.0f);
        for (int wait = 0; wait < 90 && (wait < 14 || n64_u32(gym, mario + M_ACTION) != ACT_IDLE); wait++) {
            n64gym_step(gym, 1);
        }
        if (!n64gym_save_state(gym, path) || !n64gym_load_state(gym, path)) {
            snprintf(error, error_size, "%s", gym->error);
            return 0;
        }
        float y = n64_f32(gym, mario + M_POS + 4);
        n64_set_f32(gym, mario + M_POS + 4, y + 300.0f);
        n64gym_step(gym, 1);
        int falls = (n64_u32(gym, mario + M_ACTION) & ACT_FLAG_AIR) != 0;
        if (!n64gym_load_state(gym, path)) {
            snprintf(error, error_size, "%s", gym->error);
            return 0;
        }
        if (falls) {
            return 1;
        }
    }
    snprintf(error, error_size, "Mario landed in the course but never came back to life in a saved state");
    return 0;
}

/* --- the environment --------------------------------------------------------- */

static const char* sm64_setting(const char* name, const char* fallback) {
    const char* found = getenv(name);
    return (found != NULL && found[0] != '\0') ? found : fallback;
}

/* One goal for the process, like the archive: the door, or SM64_GOAL=star. */
static int sm64_star_goal(void) { return strcmp(sm64_setting("SM64_GOAL", "door"), "star") == 0; }

/* Which course the star is in, and which act of it. The defaults are the course
 * and act a new save file can reach on its own; see the star. */
static int sm64_star_level(void) {
    int level = atoi(sm64_setting("SM64_LEVEL", "9"));
    return level > 0 ? level : LEVEL_BOB;
}

static int sm64_star_act(void) {
    if (!sm64_level_selects_act(sm64_star_level())) {
        return 0; /* what the game keeps for a secret course */
    }
    int act = atoi(sm64_setting("SM64_ACT", "1"));
    return act >= 1 && act <= 6 ? act : 1;
}

/* A file per course and act, so two of them do not overwrite each other's state
 * or demo. The first stays where it was -- bob-omb-battlefield.state,
 * star.demo -- and the rest are named beside it: star-level24-act2.state.
 * Worked out once, because every episode asks for the state. */
static void sm64_goal_path(const char* path, int level, int act, char* out, size_t size) {
    const char* dot = strrchr(path, '.');
    const char* slash = strrchr(path, '/');
    size_t directory = slash != NULL ? (size_t)(slash - path) + 1 : 0;
    snprintf(out, size, "%.*sstar-level%d-act%d%s", (int)directory, path, level, act,
             dot != NULL && (slash == NULL || dot > slash) ? dot : "");
}

static char sm64_star_state[1024];
static char sm64_star_demo[1024];       /* the fastest touch of the star on file */
static char sm64_star_spawn_demo[1024]; /* the fastest spawn, kept while that was the goal */
static pthread_once_t sm64_paths_once = PTHREAD_ONCE_INIT;

/* star-level27-act0.demo -> star-level27-act0-touch.demo. A star's demo counts
 * the frames to the touch; the ones kept while the spawn was the goal stop a
 * hundred or more frames short of it, so the two are kept apart and never
 * compared: a touch of any length beats no touch. */
static void sm64_touch_path(const char* path, char* out, size_t size) {
    const char* dot = strrchr(path, '.');
    const char* slash = strrchr(path, '/');
    if (dot == NULL || (slash != NULL && dot < slash)) {
        snprintf(out, size, "%s-touch", path);
    } else {
        snprintf(out, size, "%.*s-touch%s", (int)(dot - path), path, dot);
    }
}

static void sm64_work_out_paths(void) {
    const char* state = sm64_setting("SM64_STAR_STATE", SM64_STAR_STATE);
    const char* demo = sm64_setting("SM64_STAR_DEMO", SM64_STAR_DEMO);
    int level = sm64_star_level(), act = sm64_star_act();
    if (level == LEVEL_BOB && act == 1) {
        snprintf(sm64_star_state, sizeof(sm64_star_state), "%s", state);
        snprintf(sm64_star_spawn_demo, sizeof(sm64_star_spawn_demo), "%s", demo);
    } else {
        sm64_goal_path(state, level, act, sm64_star_state, sizeof(sm64_star_state));
        sm64_goal_path(demo, level, act, sm64_star_spawn_demo, sizeof(sm64_star_spawn_demo));
    }
    sm64_touch_path(sm64_star_spawn_demo, sm64_star_demo, sizeof(sm64_star_demo));
}

static const char* sm64_state_path(void) {
    if (!sm64_star_goal()) {
        return sm64_setting("SM64_STATE", SM64_STATE);
    }
    pthread_once(&sm64_paths_once, sm64_work_out_paths);
    return sm64_star_state;
}

/* --- the fastest door (Go-Explore's second phase) ----------------------------
 *
 * The archive reaches the door by replaying inputs, which works only because
 * the game is deterministic. A policy started from the castle grounds has never
 * seen most of that way, so the second phase trains one to repeat it:
 * robustifying, with the backward algorithm (Salimans and Chen, 2018), which is
 * what Go-Explore used.
 *
 * Every exploring run (go_explore on) that opens the door offers its inputs,
 * from the savestate, to the demo file (SM64_DEMO), which keeps the fastest
 * door any of them has found -- and so does every robustifying run (backward
 * on), so a policy that beats its demo leaves a faster one for the next run.
 * A `backward` share of episodes start on that demo
 * at the frontier -- the game as the demo left it that many frames before the
 * door -- and play from there. The frontier begins backward_step frames before
 * the door. Once backward_rate of
 * a window of episodes started from it open the door, it moves backward_step
 * frames further back, until it is the start of the demo. The rest of the
 * episodes start where the task does, so start_perf / from_start is still the
 * door rate that counts.
 *
 * Half the episodes on the demo start at a frontier already passed, nearer the
 * door, instead, and do not count toward moving it, so what the policy has
 * already learned keeps being paid for. Without them, the first run's frontier
 * went from 30 frames to 180 in 165K steps, into Lakitu's speech (348 to 102
 * frames before the door in that demo), and stopped there. Every episode on
 * the demo then started where the policy could not yet win, and by 200K steps
 * it played at random (entropy 4.3 of 4.9) and opened nothing.
 *
 * The demo a run robustifies is the one on file when it starts.
 */
#define SM64_DEMO_MAGIC "SM64DEMO"
#define BACKWARD_WINDOW 32
#define DEMO_SLACK 300 /* frames: ten seconds more than the demo took */

typedef struct {
    char magic[8];
    int32_t count;  /* inputs */
    int32_t frames; /* from the savestate to the frame the door starts to open */
} SM64DemoHeader;

/* A run's inputs, hashed (FNV-1a): the same run found twice hashes the same. */
static uint32_t sm64_inputs_hash(const SM64Input* inputs, int count) {
    uint32_t hash = 2166136261u;
    const uint8_t* byte = (const uint8_t*)inputs;
    for (size_t k = 0; k < (size_t)count * sizeof(SM64Input); k++) {
        hash = (hash ^ byte[k]) * 16777619u;
    }
    return hash;
}

static SM64Input* sm64_demo;      /* what this run robustifies, read once */
static int sm64_demo_count, sm64_demo_frames;
static int sm64_demo_best = -1;   /* frames of the fastest door on file, once it has been looked at */
static int sm64_frontier;         /* frames before the door */
static int sm64_frontier_tries, sm64_frontier_doors;
static pthread_mutex_t sm64_demo_lock = PTHREAD_MUTEX_INITIALIZER;

static const char* sm64_demo_path(void) {
    if (!sm64_star_goal()) {
        return sm64_setting("SM64_DEMO", SM64_DEMO);
    }
    pthread_once(&sm64_paths_once, sm64_work_out_paths);
    return sm64_star_demo;
}

/* A demo's inputs, or NULL if the file has none. */
static SM64Input* sm64_demo_read(const char* path, int* count, int* frames) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    SM64DemoHeader header;
    SM64Input* inputs = NULL;
    if (fread(&header, sizeof(header), 1, file) == 1 &&
        memcmp(header.magic, SM64_DEMO_MAGIC, sizeof(header.magic)) == 0 && header.count > 0) {
        inputs = (SM64Input*)malloc((size_t)header.count * sizeof(SM64Input));
        if (fread(inputs, sizeof(SM64Input), (size_t)header.count, file) != (size_t)header.count) {
            free(inputs);
            inputs = NULL;
        }
    }
    fclose(file);
    if (inputs != NULL) {
        *count = header.count;
        *frames = header.frames;
    }
    return inputs;
}

/* Write a run of inputs as a demo. Written beside the file and renamed over it,
 * so a reader never sees half. `writer` keeps two games' files apart. */
static int sm64_demo_write(const char* path, int writer, const SM64Input* inputs, int count, int frames) {
    char temporary[1024];
    snprintf(temporary, sizeof(temporary), "%s.%d", path, writer);
    FILE* file = fopen(temporary, "wb");
    if (file == NULL) {
        return 0;
    }
    SM64DemoHeader header = {.count = count, .frames = frames};
    memcpy(header.magic, SM64_DEMO_MAGIC, sizeof(header.magic));
    int written = fwrite(&header, sizeof(header), 1, file) == 1 &&
                  fwrite(inputs, sizeof(SM64Input), (size_t)count, file) == (size_t)count;
    return fclose(file) == 0 && written && rename(temporary, path) == 0;
}

/* This episode has opened the door: if no run on file was faster, it is the demo now. */
static void sm64_demo_offer(SM64* env) {
    pthread_mutex_lock(&sm64_demo_lock);
    const char* path = sm64_demo_path();
    if (sm64_demo_best < 0) {
        int count, frames;
        SM64Input* on_file = sm64_demo_read(path, &count, &frames);
        sm64_demo_best = on_file != NULL ? frames : INT_MAX;
        free(on_file);
    }
    if (env->trail_frames < sm64_demo_best) {
        if (sm64_demo_write(path, (int)env->rng, env->trail, env->trail_count, env->trail_frames)) {
            sm64_demo_best = env->trail_frames;
        } else {
            fprintf(stderr, "sm64: could not keep a door of %d frames in %s\n", env->trail_frames, path);
        }
    }
    pthread_mutex_unlock(&sm64_demo_lock);
}

/* --- the fastest runs, to watch ----------------------------------------------
 *
 * The demo keeps one run, and any slower way to the goal is gone when its
 * episode ends -- and with it the archive's cube it started from, so nothing can
 * play it again. So every episode that reaches the goal, exploring or not, also
 * offers its inputs to a folder beside the demo (star-level27-act0.demo ->
 * star-level27-act0-fastest/), which keeps the FASTEST_RUNS fastest. A file is
 * named for its frames and a hash of its inputs, 01532-9a3c1e2b.demo, so the
 * folder lists fastest first and the same run found twice is kept once.
 * `sm64_tool replay watch <file>` plays one.
 */
#ifndef FASTEST_RUNS
#define FASTEST_RUNS 10
#endif

static char sm64_fastest_dir[1024];
static char sm64_fastest[FASTEST_RUNS + 1][32]; /* file names, fastest first */
static int sm64_fastest_frames[FASTEST_RUNS + 1];
static int sm64_fastest_count = -1; /* until the folder has been read */

/* Put a run in the list, fastest first. With one too many, the slowest leaves the
 * list and the folder. */
static void sm64_fastest_insert(const char* name, int frames) {
    int k = sm64_fastest_count;
    while (k > 0 && sm64_fastest_frames[k - 1] > frames) {
        sm64_fastest_frames[k] = sm64_fastest_frames[k - 1];
        memcpy(sm64_fastest[k], sm64_fastest[k - 1], sizeof(sm64_fastest[k]));
        k--;
    }
    sm64_fastest_frames[k] = frames;
    snprintf(sm64_fastest[k], sizeof(sm64_fastest[k]), "%s", name);
    if (++sm64_fastest_count > FASTEST_RUNS) {
        char path[1100];
        snprintf(path, sizeof(path), "%s/%s", sm64_fastest_dir, sm64_fastest[FASTEST_RUNS]);
        remove(path);
        sm64_fastest_count = FASTEST_RUNS;
    }
}

/* What an earlier run left in the folder, once for the process. */
static void sm64_fastest_read(void) {
    const char* demo = sm64_demo_path();
    size_t stem = strlen(demo);
    if (stem >= 5 && strcmp(demo + stem - 5, ".demo") == 0) {
        stem -= 5;
    }
    snprintf(sm64_fastest_dir, sizeof(sm64_fastest_dir), "%.*s-fastest", (int)stem, demo);
    sm64_fastest_count = 0;
    DIR* dir = opendir(sm64_fastest_dir);
    if (dir == NULL) {
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        size_t length = strlen(entry->d_name);
        int frames;
        if (length < sizeof(sm64_fastest[0]) && length > 5 && strcmp(entry->d_name + length - 5, ".demo") == 0 &&
            sscanf(entry->d_name, "%d-", &frames) == 1) {
            sm64_fastest_insert(entry->d_name, frames);
        }
    }
    closedir(dir);
}

/* This episode has reached the goal: keep it if it is among the fastest. */
static void sm64_fastest_offer(SM64* env) {
    pthread_mutex_lock(&sm64_demo_lock);
    if (sm64_fastest_count < 0) {
        sm64_fastest_read();
    }
    if (sm64_fastest_count < FASTEST_RUNS || env->trail_frames < sm64_fastest_frames[FASTEST_RUNS - 1]) {
        char name[32];
        snprintf(name, sizeof(name), "%05d-%08x.demo", env->trail_frames, sm64_inputs_hash(env->trail, env->trail_count));
        int kept = 0;
        for (int k = 0; k < sm64_fastest_count; k++) {
            kept |= strcmp(sm64_fastest[k], name) == 0;
        }
        char path[1100];
        snprintf(path, sizeof(path), "%s/%s", sm64_fastest_dir, name);
        mkdir(sm64_fastest_dir, 0755);
        if (!kept && sm64_demo_write(path, (int)env->rng, env->trail, env->trail_count, env->trail_frames)) {
            sm64_fastest_insert(name, env->trail_frames);
        } else if (!kept) {
            fprintf(stderr, "sm64: could not keep a run of %d frames in %s\n", env->trail_frames, path);
        }
    }
    pthread_mutex_unlock(&sm64_demo_lock);
}

/* Read the demo to robustify, once for the process. */
static char sm64_demo_states[1100]; /* where the game is kept at each point of the demo episodes start from */

/* Remove every folder of states beside this one for the same demo file -- the
 * states of demos since replaced. Only files named like a state go, and a folder
 * with anything else in it stays. */
static void sm64_remove_other_states(const char* keep) {
    const char* slash = strrchr(keep, '/');
    const char* tag = strstr(slash != NULL ? slash : keep, "-states-");
    if (tag == NULL) {
        return;
    }
    char directory[1100];
    snprintf(directory, sizeof(directory), "%.*s", slash != NULL ? (int)(slash - keep) : 1, slash != NULL ? keep : ".");
    const char* name = slash != NULL ? slash + 1 : keep;
    size_t prefix = (size_t)(tag - name) + strlen("-states-");
    DIR* dir = opendir(directory);
    if (dir == NULL) {
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, name, prefix) != 0 || strcmp(entry->d_name, name) == 0) {
            continue;
        }
        char other[1300];
        snprintf(other, sizeof(other), "%s/%s", directory, entry->d_name);
        DIR* states = opendir(other);
        if (states == NULL) {
            continue;
        }
        struct dirent* state;
        while ((state = readdir(states)) != NULL) {
            int frame;
            if (sscanf(state->d_name, "%d.state", &frame) == 1) {
                char path[1600];
                snprintf(path, sizeof(path), "%s/%s", other, state->d_name);
                remove(path);
            }
        }
        closedir(states);
        rmdir(other);
    }
    closedir(dir);
}

static void sm64_demo_load(SM64* env) {
    pthread_mutex_lock(&sm64_demo_lock);
    if (sm64_demo == NULL) {
        const char* path = sm64_demo_path();
        sm64_demo = sm64_demo_read(path, &sm64_demo_count, &sm64_demo_frames);
        /* A star with no touch on file yet works back along the fastest spawn
         * kept while that was the goal: the same route as far as the box. Its
         * frames end where the star spawns, so an episode from its end has
         * DEMO_SLACK to wait out the cutscene and take the star, and the first
         * touch from anywhere on it is the touch demo. */
        if (sm64_demo == NULL && env->star) {
            path = sm64_star_spawn_demo;
            sm64_demo = sm64_demo_read(path, &sm64_demo_count, &sm64_demo_frames);
            if (sm64_demo != NULL) {
                fprintf(stderr, "sm64: no touch of the star on file yet; working back along the spawn in %s\n", path);
            }
        }
        if (sm64_demo == NULL) {
            fprintf(stderr, "sm64: backward needs a door to work back from, and there is none in %s.\n"
                            "      Explore first: ./build/sm64_tool explore\n", path);
            exit(1);
        }
        int first = env->backward_start > 0 ? env->backward_start : env->backward_step > 0 ? env->backward_step : 1;
        sm64_frontier = first < sm64_demo_frames ? first : sm64_demo_frames;
        /* door.demo -> door-states-9a3c1e2b/, named for this demo's inputs, so
         * another demo's states are never mistaken for its own. The states of
         * demos this one has replaced are 8 MB each and no use now, so they go. */
        size_t stem = strlen(path);
        if (stem >= 5 && strcmp(path + stem - 5, ".demo") == 0) {
            stem -= 5;
        }
        snprintf(sm64_demo_states, sizeof(sm64_demo_states), "%.*s-states-%08x", (int)stem, path,
                 sm64_inputs_hash(sm64_demo, sm64_demo_count));
        sm64_remove_other_states(sm64_demo_states);
    }
    pthread_mutex_unlock(&sm64_demo_lock);
}

/* Put the game where the trail, a prefix of the demo, leads.
 *
 * Replaying it costs as many frames as the demo has before that point: on the
 * slide, 1200 frames of demo for an episode 30 frames from the star that will
 * itself last 330, and the eight games step in lockstep, so one replaying holds
 * the other seven. The game is deterministic, so the first episode to reach a
 * point on the demo saves the game there, in a folder beside the demo, and every
 * episode after loads it -- 8 MB read instead of a second of play. Which game
 * saved it makes no difference to what is in it.
 *
 * A loaded start does not mark the cubes the demo passed through as this
 * episode's, as a replayed one does. That only matters with novelty or the
 * archive on, which robustifying turns off. */
static void sm64_demo_go(SM64* env) {
    if (env->trail_count == 0) {
        return; /* the savestate itself, which reset has just loaded */
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/%05d.state", sm64_demo_states, env->trail_frames);
    struct stat ignored;
    if (stat(path, &ignored) == 0 && n64gym_load_state(&env->gym, path)) {
        env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
        /* The camera, as the last input of the replay would have measured it. */
        const SM64Input* last = &env->trail[env->trail_count - 1];
        env->had_stick = last->stick_x != 0.0f || last->stick_y != 0.0f;
        if (env->had_stick) {
            float angle = atan2f(last->stick_x, -last->stick_y) * (65536.0f / (2.0f * (float)M_PI));
            env->last_stick_angle = (int)lroundf(angle) & 0xFFFF;
            sm64_watch_camera(env);
        }
        n64gym_draw(&env->gym, N64B_GYM_DRAW_LAST_FRAME);
        return;
    }
    sm64_replay_trail(env);
    mkdir(sm64_demo_states, 0755);
    char temporary[1300];
    snprintf(temporary, sizeof(temporary), "%s.%d", path, (int)env->rng);
    if (!n64gym_save_state(&env->gym, temporary) || rename(temporary, path) != 0) {
        fprintf(stderr, "sm64: could not keep the game at frame %d of the demo in %s\n", env->trail_frames, path);
        remove(temporary);
    }
}

/* Start on the demo: at the frontier, or, half the time, at any of the
 * frontiers already passed, nearer the door. Starts are on the frontier's grid
 * of backward_step frames, so there are as many places to start as there are
 * frontiers, and as many states to keep (see sm64_demo_go). */
static void sm64_demo_start(SM64* env) {
    pthread_mutex_lock(&sm64_demo_lock);
    int frontier = sm64_frontier;
    pthread_mutex_unlock(&sm64_demo_lock);
    int step = env->backward_step > 0 ? env->backward_step : 1;
    int before = frontier;
    env->start_frontier = frontier;
    if (rand_r(&env->seed) % 2 == 0) {
        /* Rehearsal: does not count toward moving the frontier. */
        int passed = frontier / step > 0 ? frontier / step : 1;
        before = step * (1 + (int)(rand_r(&env->seed) % (unsigned)passed));
        if (before > frontier) {
            before = frontier;
        }
        env->start_frontier = -1;
    }
    env->trail_count = 0;
    env->trail_frames = 0;
    for (int k = 0; k < sm64_demo_count; k++) {
        if (env->trail_frames + sm64_demo[k].frames > sm64_demo_frames - before) {
            break;
        }
        sm64_trail_push(env, sm64_demo[k].buttons, sm64_demo[k].frames, sm64_demo[k].stick_x,
                        sm64_demo[k].stick_y);
    }
    sm64_demo_go(env);

    /* The clock starts partway, so an episode has as long to reach the door as
     * the demo took from here plus DEMO_SLACK, never more than the whole clock,
     * and one that will not make it is over in about that long.
     *
     * The second run started every one with the whole clock, 900 frames to get
     * from a few seconds out. Most episodes ran all of it and paid -1, entropy
     * fell to 0.01 by 260K steps, and the frontier never got past 120. The runs
     * after that set the clock to what the demo's read at that point, and the
     * two that trained stably stuck at 570: that far back, the explorer's
     * wandering at the start of the demo had already spent a hundred frames,
     * so an episode there had less time than one from the real start. */
    int budget = sm64_demo_frames - env->trail_frames + DEMO_SLACK;
    env->start_tick = budget < env->max_ticks ? env->max_ticks - budget : 0;
}

/* An episode that began on the demo is over: count it toward moving the frontier
 * back, if the frontier is still where it began. */
static void sm64_demo_result(SM64* env, int door) {
    pthread_mutex_lock(&sm64_demo_lock);
    if (env->start_frontier == sm64_frontier && sm64_frontier < sm64_demo_frames) {
        sm64_frontier_tries++;
        sm64_frontier_doors += door;
        if (sm64_frontier_tries >= BACKWARD_WINDOW) {
            if ((float)sm64_frontier_doors >= env->backward_rate * (float)sm64_frontier_tries) {
                sm64_frontier += env->backward_step > 0 ? env->backward_step : 1;
                if (sm64_frontier > sm64_demo_frames) {
                    sm64_frontier = sm64_demo_frames;
                }
            }
            sm64_frontier_tries = 0;
            sm64_frontier_doors = 0;
        }
    }
    pthread_mutex_unlock(&sm64_demo_lock);
}

/* --- an archive that starts with the runs on file ------------------------------
 *
 * The archive is the process's, so every run began with none and spent its
 * first ten minutes finding the way down the slide again -- and two runs' finds
 * never met: the fastest way over the top was in one run's archive and the best
 * jump off the upper track in another's, and no episode could start on the one
 * and make the other.
 *
 * With go_explore_seed on, the first game to start plays every run on file once
 * -- the demo, the folder of the fastest, and any kept by hand in a folder
 * beside them, <demo>-seeds/ -- and offers each cube on the way
 * to the archive, as the episode that found it did, then credits those cubes as
 * on the way to the goal, which they are. The archive keeps the shortest way to
 * a cube, so what a run starts with is the best of all of them at every place,
 * and half its archive starts are along them from the first episode.
 *
 * A run stops being offered the step it reaches the goal, as an episode does:
 * a cube kept after that would start episodes with the star already his.
 */
static int sm64_seeded;
static pthread_mutex_t sm64_seed_lock = PTHREAD_MUTEX_INITIALIZER;

/* Play one run on file into the archive. Returns whether the file had a run in it. */
static int sm64_seed_from(SM64* env, const char* path) {
    int count, frames;
    SM64Input* inputs = sm64_demo_read(path, &count, &frames);
    if (inputs == NULL) {
        return 0;
    }
    if (!n64gym_load_state(&env->gym, sm64_state_path())) {
        fprintf(stderr, "sm64: could not put the game back to play %s: %s\n", path, env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    env->trail_count = 0;
    env->trail_frames = 0;
    env->episode++; /* the cubes this run enters are marked as its own */
    int reached = 0;
    n64gym_draw(&env->gym, N64B_GYM_DRAW_NOTHING);
    for (int k = 0; k < count; k++) {
        n64gym_pad(&env->gym, inputs[k].buttons, inputs[k].stick_x, inputs[k].stick_y);
        if (!n64gym_step(&env->gym, inputs[k].frames)) {
            fprintf(stderr, "sm64: the game stopped while playing %s: %s\n", path, env->gym.error);
            exit(1);
        }
        sm64_trail_push(env, inputs[k].buttons, inputs[k].frames, inputs[k].stick_x, inputs[k].stick_y);
        if (sm64_goal_reached(env)) {
            reached = 1;
            break;
        }
        int cube = sm64_in_the_world(env) ? sm64_cube(env, sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2)) : -1;
        if (cube < 0 || env->entered[cube] == env->episode) {
            continue;
        }
        env->entered[cube] = env->episode;
        sm64_archive_offer(env, cube);
    }
    if (reached) {
        sm64_credit_door(env, -1);
    } else {
        fprintf(stderr, "sm64: %s does not reach the goal when played; its cubes are kept, not credited\n", path);
    }
    free(inputs);
    return 1;
}

static void sm64_archive_seed(SM64* env) {
    pthread_mutex_lock(&sm64_seed_lock);
    if (!sm64_seeded) {
        sm64_seeded = 1;
        char names[FASTEST_RUNS][32];
        int listed;
        pthread_mutex_lock(&sm64_demo_lock);
        if (sm64_fastest_count < 0) {
            sm64_fastest_read();
        }
        listed = sm64_fastest_count < FASTEST_RUNS ? sm64_fastest_count : FASTEST_RUNS;
        memcpy(names, sm64_fastest, sizeof(names));
        pthread_mutex_unlock(&sm64_demo_lock);

        int runs = sm64_seed_from(env, sm64_demo_path());
        for (int k = 0; k < listed; k++) {
            char path[1100];
            snprintf(path, sizeof(path), "%s/%s", sm64_fastest_dir, names[k]);
            runs += sm64_seed_from(env, path);
        }
        /* And any run kept by hand beside them, in <demo>-seeds/. The folder of
         * the fastest drops a run once ten are faster, and with it whatever only
         * that run had: a way over the top, a line through a turn. */
        char seeds[1100];
        const char* demo = sm64_demo_path();
        size_t stem = strlen(demo);
        if (stem >= 5 && strcmp(demo + stem - 5, ".demo") == 0) {
            stem -= 5;
        }
        snprintf(seeds, sizeof(seeds), "%.*s-seeds", (int)stem, demo);
        DIR* dir = opendir(seeds);
        if (dir != NULL) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL) {
                size_t length = strlen(entry->d_name);
                if (length > 5 && strcmp(entry->d_name + length - 5, ".demo") == 0) {
                    char path[1400];
                    snprintf(path, sizeof(path), "%s/%s", seeds, entry->d_name);
                    runs += sm64_seed_from(env, path);
                }
            }
            closedir(dir);
        }
        fprintf(stderr, "sm64: the archive starts with %d cubes, %d of them on the way to the goal, from %d runs on file\n",
                sm64_archive_size, sm64_door_cube_count, runs);
        n64gym_draw(&env->gym, N64B_GYM_DRAW_LAST_FRAME);
        if (!n64gym_load_state(&env->gym, sm64_state_path())) {
            fprintf(stderr, "sm64: could not put the game back after seeding the archive: %s\n", env->gym.error);
            exit(1);
        }
        env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
        env->trail_count = 0;
        env->trail_frames = 0;
    }
    pthread_mutex_unlock(&sm64_seed_lock);
}

void init(SM64* env) {
    N64GymOptions options = {
        .host = sm64_setting("SM64_HOST", SM64_HOST),
        .module = sm64_setting("SM64_MODULE", SM64_MODULE),
        .rom = sm64_setting("SM64_ROM", SM64_ROM),
        .game_id = "NSME",
        .config_dir = sm64_setting("SM64_CONFIG_DIR", SM64_CONFIG_DIR),
        .log = sm64_setting("SM64_LOG", SM64_LOG),
        .windowed = env->window,
        .picture = env->picture_width > 0,
        .index = (int)env->rng,
    };
    if (!n64gym_open(&env->gym, &options)) {
        fprintf(stderr, "sm64: %s\n", env->gym.error);
        fprintf(stderr, "      host   %s\n      module %s\n      rom    %s\n", options.host,
                options.module, options.rom);
        exit(1);
    }
    env->opened = 1;
    /* Only the frame at the end of a step is ever seen, and Super Mario 64 draws
     * every frame from nothing, so the others are not drawn. */
    n64gym_draw(&env->gym, N64B_GYM_DRAW_LAST_FRAME);
    env->star = sm64_star_goal();
    env->goal_level = sm64_star_level();
    env->goal_act = sm64_star_act();
    env->entered = (uint32_t*)calloc(NOVELTY_CUBES, sizeof(uint32_t));
    env->seed = 0x9E3779B9u ^ (env->rng * 2654435761u);
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    if (env->backward > 0.0f) {
        sm64_demo_load(env);
    }

    const char* state = sm64_state_path();
    struct stat ignored;
    if (stat(state, &ignored) != 0) {
        /* Nobody has played through the intro yet. Do it once, here, and every
         * episode of every run from now on starts from it. */
        char error[256];
        fprintf(stderr, "sm64: playing through the intro once to make %s\n", state);
        if (!sm64_make_state(&env->gym, state, env->star, env->goal_level, env->goal_act, error,
                             sizeof(error))) {
            fprintf(stderr, "sm64: %s\n", error);
            exit(1);
        }
    }
    /* Once in a while a game that has just booted is not yet in the shape every
     * state is taken in, and says so (one start in about twenty, of eight games
     * drawing pictures at once). A frame later it is. */
    int loaded = n64gym_load_state(&env->gym, state);
    for (int tries = 0; !loaded && tries < 30; tries++) {
        n64gym_step(&env->gym, 1);
        loaded = n64gym_load_state(&env->gym, state);
    }
    if (!loaded) {
        fprintf(stderr, "sm64: could not start from %s: %s\n", state, env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    env->stars_at_start = n64_s16(&env->gym, env->mario + M_NUM_STARS);
    if (sm64_level(env) != sm64_home_level(env)) {
        fprintf(stderr, "sm64: %s does not start where this goal does; delete it to make it again\n", state);
        exit(1);
    }
    if (env->star && n64_s16(&env->gym, CURR_ACT_NUM) != env->goal_act) {
        fprintf(stderr, "sm64: %s is act %d and SM64_ACT asks for %d; delete it to make it again\n", state,
                n64_s16(&env->gym, CURR_ACT_NUM), env->goal_act);
        exit(1);
    }
    if (env->go_explore > 0.0f && env->go_explore_seed) {
        sm64_archive_seed(env);
    }
}

/* The game's picture, shrunk to picture_width by picture_height: each pixel is
 * the average of the block of the frame it covers. Black while there is no
 * picture, which is after a load until a frame has been drawn. */
static void sm64_write_picture(SM64* env, float* out) {
    int width, height;
    const uint8_t* picture = n64gym_picture(&env->gym, &width, &height);
    int size = env->picture_width * env->picture_height * PICTURE_CHANNELS;
    if (picture == NULL) {
        memset(out, 0, (size_t)size * sizeof(float));
        return;
    }
    for (int y = 0; y < env->picture_height; y++) {
        int top = y * height / env->picture_height;
        int bottom = (y + 1) * height / env->picture_height;
        bottom = bottom > top ? bottom : top + 1;
        for (int x = 0; x < env->picture_width; x++) {
            int left = x * width / env->picture_width;
            int right = (x + 1) * width / env->picture_width;
            right = right > left ? right : left + 1;
            unsigned sum[PICTURE_CHANNELS] = {0};
            for (int row = top; row < bottom; row++) {
                const uint8_t* pixel = picture + 4 * (row * width + left);
                for (int column = left; column < right; column++, pixel += 4) {
                    for (int c = 0; c < PICTURE_CHANNELS; c++) {
                        sum[c] += pixel[c];
                    }
                }
            }
            float scale = 1.0f / (255.0f * (float)((bottom - top) * (right - left)));
            for (int c = 0; c < PICTURE_CHANNELS; c++) {
                *out++ = (float)sum[c] * scale;
            }
        }
    }
}

static void sm64_write_observations(SM64* env, float speed) {
    obs_t* obs = env->agents[0].observations;
    N64Gym* gym = &env->gym;
    uint32_t action = sm64_action(env);

    if (env->picture_width > 0) {
        sm64_write_picture(env, obs + NUM_OBS);
    }
    memset(obs, 0, NUM_OBS * sizeof(float));
    /* The clock is the env's, not the game's, so it is there with state off too:
     * the reward is made of it, and a critic that cannot see it cannot predict
     * even an episode that only runs out. */
    obs[OBS_TIME_LEFT] = 1.0f - (float)env->tick / (float)env->max_ticks;
    if (!env->state) {
        return;
    }
    obs[OBS_POS_X] = sm64_pos(env, 0) / 4000.0f;
    obs[OBS_POS_Y] = sm64_pos(env, 1) / 2000.0f;
    obs[OBS_POS_Z] = sm64_pos(env, 2) / 4000.0f;
    obs[OBS_VEL_X] = sm64_vel(env, 0) / 60.0f;
    obs[OBS_VEL_Y] = sm64_vel(env, 1) / 60.0f;
    obs[OBS_VEL_Z] = sm64_vel(env, 2) / 60.0f;
    obs[OBS_FORWARD_VEL] = sm64_forward_vel(env) / 60.0f;
    obs[OBS_SPEED] = speed / 60.0f;

    int yaw = sm64_face_yaw(env);
    obs[OBS_FACE_SIN] = sinf(sm64_radians(yaw));
    obs[OBS_FACE_COS] = cosf(sm64_radians(yaw));
    obs[OBS_PITCH] = (float)n64_s16(gym, env->mario + M_FACE_ANGLE) / 32768.0f;
    /* Where the camera is, relative to the way he faces: the same number the
     * stick is aimed with, so the policy can see the lag in it. */
    obs[OBS_CAMERA_SIN] = sinf(sm64_radians(env->camera_offset - yaw));
    obs[OBS_CAMERA_COS] = cosf(sm64_radians(env->camera_offset - yaw));

    float y = sm64_pos(env, 1);
    float floor_height = n64_f32(gym, env->mario + M_FLOOR_HEIGHT);
    float ceil_height = n64_f32(gym, env->mario + M_CEIL_HEIGHT);
    obs[OBS_HEIGHT_ABOVE_FLOOR] = fminf((y - floor_height) / 500.0f, 4.0f);
    uint32_t floor = n64_u32(gym, env->mario + M_FLOOR);
    obs[OBS_FLOOR_SLOPE] = n64_is_ram(floor) ? n64_f32(gym, floor + SURFACE_NORMAL_Y) : 0.0f;
    obs[OBS_HEADROOM] = fminf((ceil_height - y) / 1000.0f, 4.0f);
    obs[OBS_TOUCHING_WALL] = n64_is_ram(n64_u32(gym, env->mario + M_WALL)) ? 1.0f : 0.0f;
    obs[OBS_WATER] = fmaxf(fminf(((float)n64_s16(gym, env->mario + M_WATER_LEVEL) - y) / 500.0f,
                                 2.0f), -2.0f);
    obs[OBS_ON_GROUND] = (action & ACT_FLAG_AIR) ? 0.0f : 1.0f;

    obs[OBS_GROUP_STATIONARY + ((action & ACT_GROUP_MASK) >> 6)] = 1.0f;
    obs[OBS_ACTION_TIMER] = fminf((float)n64_u16(gym, env->mario + M_ACTION_TIMER) / 60.0f, 4.0f);
    obs[OBS_ACTION_STATE] = fminf((float)n64_u16(gym, env->mario + M_ACTION_STATE) / 8.0f, 4.0f);
    obs[OBS_HEALTH] = (float)n64_s16(gym, env->mario + M_HEALTH) / 2176.0f;

    obs[OBS_FLAG_STATIONARY] = (action & ACT_FLAG_STATIONARY) ? 1.0f : 0.0f;
    obs[OBS_FLAG_MOVING] = (action & ACT_FLAG_MOVING) ? 1.0f : 0.0f;
    obs[OBS_FLAG_AIR] = (action & ACT_FLAG_AIR) ? 1.0f : 0.0f;
    obs[OBS_FLAG_SWIMMING] = (action & (ACT_FLAG_SWIMMING | ACT_FLAG_METAL_WATER)) ? 1.0f : 0.0f;
    obs[OBS_FLAG_SLIDING] = (action & ACT_FLAG_BUTT_OR_STOMACH_SLIDE) ? 1.0f : 0.0f;
    obs[OBS_FLAG_DIVING] = (action & ACT_FLAG_DIVING) ? 1.0f : 0.0f;
    obs[OBS_FLAG_POLE] = (action & ACT_FLAG_ON_POLE) ? 1.0f : 0.0f;
    obs[OBS_FLAG_HANGING] = (action & ACT_FLAG_HANGING) ? 1.0f : 0.0f;
    obs[OBS_FLAG_IDLE] = (action & ACT_FLAG_IDLE) ? 1.0f : 0.0f;
    obs[OBS_FLAG_ATTACKING] = (action & ACT_FLAG_ATTACKING) ? 1.0f : 0.0f;
}

/* Required function */
void puf_reset(SM64* env) {
    if (!n64gym_load_state(&env->gym, sm64_state_path())) {
        fprintf(stderr, "sm64: could not put the game back: %s\n", env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    env->camera_offset = 0;
    env->had_stick = 0;
    env->trail_count = 0;
    env->trail_frames = 0;
    env->from_start = 1;
    env->replayed_frames = 0;

    env->start_cube = -1;
    env->start_frontier = -1;
    env->start_tick = 0;
    if (env->backward > 0.0f && (double)rand_r(&env->seed) / ((double)RAND_MAX + 1.0) < env->backward) {
        sm64_demo_start(env);
        env->from_start = 0;
        env->replayed_frames = env->trail_frames;
    } else if (env->go_explore > 0.0f && (double)rand_r(&env->seed) / ((double)RAND_MAX + 1.0) < env->go_explore &&
        (env->start_cube = sm64_archive_start(env)) >= 0) {
        env->from_start = 0;
        env->replayed_frames = env->trail_frames;
    } else if (env->random_start > 0) {
        /* A few frames of some direction, so that not every episode is the same
         * episode. Without it there is one starting state and the policy can
         * learn one trajectory through it. */
        int frames = (int)(rand_r(&env->seed) % (unsigned)(env->random_start + 1));
        if (frames > 0) {
            float x, y;
            sm64_aim(env, 1 + (int)(rand_r(&env->seed) % STICK_DIRECTIONS), &x, &y);
            n64gym_pad(&env->gym, 0, x, y);
            n64gym_step(&env->gym, frames);
            sm64_trail_push(env, 0, frames, x, y);
            sm64_watch_camera(env);
        }
    }

    env->tick = env->start_tick;
    env->steps = 0;
    env->episode_return = 0.0f;
    env->distance = 0.0f;
    env->top_speed = 0.0f;
    env->forward_vel_sum = 0.0f;
    env->airborne_frames = 0;
    env->dialog_frames = 0;
    env->deaths = 0;
    env->last_x = sm64_pos(env, 0);
    env->last_z = sm64_pos(env, 2);
    env->closest = sm64_goal_distance(env, env->last_x, env->last_z);
    env->episode++;
    env->novelty_earned = 0.0f;
    env->ghost_earned = 0.0f;
    env->cells_entered = 0;
    sm64_write_observations(env, 0.0f);
}

/* A death, with respawn on: the game goes back to the savestate and the episode
 * carries on, with the clock where the death left it.
 *
 * The trail goes back with it. The trail is every input since the savestate, and
 * it is what the archive keeps as the way to a place -- so after a load, which
 * leaves the game the savestate to the bit, the way to anywhere is the inputs
 * from here and the trail starts empty again. Everything the archive is offered
 * after a death still replays.
 *
 * An episode that began from a cube in the archive does not go back to that
 * cube. It goes back to where the task starts, because that is the state there
 * is, and it keeps whatever is left of its clock. */
static void sm64_respawn(SM64* env) {
    if (!n64gym_load_state(&env->gym, sm64_state_path())) {
        fprintf(stderr, "sm64: could not put the game back after a death: %s\n", env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    env->camera_offset = 0;
    env->had_stick = 0;
    env->trail_count = 0;
    env->trail_frames = 0;
    env->deaths++;
    env->last_x = sm64_pos(env, 0);
    env->last_z = sm64_pos(env, 2);
}

enum { ENDED_CLOCK, ENDED_DOOR, ENDED_EARLY };

static void sm64_end_episode(SM64* env, int how) {
    float frames = (float)(env->tick > env->start_tick ? env->tick - env->start_tick : 1);
    int door = how == ENDED_DOOR;
    env->log.perf += (float)door;
    env->log.score += door ? 1.0f - (float)env->tick / (float)env->max_ticks : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->steps;
    env->log.frames += door ? (float)env->tick : (float)env->max_ticks;
    env->log.closest += door ? 0.0f : env->closest;
    env->log.novelty += env->novelty_earned;
    env->log.ghost += env->ghost_earned;
    env->log.cells += (float)env->cells_entered;
    env->log.explored += (float)sm64_cubes_explored;
    env->log.from_start += (float)env->from_start;
    env->log.start_perf += (float)(door && env->from_start);
    env->log.archive += (float)sm64_archive_size;
    env->log.replayed += (float)env->replayed_frames;
    env->log.door_cubes += (float)sm64_door_cube_count;
    env->log.frontier += (float)sm64_frontier;
    if (env->start_frontier >= 0) {
        sm64_demo_result(env, door);
    }
    env->log.distance += env->distance;
    env->log.top_speed += env->top_speed;
    env->log.forward_vel += env->forward_vel_sum / frames;
    env->log.airborne += (float)env->airborne_frames / frames;
    env->log.dialog += (float)env->dialog_frames / frames;
    env->log.ended_early += (float)(how == ENDED_EARLY);
    env->log.deaths += (float)env->deaths;
    env->log.n += 1.0f;
    env->agents[0].terminals[0] = 1.0f;
    puf_reset(env);
}

/* Required function */
void puf_step(SM64* env) {
    Agent* agent = &env->agents[0];
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;
    env->steps++;

    int direction = (int)agent->actions[0];
    if (direction < 0 || direction > STICK_DIRECTIONS) {
        direction = 0;
    }
    uint16_t buttons = 0;
    if ((int)agent->actions[1] == 1) buttons |= BUTTON_A;
    if ((int)agent->actions[2] == 1) buttons |= BUTTON_B;
    if ((int)agent->actions[3] == 1) buttons |= BUTTON_Z;

    float stick_x, stick_y;
    sm64_aim(env, direction, &stick_x, &stick_y);
    n64gym_pad(&env->gym, buttons, stick_x, stick_y);
    sm64_trail_push(env, buttons, env->frameskip, stick_x, stick_y);
    if (!n64gym_step(&env->gym, env->frameskip)) {
        fprintf(stderr, "sm64: the game stopped: %s\n", env->gym.error);
        exit(1);
    }
    env->tick += env->frameskip;
    sm64_watch_camera(env);

    uint32_t action = sm64_action(env);
    float step_cost = env->time_penalty * (float)env->frameskip / (float)env->max_ticks;

    /* The door, the frame it starts to open, or the star, the frame he touches
     * it. The warp inside or the star's dance comes after, and is the same for
     * every run, so it is not counted. */
    if (sm64_goal_reached(env)) {
        agent->rewards[0] = DOOR_REWARD - step_cost;
        env->episode_return += agent->rewards[0];
        sm64_fastest_offer(env);
        /* An exploring run offers its way to the demo, and so does a
         * robustifying one: a policy that gets there faster than the demo it
         * started on -- from the real start, or from a frontier, the replayed
         * part included -- is the demo the next run should work back along. The
         * run under way keeps the demo it read when it started. */
        if (env->go_explore > 0.0f || env->backward > 0.0f) {
            sm64_demo_offer(env);
        }
        if (env->go_explore > 0.0f) {
            sm64_credit_door(env, sm64_cube(env, sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2)));
        }
        sm64_end_episode(env, ENDED_DOOR);
        return;
    }
    /* A death, or any other way out of the castle grounds (or the course), is
     * not the goal, and it must not be a way to stop the clock either: it pays
     * for the rest of the clock at once, the same as running it out.
     *
     * With respawn on, the episode carries on instead: the game goes back to the
     * savestate and the clock keeps running, so a death costs the frames it
     * wasted and the ground it gave up, and is still not a way to stop the
     * clock. That is for a course a policy falls out of before it has learned
     * anything -- in Whomp's Fortress, 70% of episodes ended in a death, most
     * inside the first half of the clock, and each one that did explored a third
     * of what a full episode does. */
    if (sm64_level(env) != sm64_home_level(env) || n64_s16(&env->gym, env->mario + M_HEALTH) < 0x100) {
        if (env->respawn) {
            sm64_respawn(env);
            agent->rewards[0] = -step_cost;
            env->episode_return += agent->rewards[0];
            if (env->tick >= env->max_ticks) {
                sm64_end_episode(env, ENDED_CLOCK);
                return;
            }
            sm64_write_observations(env, 0.0f);
            return;
        }
        int left = env->max_ticks - env->tick;
        agent->rewards[0] = -step_cost - env->time_penalty * (float)(left > 0 ? left : 0) / (float)env->max_ticks;
        env->episode_return += agent->rewards[0];
        env->deaths++;
        sm64_end_episode(env, ENDED_EARLY);
        return;
    }

    float x = sm64_pos(env, 0);
    float z = sm64_pos(env, 2);
    float moved = sqrtf((x - env->last_x) * (x - env->last_x) + (z - env->last_z) * (z - env->last_z));
    env->last_x = x;
    env->last_z = z;
    float speed = moved / (float)env->frameskip;
    env->distance += moved;
    env->forward_vel_sum += fabsf(sm64_forward_vel(env)) * (float)env->frameskip;
    if (action & ACT_FLAG_AIR) {
        env->airborne_frames += env->frameskip;
    }
    if ((action & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE) {
        env->dialog_frames += env->frameskip;
    }
    if (speed > env->top_speed) {
        env->top_speed = speed;
    }

    agent->rewards[0] = -step_cost + sm64_novelty(env, x, sm64_pos(env, 1), z);
    env->episode_return += agent->rewards[0];
    env->closest = fminf(env->closest, sm64_goal_distance(env, x, z));

    if (env->tick >= env->max_ticks) {
        sm64_end_episode(env, ENDED_CLOCK);
        return;
    }
    sm64_write_observations(env, speed);
}

/* Required function. The game draws itself in its own window when the env was
 * made with `window = 1`; all there is to do here is not run it faster than a
 * television would. */
void puf_render(SM64* env) {
    if (!env->window) {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double seconds = now.tv_sec + now.tv_nsec / 1e9;
    double frame = (double)env->frameskip / 30.0; /* the game runs at thirty frames a second */
    if (env->next_frame_time == 0.0 || seconds > env->next_frame_time + 1.0) {
        env->next_frame_time = seconds;
    }
    env->next_frame_time += frame;
    double wait = env->next_frame_time - seconds;
    if (wait > 0.0) {
        struct timespec sleep = {(time_t)wait, (long)((wait - (double)(time_t)wait) * 1e9)};
        nanosleep(&sleep, NULL);
    }
}

/* Required function. Do not free the agent buffers: the vecenv owns them */
void puf_close(SM64* env) {
    if (env->opened) {
        n64gym_close(&env->gym);
        env->opened = 0;
    }
    free(env->entered);
    free(env->trail);
    env->trail = NULL;
    env->entered = NULL;
}

/* Required function: read this env's settings from [env] in its config, and start its game */
void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = (int)dict_get(kwargs, "frameskip");
    env->max_ticks = (int)dict_get(kwargs, "max_ticks");
    env->random_start = (int)dict_get(kwargs, "random_start");
    env->time_penalty = (float)dict_get(kwargs, "time_penalty");
    env->novelty = (float)dict_get(kwargs, "novelty");
    env->novelty_episode = (float)dict_get(kwargs, "novelty_episode");
    env->novelty_cell = (float)dict_get(kwargs, "novelty_cell");
    env->respawn = (int)dict_get(kwargs, "respawn");
    env->go_explore = (float)dict_get(kwargs, "go_explore");
    env->go_explore_door = (float)dict_get(kwargs, "go_explore_door");
    env->go_explore_seed = (int)dict_get(kwargs, "go_explore_seed");
    env->ghost = (float)dict_get(kwargs, "ghost");
    env->backward = (float)dict_get(kwargs, "backward");
    env->backward_step = (int)dict_get(kwargs, "backward_step");
    env->backward_rate = (float)dict_get(kwargs, "backward_rate");
    env->backward_start = (int)dict_get(kwargs, "backward_start");
    env->window = (int)dict_get(kwargs, "window");
    env->picture_width = (int)dict_get(kwargs, "picture_width");
    env->picture_height = (int)dict_get(kwargs, "picture_height");
    env->state = (int)dict_get(kwargs, "state");
    sm64_obs_size = NUM_OBS + env->picture_width * env->picture_height * PICTURE_CHANNELS;
    init(env);
}

/* Required function: the Log averaged over episodes, by name */
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "frames", log->frames);
    dict_set(out, "closest", log->closest);
    dict_set(out, "novelty", log->novelty);
    dict_set(out, "ghost", log->ghost);
    dict_set(out, "cells", log->cells);
    dict_set(out, "explored", log->explored);
    dict_set(out, "from_start", log->from_start);
    dict_set(out, "start_perf", log->start_perf);
    dict_set(out, "archive", log->archive);
    dict_set(out, "replayed", log->replayed);
    dict_set(out, "door_cubes", log->door_cubes);
    dict_set(out, "frontier", log->frontier);
    dict_set(out, "distance", log->distance);
    dict_set(out, "top_speed", log->top_speed);
    dict_set(out, "forward_vel", log->forward_vel);
    dict_set(out, "airborne", log->airborne);
    dict_set(out, "dialog", log->dialog);
    dict_set(out, "ended_early", log->ended_early);
    dict_set(out, "deaths", log->deaths);
}

#endif
