/* Super Mario 64: get one star as soon as Mario can.
 *
 * The game is the real cartridge, statically recompiled to native arm64 by
 * N64Bundler and run in a process of its own (see n64b_gym.h). This file is the
 * environment around it: which star, how Mario gets to its course, what an
 * action does to the controller, what his state is worth, and where all of that
 * lives in the console's memory.
 *
 * Which star is config. `star` in [env] names it the way the game numbers a
 * course's stars: pss-2 is Peach's Secret Slide's second, wf-1 is Whomp's
 * Fortress's first, and a course alone, bob, is any star in it (see the
 * courses). Episodes start where the course's painting drops Mario, entered for
 * the star's act, from a savestate made the first time that star is asked for
 * (see getting there). Everything a star keeps -- that savestate, the fastest
 * runs to it, the runs that seed the archive -- is in a folder of its own,
 * build/sm64/<star>/.
 *
 * Reward is the star, the clock, and novelty. The star pays STAR_REWARD and ends
 * the episode the frame Mario touches it: that star and no other, by its flag
 * going on in the save file. Every frame costs time_penalty / max_ticks, so an
 * episode that runs out of time has paid time_penalty in all. Nothing tells the
 * policy where the star is or pays for getting nearer to it: it sees what Mario
 * is doing and where he is, and finds out about the star by taking it.
 *
 * With the goal and the clock alone, every episode is worth exactly
 * -time_penalty until the first success. When the goal was the castle's front
 * door, which is this env's history, 12M steps of that never found it: entropy
 * stayed at its maximum, and the typical episode ended as far from the door as
 * it began. Nothing in that setup remembers where Mario has been, so exploring
 * is jittering the stick.
 *
 * Novelty is that memory, and it knows nothing about the star. A course is cut
 * into cubes novelty_cell units a side. The first time in an episode that Mario
 * enters a cube, he is paid
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
 * to the start, and the door opened too rarely for the policy to learn the way.
 *
 * The first part never fades: an episode that covers ground is always worth
 * more than one that does not, so the far side of a course keeps being reached.
 * It is small. 0.02 a cube is about what running costs in clock, so a run
 * straight to the goal is still worth more than wandering all episode.
 *
 * Novelty with both parts did not get to the door either: its episodes swam the
 * moat, dozens of new cubes that lead nowhere, while the way to the door was a
 * bridge of two cubes and then 250 frames of Lakitu with nothing new at all.
 *
 * So some episodes start further on (Go-Explore). Every cube any episode has
 * entered is kept in an archive with the shortest run of pad inputs, from the
 * savestate, that reached it. The game is deterministic -- 1400 random inputs
 * replayed give the same Mario to the bit, in the same game or another -- so
 * the inputs are the place, at a few kilobytes instead of a savestate's eight
 * megabytes. A go_explore share of episodes pick a cube, weighted to the ones
 * fewest episodes have entered, replay its inputs, and only then start the
 * clock. The rest start where the task does. The reward is the same either
 * way; only where some episodes begin has changed, and the log keeps the star
 * rate from the real start apart (start_perf / from_start).
 *
 * Exploring is Go-Explore's first phase, and what it finds is a way to the
 * star, not a policy that can take it from the start. Its second phase makes
 * one: every run that reaches the star offers its inputs to a demo file, which
 * keeps the fastest, and a `backward` share of episodes start on that demo,
 * moving back from the star as the policy learns (see the fastest run).
 *
 * A death, or a warp out of the course, pays for the rest of the clock at
 * once, so no episode is ever worth less than running it out, and none worth
 * more: stopping the clock early is not a way to lose less. With respawn on, a
 * death puts the game back instead and the episode carries on.
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
#define CURR_COURSE_NUM 0x8033BAC6 /* s16; Bob-omb Battlefield is course 1 */
#define CURR_ACT_NUM 0x8033BAC8    /* s16 */
#define PLAYER1_CONTROLLER 0x8033AF90

#define LEVEL_CASTLE_GROUNDS 16

/* As much as rewards are clipped to in a step: more would buy nothing. */
#define STAR_REWARD 1.0f

/* Which stars are his: gSaveBuffer.files[0][0].courseStars, a byte a course
 * (course 1 first), a bit a star (star 1 the lowest). The first file is the one
 * the savestate opens. Found by snapshotting the console's memory before and
 * after a touch: the save file starts at 0x80207700 with its signature at +0x34,
 * and taking Peach's Secret Slide's star turns on bit 1 of the byte for course
 * 19, the frame the star count goes up. */
#define SAVE_FILE_COURSE_STARS 0x8020770C

/* A star that a box or a boss lets out, or that the slide awards for a time, is
 * spawned: it stops time, plays its cutscene for about a hundred frames while
 * Mario stands frozen, and lands where he can take it. It is not the goal -- a
 * star that has spawned is not yet his -- but sm64_tool says when one spawned
 * in a run it replays, from gTimeStopState: the halfword at 0x8033D482, found
 * by diffing the console's memory before and during the freeze. A spawning star
 * sets it to ENABLED (2) | MARIO_AND_DOORS (8), with ACTIVE (0x40) coming on as
 * it takes hold. Nothing else in a course sets MARIO_AND_DOORS: dialog sets
 * DIALOG (4) instead, and the doors that set it are in the castle. */
#define TIME_STOP_STATE 0x8033D482 /* s16 */
#define TIME_STOP_MARIO_AND_DOORS 0x08

/* The way into a course (see getting there): the castle grounds keep a warp
 * node for each half of the front door, {id, level, area, node} in one word,
 * leading to the castle's inside. Pointed at a course's PAINTING_NODE instead,
 * the door takes Mario there the way a painting does, act select and all. */
#define DOOR_WARP_NODE_LEFT 0x80196414
#define DOOR_WARP_NODE_RIGHT 0x80196420
#define DOOR_WARP_TO_CASTLE_LEFT 0x00060100 /* node 0 -> level 6, area 1, node 0 */
#define DOOR_WARP_TO_CASTLE_RIGHT 0x01060101
#define PAINTING_NODE 0x0A /* where a course's painting puts Mario, in every course */

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
#define ACT_WATER_IDLE 0x380022C0
#define ACT_FLYING 0x10880899

/* libultra's button bits, as the pad reports them. */
#define BUTTON_A 0x8000
#define BUTTON_B 0x4000
#define BUTTON_Z 0x2000
#define BUTTON_START 0x1000

#define STICK_DIRECTIONS 16
#define ACTION_HEADS 4

/* --- the courses -----------------------------------------------------------------
 *
 * A star is named for its course and its number there, as the act select and
 * the save file number them: wf-1 is Whomp's Fortress's first star, bob-7
 * Bob-omb Battlefield's hundred coins, pss-2 the slide's second -- the one for
 * reaching the bottom inside 21 seconds, which every slide run on file is.
 * A course's name alone, bob, is any star in it.
 *
 * A course's objects depend on the act it was entered for, and a star is only
 * there in the acts that spawn it. Stars 1 to 6 of a course with an act select
 * are entered for their own act, and the hundred coins, or any star, for act 1;
 * `act` in [env] enters for another. A new save file's act select offers act 1
 * and nothing else, so the act is written over the selected one while the
 * course loads, before the level script spawns anything (see getting there).
 *
 * The courses after rr have no act select: one way in and one course, and the
 * game keeps the act at 0. For those nothing is written over it, and A is not
 * pressed on the way in.
 *
 * Every course is entered through PAINTING_NODE, and every one of them lands
 * Mario standing in the course, in the right course number, from a savestate
 * made from nothing (sm64_tool state).
 */
typedef struct {
    const char* name;  /* what a star's name starts with: wf in wf-1 */
    const char* title;
    int level;         /* LEVEL_*: what the game loads, and what a warp names */
    int course;        /* COURSE_*: what the save file keeps a course's stars by */
    int stars;
} SM64Course;

static const SM64Course sm64_courses[] = {
    {"bob", "Bob-omb Battlefield", 9, 1, 7},
    {"wf", "Whomp's Fortress", 24, 2, 7},
    {"jrb", "Jolly Roger Bay", 12, 3, 7},
    {"ccm", "Cool, Cool Mountain", 5, 4, 7},
    {"bbh", "Big Boo's Haunt", 4, 5, 7},
    {"hmc", "Hazy Maze Cave", 7, 6, 7},
    {"lll", "Lethal Lava Land", 22, 7, 7},
    {"ssl", "Shifting Sand Land", 8, 8, 7},
    {"ddd", "Dire, Dire Docks", 23, 9, 7},
    {"sl", "Snowman's Land", 10, 10, 7},
    {"wdw", "Wet-Dry World", 11, 11, 7},
    {"ttm", "Tall, Tall Mountain", 36, 12, 7},
    {"thi", "Tiny-Huge Island", 13, 13, 7},
    {"ttc", "Tick Tock Clock", 14, 14, 7},
    {"rr", "Rainbow Ride", 15, 15, 7},
    {"bitdw", "Bowser in the Dark World", 17, 16, 1},
    {"bitfs", "Bowser in the Fire Sea", 19, 17, 1},
    {"bits", "Bowser in the Sky", 21, 18, 1},
    {"pss", "Peach's Secret Slide", 27, 19, 2},
    {"cotmc", "Cavern of the Metal Cap", 28, 20, 1},
    {"totwc", "Tower of the Wing Cap", 29, 21, 1},
    {"vcutm", "Vanish Cap Under the Moat", 18, 22, 1},
    {"wmotr", "Wing Mario Over the Rainbow", 31, 23, 1},
    {"sa", "The Secret Aquarium", 20, 24, 1},
};
#define SM64_COURSES ((int)(sizeof(sm64_courses) / sizeof(sm64_courses[0])))
#define SM64_ACT_SELECT_COURSES 15 /* bob to rr */

/* One star, as the config names it. */
typedef struct {
    const SM64Course* course;
    int star;       /* 1 to course->stars; 0 is any star in the course */
    int act;        /* the act it is entered for; 0 for a course with no act select */
    char name[32];  /* its folder: pss-2, or bob-7-act3 for a star entered for an act not its own */
} SM64Goal;

static inline int sm64_course_selects_act(const SM64Course* course) {
    return course->course <= SM64_ACT_SELECT_COURSES;
}

/* Read `star` and `act` from [env] into a goal. Returns 0 and says why if they
 * name no star there is. */
static int sm64_goal_parse(const char* star, int act, SM64Goal* goal, char* error, size_t size) {
    const char* dash = strchr(star, '-');
    size_t length = dash != NULL ? (size_t)(dash - star) : strlen(star);
    memset(goal, 0, sizeof(*goal));
    for (int k = 0; k < SM64_COURSES; k++) {
        if (strlen(sm64_courses[k].name) == length && strncmp(sm64_courses[k].name, star, length) == 0) {
            goal->course = &sm64_courses[k];
        }
    }
    if (goal->course == NULL) {
        int used = snprintf(error, size, "there is no course called '%.*s'; the courses are", (int)length, star);
        for (int k = 0; k < SM64_COURSES && used > 0 && (size_t)used < size; k++) {
            used += snprintf(error + used, size - (size_t)used, " %s", sm64_courses[k].name);
        }
        return 0;
    }
    const SM64Course* course = goal->course;
    if (dash != NULL) {
        char* end;
        long number = strtol(dash + 1, &end, 10);
        if (end == dash + 1 || *end != '\0' || number < 1 || number > course->stars) {
            snprintf(error, size, "'%s': %s has stars %s%d", star, course->title, course->stars > 1 ? "1 to " : "",
                     course->stars);
            return 0;
        }
        goal->star = (int)number;
    }
    int own = !sm64_course_selects_act(course) ? 0 : goal->star >= 1 && goal->star <= 6 ? goal->star : 1;
    if (act == 0) {
        act = own;
    } else if (!sm64_course_selects_act(course)) {
        snprintf(error, size, "%s has no act select, so act has to be 0, not %d", course->title, act);
        return 0;
    } else if (act < 1 || act > 6) {
        snprintf(error, size, "act %d: a course's acts are 1 to 6", act);
        return 0;
    }
    goal->act = act;
    int written = snprintf(goal->name, sizeof(goal->name), "%.*s", (int)strlen(star), star);
    if (act != own && written > 0 && (size_t)written < sizeof(goal->name)) {
        snprintf(goal->name + written, sizeof(goal->name) - (size_t)written, "-act%d", act);
    }
    return 1;
}

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
    float perf;            /* fraction of episodes that took the star */
    float score;           /* fraction of the clock left when he took it, 0 if he did not */
    float episode_return;
    float episode_length;  /* agent steps */
    float frames;          /* frames to the star, the whole clock without it: what is minimized */
    float novelty;         /* what novelty paid this episode */
    float ghost;           /* what beating the archive to cubes on the way to the star paid this episode */
    float cells;           /* cubes entered this episode */
    float explored;        /* cubes any game has ever entered, as of the end of this episode */
    float from_start;      /* fraction of episodes that began where the task does, not from the archive */
    float start_perf;      /* fraction that began there and took the star: the star rate is start_perf / from_start */
    float archive;         /* cubes in the archive */
    float replayed;        /* frames replayed to reach the episode's starting cube, 0 from the real start */
    float star_cubes;      /* cubes that have been on the way to the star */
    float frontier;        /* frames before the star that episodes on the demo start at, 0 with no demo */
    float distance;        /* the length of the path he ran */
    float top_speed;       /* the best single frame of it, units per frame */
    float forward_vel;     /* what the game thought his speed was, on average */
    float airborne;        /* fraction of frames off the ground */
    float dialog;          /* fraction of frames in a cutscene: dialog, a star spawning */
    float ended_early;     /* fraction of episodes cut short by a death or a warp out of the course */
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

    SM64Goal goal;        /* which star: `star` and `act` in [env] */
    int frameskip;        /* frames of the game per agent step */
    int max_ticks;        /* frames of the game per episode */
    int random_start;     /* frames of random stick held after loading the state */
    float time_penalty;   /* reward the whole clock costs, spent a frame at a time */
    float novelty;        /* reward for entering a cube no episode has entered before, fading as 1 / sqrt(episodes) */
    float novelty_episode; /* reward for entering any cube for the first time this episode, which never fades */
    float novelty_cell;   /* the side of a cube, in units */
    int respawn;          /* a death puts the game back and the episode goes on, rather than ending it */
    float go_explore;     /* share of episodes that start from a cube in the archive */
    float go_explore_star; /* share of those that start from a cube on the way to the star, once there is one */
    int go_explore_seed;  /* start the archive with the runs on file rather than with nothing */
    float ghost;          /* reward per frame sooner than the archive's way into a cube on the way to the star */
    float backward;       /* share of episodes that start on the demo, a little before the frontier */
    int backward_step;    /* frames the frontier moves back at a time */
    float backward_rate;  /* the star rate from the frontier that moves it back */
    int backward_start;   /* frames before the star the frontier begins at; 0 is backward_step */
    int window;           /* draw the game, for watching a policy play */
    int picture_width;    /* the game's picture in the observation, shrunk to this; 0 is none */
    int picture_height;
    int state;            /* Mario's state, read out of memory, in the observation; 0 leaves it zero */

    N64Gym gym;
    int opened;
    int star_flags_at_start; /* the savestate's stars in this course, which the star adds to */
    uint32_t mario;       /* where MarioState is, read once from its pointer */
    float last_x, last_z;
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

/* The stars of the goal's course that the save file has. */
static inline int sm64_star_flags(SM64* env) {
    return n64_u8(&env->gym, SAVE_FILE_COURSE_STARS + (uint32_t)(env->goal.course->course - 1));
}

/* The star, the frame it is his: its flag in the save file going on as he
 * touches it, the same frame the star count goes up. Any other star in the
 * course is not it, and taking one throws Mario out of the course, which ends
 * the episode as a death does. */
static int sm64_goal_reached(SM64* env) {
    int wanted = env->goal.star > 0 ? 1 << (env->goal.star - 1) : 0x7F;
    return (sm64_star_flags(env) & ~env->star_flags_at_start & wanted) != 0;
}

/* Whether a star is spawning: time stopped for Mario and the doors (see
 * TIME_STOP_STATE). Not the goal, but sm64_tool reports it. */
static int sm64_star_spawning(SM64* env) {
    return (n64_s16(&env->gym, TIME_STOP_STATE) & TIME_STOP_MARIO_AND_DOORS) != 0;
}

/* --- novelty ------------------------------------------------------------------
 *
 * Cubes over the whole of a level's space: x and z from -8192 to 8192, y from
 * -4096 to 8192. The visit counts are one table for the process, because the
 * games stepping in parallel are exploring the same course, and a cube one of
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

/* --- the archive (Go-Explore) ------------------------------------------------
 *
 * For every cube, the shortest run of pad inputs from the savestate that has
 * reached it. One archive for the process, like the visit counts, behind a
 * lock: the games reset and step in parallel. A run longer than
 * ARCHIVE_EPISODES episodes' worth of frames is not kept, so no reset replays
 * more than that: 2700 frames with 900-frame episodes.
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
 * so a star from here credits them too. (The episode number goes up once reset
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

/* Which cubes have been on the way to the star, and how many episodes have
 * started from each. Rarity alone was not enough: after 8M steps of starting
 * from the cubes fewest episodes had entered -- far corners of the moat and the
 * lake, mostly, when the goal was the castle door -- the door opened once in a
 * thousand, from either kind of start. So once an episode has taken the star,
 * every cube it passed through, the replayed part included, is credited, and a
 * go_explore_star share of archive starts are drawn from those cubes, least
 * started-from first: all the way along some path to the star, from near the
 * start to the step before it. That is still only the reward speaking. Nothing
 * says where the star is. */
static uint32_t sm64_cube_stars[NOVELTY_CUBES];
static uint32_t sm64_cube_starts[NOVELTY_CUBES];
static int sm64_star_cubes[NOVELTY_CUBES];
static int sm64_star_cube_count;

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

/* The episode has taken the star: credit every cube it was in. */
static void sm64_credit_star(SM64* env, int star_cube) {
    if (star_cube >= 0) {
        env->entered[star_cube] = env->episode;
    }
    pthread_mutex_lock(&sm64_archive_lock);
    for (int cube = 0; cube < NOVELTY_CUBES; cube++) {
        if (env->entered[cube] != env->episode || sm64_archive[cube].inputs == NULL) {
            continue;
        }
        if (sm64_cube_stars[cube]++ == 0) {
            sm64_star_cubes[sm64_star_cube_count++] = cube;
        }
    }
    pthread_mutex_unlock(&sm64_archive_lock);
}

/* Start from a cube in the archive: copy its run onto this episode's trail and
 * replay it. The game is left where that run left it. Returns the cube, or -1
 * if the archive is empty. */
static int sm64_archive_start(SM64* env) {
    pthread_mutex_lock(&sm64_archive_lock);
    if (sm64_archive_size == 0) {
        pthread_mutex_unlock(&sm64_archive_lock);
        return -1;
    }
    int cube;
    if (sm64_star_cube_count > 0 &&
        (double)rand_r(&env->seed) / ((double)RAND_MAX + 1.0) < env->go_explore_star) {
        cube = sm64_draw(env, sm64_star_cubes, sm64_star_cube_count, sm64_cube_starts);
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
 * The clock pays for speed only at the star, and little: thirty frames off a
 * 770-frame star is 0.02 of reward. What a run does get faster by is the
 * archive, which keeps the shortest way into every cube -- so the archive's way
 * into a cube is a ghost to race, as in a racing game, and beating it is paid
 * where it happens rather than 400 frames later.
 *
 * The first time in an episode Mario enters a cube that has been on the way to
 * the star, sooner than the archive's way into it, he is paid `ghost` a frame
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
    if (sm64_cube_stars[cube] > 0 && cell->inputs != NULL && env->trail_frames < cell->frames) {
        pay = fminf(env->ghost * (float)(cell->frames - env->trail_frames), GHOST_MOST);
    }
    pthread_mutex_unlock(&sm64_archive_lock);
    return pay;
}

/* Novelty's pay for being where Mario is now: something the first time this
 * episode he is in a cube, nothing after. */
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

/* --- getting there --------------------------------------------------------------
 *
 * Two presses of Start reach the file select and open the first file; from
 * there the game plays Peach's letter and Lakitu's arrival, which is a minute
 * and a half of cutscene that waits on A for its text boxes. Mashing A through
 * it takes about 1500 frames and ends with Mario standing outside the castle in
 * ACT_IDLE.
 *
 * Start is deliberately not mashed after the menus: in the game it opens the
 * pause screen, and a game paused is a game that never gets anywhere.
 *
 * Then the castle's front door is pointed at the star's course (see
 * DOOR_WARP_NODE_LEFT) and Mario is put in front of it with the stick pushed
 * forward. He opens it, the game fades to the act select, A picks the only act
 * a new save has while the star's is written over it, and he lands where the
 * course's painting would have dropped him. A course with no act select drops
 * him straight in, and there A is not pressed. That is done once for a star and
 * saved, and every episode starts from it.
 */

/* Where a course leaves Mario once it has him: standing, or, in the courses he
 * arrives in by water or in flight -- Dire, Dire Docks, the Secret Aquarium, the
 * Tower of the Wing Cap -- floating or flying. */
static int sm64_arrived(uint32_t action) {
    return action == ACT_IDLE || action == ACT_WATER_IDLE || action == ACT_FLYING;
}

/* Whether Mario answers the pad in the game as it is: A and the stick held for a
 * few frames change what he is doing, or move him across the ground. Under a
 * course's opening camera neither happens. Across the ground, because in the
 * Secret Aquarium he sinks whatever is held. The game is put back after. */
static int sm64_answers(N64Gym* gym, const char* path) {
    uint32_t mario = n64_u32(gym, MARIO_STATE_PTR);
    uint32_t action = n64_u32(gym, mario + M_ACTION);
    float x = n64_f32(gym, mario + M_POS), z = n64_f32(gym, mario + M_POS + 8);
    n64gym_pad(gym, BUTTON_A, 0.0f, 1.0f);
    n64gym_step(gym, 10);
    n64gym_pad(gym, 0, 0.0f, 0.0f);
    float dx = n64_f32(gym, mario + M_POS) - x, dz = n64_f32(gym, mario + M_POS + 8) - z;
    int answered = n64_u32(gym, mario + M_ACTION) != action || dx * dx + dz * dz > 1.0f;
    return n64gym_load_state(gym, path) ? answered : -1;
}

static int sm64_make_state(N64Gym* gym, const char* path, const SM64Goal* goal, char* error, size_t error_size) {
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

    if (n64_u32(gym, DOOR_WARP_NODE_LEFT) != DOOR_WARP_TO_CASTLE_LEFT ||
        n64_u32(gym, DOOR_WARP_NODE_RIGHT) != DOOR_WARP_TO_CASTLE_RIGHT) {
        snprintf(error, error_size, "the castle door's warp nodes are not at %08x and %08x",
                 DOOR_WARP_NODE_LEFT, DOOR_WARP_NODE_RIGHT);
        return 0;
    }
    int level = goal->course->level, act = goal->act;
    n64_set_u32(gym, DOOR_WARP_NODE_LEFT, 0x00000100 | ((uint32_t)level << 16) | PAINTING_NODE);
    n64_set_u32(gym, DOOR_WARP_NODE_RIGHT, 0x01000100 | ((uint32_t)level << 16) | PAINTING_NODE);
    uint32_t mario = n64_u32(gym, MARIO_STATE_PTR);
    n64_set_f32(gym, mario + M_POS, -76.0f); /* in front of the left half, facing it */
    n64_set_f32(gym, mario + M_POS + 4, 803.0f);
    n64_set_f32(gym, mario + M_POS + 8, -2900.0f);

    int selects_act = sm64_course_selects_act(goal->course);
    int spawned = 0, landed = 0;
    for (int i = 0; i < 1500 && !landed; i++) {
        int now_in = n64_s16(gym, CURR_LEVEL_NUM);
        /* Up on the stick until the door has him, then A for the act select --
         * but not once he has appeared in the course, where A is a jump, and
         * not at all for a course that never asks. */
        n64gym_pad(gym, (selects_act && now_in == level && !spawned && (i % 16) < 2) ? BUTTON_A : 0,
                   0.0f, now_in == LEVEL_CASTLE_GROUNDS ? 1.0f : 0.0f);
        if (!n64gym_step(gym, 1)) {
            snprintf(error, error_size, "the game stopped on the way to %s: %s", goal->course->title, gym->error);
            return 0;
        }
        /* The act the save file could not offer, written over the one it did,
         * every frame of the load: the level script reads it when it spawns the
         * course's objects, and that is what decides which stars are there. */
        if (selects_act && n64_s16(gym, CURR_LEVEL_NUM) == level) {
            n64_set_s16(gym, CURR_ACT_NUM, (int16_t)act);
        }
        /* Until the course has spawned him, he is still the Mario who walked
         * into the door. It drops him in from the air, or into the water. */
        mario = n64_u32(gym, MARIO_STATE_PTR);
        uint32_t action = n64_is_ram(mario) ? n64_u32(gym, mario + M_ACTION) : 0;
        spawned = spawned || (n64_s16(gym, CURR_LEVEL_NUM) == level && (action & (ACT_FLAG_AIR | ACT_FLAG_SWIMMING)));
        landed = spawned && sm64_arrived(action);
    }
    if (!landed || n64_s16(gym, CURR_LEVEL_NUM) != level || n64_s16(gym, CURR_COURSE_NUM) != goal->course->course ||
        (selects_act && n64_s16(gym, CURR_ACT_NUM) != act)) {
        snprintf(error, error_size,
                 "the door never left Mario standing in %s (level %d course %d act %d), "
                 "but in level %d course %d act %d",
                 goal->course->title, level, goal->course->course, act, n64_s16(gym, CURR_LEVEL_NUM),
                 n64_s16(gym, CURR_COURSE_NUM), n64_s16(gym, CURR_ACT_NUM));
        return 0;
    }
    /* He lands with the course's opening camera still holding the level: Mario
     * does not move, whatever is held, until a button is pressed after about
     * two seconds of it, and that press is spent on the camera. A state saved
     * before then loads into a game where he never moves at all. So B is
     * pressed every half second, and a state is kept only once, put back, Mario
     * answers the pad. */
    for (int tries = 0; tries < 30; tries++) {
        n64gym_pad(gym, BUTTON_B, 0.0f, 0.0f);
        n64gym_step(gym, 2);
        n64gym_pad(gym, 0, 0.0f, 0.0f);
        for (int wait = 0; wait < 90 && (wait < 14 || !sm64_arrived(n64_u32(gym, mario + M_ACTION))); wait++) {
            n64gym_step(gym, 1);
        }
        if (!n64gym_save_state(gym, path) || !n64gym_load_state(gym, path)) {
            snprintf(error, error_size, "%s", gym->error);
            return 0;
        }
        int answers = sm64_answers(gym, path);
        if (answers < 0) {
            snprintf(error, error_size, "%s", gym->error);
            return 0;
        }
        if (answers) {
            return 1;
        }
    }
    snprintf(error, error_size, "Mario landed in the course but never came back to life in a saved state");
    return 0;
}

/* --- where a star keeps its things ----------------------------------------------
 *
 * A folder a star, <SM64_DATA>/<star>/ (build/sm64/pss-2/), so two stars never
 * overwrite each other's:
 *
 *     start.state      where every episode starts, made the first time
 *     fastest.demo     the fastest run to the star on file, which backward works along
 *     fastest/         the FASTEST_RUNS fastest, to watch and to seed the archive with
 *     seeds/           runs kept by hand, seeding the archive too
 *     states-<hash>/   the game at points along a demo, for starting episodes there
 *
 * One star for the process, like the archive: every game in it is after the same
 * one. Worked out once, at the first game's init, because every episode asks.
 */
static const char* sm64_setting(const char* name, const char* fallback) {
    const char* found = getenv(name);
    return (found != NULL && found[0] != '\0') ? found : fallback;
}

static SM64Goal sm64_goal;
static char sm64_dir[1024];
static char sm64_state_file[1100];
static char sm64_demo_file[1100];
static pthread_mutex_t sm64_goal_lock = PTHREAD_MUTEX_INITIALIZER;

/* mkdir -p */
static void sm64_make_dirs(const char* path) {
    char partial[1024];
    snprintf(partial, sizeof(partial), "%s", path);
    for (char* p = partial + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(partial, 0755);
            *p = '/';
        }
    }
    mkdir(partial, 0755);
}

static void sm64_use_goal(const SM64Goal* goal) {
    pthread_mutex_lock(&sm64_goal_lock);
    if (sm64_dir[0] == '\0') {
        sm64_goal = *goal;
        snprintf(sm64_dir, sizeof(sm64_dir), "%s/%s", sm64_setting("SM64_DATA", SM64_DATA), goal->name);
        snprintf(sm64_state_file, sizeof(sm64_state_file), "%s/start.state", sm64_dir);
        snprintf(sm64_demo_file, sizeof(sm64_demo_file), "%s/fastest.demo", sm64_dir);
        sm64_make_dirs(sm64_dir);
    } else if (strcmp(sm64_goal.name, goal->name) != 0) {
        fprintf(stderr, "sm64: one star a process, and this one is already after %s, not %s\n", sm64_goal.name,
                goal->name);
        exit(1);
    }
    pthread_mutex_unlock(&sm64_goal_lock);
}

static const char* sm64_state_path(void) { return sm64_state_file; }
static const char* sm64_demo_path(void) { return sm64_demo_file; }

/* --- the fastest run (Go-Explore's second phase) -------------------------------
 *
 * The archive reaches the star by replaying inputs, which works only because
 * the game is deterministic. A policy started from the savestate has never seen
 * most of that way, so the second phase trains one to repeat it: robustifying,
 * with the backward algorithm (Salimans and Chen, 2018), which is what
 * Go-Explore used.
 *
 * Every exploring run (go_explore on) that takes the star offers its inputs,
 * from the savestate, to the demo file, which keeps the fastest run any of them
 * has found -- and so does every robustifying run (backward on), so a policy
 * that beats its demo leaves a faster one for the next run. A `backward` share
 * of episodes start on that demo at the frontier -- the game as the demo left
 * it that many frames before the star -- and play from there. The frontier
 * begins backward_step frames before the star. Once backward_rate of a window
 * of episodes started from it take the star, it moves backward_step frames
 * further back, until it is the start of the demo. The rest of the episodes
 * start where the task does, so start_perf / from_start is still the star rate
 * that counts.
 *
 * Half the episodes on the demo start at a frontier already passed, nearer the
 * star, instead, and do not count toward moving it, so what the policy has
 * already learned keeps being paid for. Without them, the first run on the
 * castle door moved its frontier from 30 frames to 180 in 165K steps, into
 * Lakitu's speech, and stopped there. Every episode on the demo then started
 * where the policy could not yet win, and by 200K steps it played at random
 * (entropy 4.3 of 4.9) and opened nothing.
 *
 * The demo a run robustifies is the one on file when it starts.
 */
#define SM64_DEMO_MAGIC "SM64DEMO"
#define BACKWARD_WINDOW 32
#define DEMO_SLACK 300 /* frames: ten seconds more than the demo took */

typedef struct {
    char magic[8];
    int32_t count;  /* inputs */
    int32_t frames; /* from the savestate to the frame the star is his */
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
static int sm64_demo_best = -1;   /* frames of the fastest run on file, once it has been looked at */
static int sm64_frontier;         /* frames before the star */
static int sm64_frontier_tries, sm64_frontier_stars;
static pthread_mutex_t sm64_demo_lock = PTHREAD_MUTEX_INITIALIZER;

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
    char temporary[1200];
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

/* This episode has taken the star: if no run on file was faster, it is the demo now. */
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
            fprintf(stderr, "sm64: could not keep a star of %d frames in %s\n", env->trail_frames, path);
        }
    }
    pthread_mutex_unlock(&sm64_demo_lock);
}

/* --- the fastest runs, to watch ----------------------------------------------
 *
 * The demo keeps one run, and any slower way to the star is gone when its
 * episode ends -- and with it the archive's cube it started from, so nothing can
 * play it again. So every episode that takes the star, exploring or not, also
 * offers its inputs to the star's fastest/ folder, which keeps the FASTEST_RUNS
 * fastest. A file is named for its frames and a hash of its inputs,
 * 01532-9a3c1e2b.demo, so the folder lists fastest first and the same run found
 * twice is kept once. `sm64_tool replay watch <file>` plays one.
 */
#ifndef FASTEST_RUNS
#define FASTEST_RUNS 10
#endif

static char sm64_fastest_dir[1100];
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
        char path[1200];
        snprintf(path, sizeof(path), "%s/%s", sm64_fastest_dir, sm64_fastest[FASTEST_RUNS]);
        remove(path);
        sm64_fastest_count = FASTEST_RUNS;
    }
}

/* What an earlier run left in the folder, once for the process. */
static void sm64_fastest_read(void) {
    snprintf(sm64_fastest_dir, sizeof(sm64_fastest_dir), "%s/fastest", sm64_dir);
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

/* This episode has taken the star: keep it if it is among the fastest. */
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
        char path[1200];
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
static char sm64_demo_states[1200]; /* where the game is kept at each point of the demo episodes start from */

/* Remove every other folder of states in the star's folder -- the states of
 * demos since replaced, 8 MB each and no use now. Only files named like a state
 * go, and a folder with anything else in it stays. */
static void sm64_remove_other_states(const char* keep) {
    const char* slash = strrchr(keep, '/');
    const char* name = slash != NULL ? slash + 1 : keep;
    DIR* dir = opendir(sm64_dir);
    if (dir == NULL) {
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "states-", strlen("states-")) != 0 || strcmp(entry->d_name, name) == 0) {
            continue;
        }
        char other[1400];
        snprintf(other, sizeof(other), "%s/%s", sm64_dir, entry->d_name);
        DIR* states = opendir(other);
        if (states == NULL) {
            continue;
        }
        struct dirent* state;
        while ((state = readdir(states)) != NULL) {
            int frame;
            if (sscanf(state->d_name, "%d.state", &frame) == 1) {
                char path[1700];
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
        if (sm64_demo == NULL) {
            fprintf(stderr, "sm64: backward works back along the fastest run to the star, and there is none in %s.\n"
                            "      Find one first: make sm64-explore STAR=%s\n", path, env->goal.name);
            exit(1);
        }
        int first = env->backward_start > 0 ? env->backward_start : env->backward_step > 0 ? env->backward_step : 1;
        sm64_frontier = first < sm64_demo_frames ? first : sm64_demo_frames;
        /* states-9a3c1e2b/, named for this demo's inputs, so another demo's
         * states are never mistaken for its own. */
        snprintf(sm64_demo_states, sizeof(sm64_demo_states), "%s/states-%08x", sm64_dir,
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
    char path[1300];
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
    char temporary[1400];
    snprintf(temporary, sizeof(temporary), "%s.%d", path, (int)env->rng);
    if (!n64gym_save_state(&env->gym, temporary) || rename(temporary, path) != 0) {
        fprintf(stderr, "sm64: could not keep the game at frame %d of the demo in %s\n", env->trail_frames, path);
        remove(temporary);
    }
}

/* Start on the demo: at the frontier, or, half the time, at any of the
 * frontiers already passed, nearer the star. Starts are on the frontier's grid
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

    /* The clock starts partway, so an episode has as long to reach the star as
     * the demo took from here plus DEMO_SLACK, never more than the whole clock,
     * and one that will not make it is over in about that long.
     *
     * On the castle door, the second run started every one with the whole
     * clock, 900 frames to get from a few seconds out. Most episodes ran all of
     * it and paid -1, entropy fell to 0.01 by 260K steps, and the frontier never
     * got past 120. The runs after that set the clock to what the demo's read at
     * that point, and the two that trained stably stuck at 570: that far back,
     * the explorer's wandering at the start of the demo had already spent a
     * hundred frames, so an episode there had less time than one from the real
     * start. */
    int budget = sm64_demo_frames - env->trail_frames + DEMO_SLACK;
    env->start_tick = budget < env->max_ticks ? env->max_ticks - budget : 0;
}

/* An episode that began on the demo is over: count it toward moving the frontier
 * back, if the frontier is still where it began. */
static void sm64_demo_result(SM64* env, int star) {
    pthread_mutex_lock(&sm64_demo_lock);
    if (env->start_frontier == sm64_frontier && sm64_frontier < sm64_demo_frames) {
        sm64_frontier_tries++;
        sm64_frontier_stars += star;
        if (sm64_frontier_tries >= BACKWARD_WINDOW) {
            if ((float)sm64_frontier_stars >= env->backward_rate * (float)sm64_frontier_tries) {
                sm64_frontier += env->backward_step > 0 ? env->backward_step : 1;
                if (sm64_frontier > sm64_demo_frames) {
                    sm64_frontier = sm64_demo_frames;
                }
            }
            sm64_frontier_tries = 0;
            sm64_frontier_stars = 0;
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
 * -- the demo, the fastest/ folder, and any kept by hand in seeds/ -- and offers
 * each cube on the way to the archive, as the episode that found it did, then
 * credits those cubes as on the way to the star, which they are. The archive
 * keeps the shortest way to a cube, so what a run starts with is the best of
 * all of them at every place, and half its archive starts are along them from
 * the first episode.
 *
 * A run stops being offered the step it reaches the star, as an episode does:
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
        sm64_credit_star(env, -1);
    } else {
        fprintf(stderr, "sm64: %s does not reach the star when played; its cubes are kept, not credited\n", path);
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
            char path[1200];
            snprintf(path, sizeof(path), "%s/%s", sm64_fastest_dir, names[k]);
            runs += sm64_seed_from(env, path);
        }
        /* And any run kept by hand in seeds/. The folder of the fastest drops a
         * run once ten are faster, and with it whatever only that run had: a way
         * over the top, a line through a turn. */
        char seeds[1100];
        snprintf(seeds, sizeof(seeds), "%s/seeds", sm64_dir);
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
        fprintf(stderr, "sm64: the archive starts with %d cubes, %d of them on the way to the star, from %d runs on file\n",
                sm64_archive_size, sm64_star_cube_count, runs);
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
    env->entered = (uint32_t*)calloc(NOVELTY_CUBES, sizeof(uint32_t));
    env->seed = 0x9E3779B9u ^ (env->rng * 2654435761u);
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    sm64_use_goal(&env->goal);
    if (env->backward > 0.0f) {
        sm64_demo_load(env);
    }

    const char* state = sm64_state_path();
    struct stat ignored;
    if (stat(state, &ignored) != 0) {
        /* Nobody has been to this star yet. Get there once, here, and every
         * episode of every run from now on starts from it. */
        char error[256];
        fprintf(stderr, "sm64: playing through the intro and into %s once, to make %s\n", env->goal.course->title,
                state);
        if (!sm64_make_state(&env->gym, state, &env->goal, error, sizeof(error))) {
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
    env->star_flags_at_start = sm64_star_flags(env);
    if (sm64_level(env) != env->goal.course->level || n64_s16(&env->gym, CURR_COURSE_NUM) != env->goal.course->course ||
        (env->goal.act > 0 && n64_s16(&env->gym, CURR_ACT_NUM) != env->goal.act)) {
        fprintf(stderr, "sm64: %s is level %d course %d act %d, where %s is level %d course %d act %d;"
                        " delete it to make it again\n",
                state, sm64_level(env), n64_s16(&env->gym, CURR_COURSE_NUM), n64_s16(&env->gym, CURR_ACT_NUM),
                env->goal.name, env->goal.course->level, env->goal.course->course, env->goal.act);
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

enum { ENDED_CLOCK, ENDED_STAR, ENDED_EARLY };

static void sm64_end_episode(SM64* env, int how) {
    float frames = (float)(env->tick > env->start_tick ? env->tick - env->start_tick : 1);
    int star = how == ENDED_STAR;
    env->log.perf += (float)star;
    env->log.score += star ? 1.0f - (float)env->tick / (float)env->max_ticks : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->steps;
    env->log.frames += star ? (float)env->tick : (float)env->max_ticks;
    env->log.novelty += env->novelty_earned;
    env->log.ghost += env->ghost_earned;
    env->log.cells += (float)env->cells_entered;
    env->log.explored += (float)sm64_cubes_explored;
    env->log.from_start += (float)env->from_start;
    env->log.start_perf += (float)(star && env->from_start);
    env->log.archive += (float)sm64_archive_size;
    env->log.replayed += (float)env->replayed_frames;
    env->log.star_cubes += (float)sm64_star_cube_count;
    env->log.frontier += (float)sm64_frontier;
    if (env->start_frontier >= 0) {
        sm64_demo_result(env, star);
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

    /* The star, the frame he touches it. The dance and the warp out come after,
     * and are the same for every run, so they are not counted. */
    if (sm64_goal_reached(env)) {
        agent->rewards[0] = STAR_REWARD - step_cost;
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
            sm64_credit_star(env, sm64_cube(env, sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2)));
        }
        sm64_end_episode(env, ENDED_STAR);
        return;
    }
    /* A death, or any other way out of the course -- another star included -- is
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
    if (sm64_level(env) != env->goal.course->level || n64_s16(&env->gym, env->mario + M_HEALTH) < 0x100) {
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
    char error[512];
    if (!sm64_goal_parse(dict_get_str(kwargs, "star"), (int)dict_get(kwargs, "act"), &env->goal, error,
                         sizeof(error))) {
        fprintf(stderr, "sm64: [env] star: %s\n", error);
        exit(1);
    }
    env->frameskip = (int)dict_get(kwargs, "frameskip");
    env->max_ticks = (int)dict_get(kwargs, "max_ticks");
    env->random_start = (int)dict_get(kwargs, "random_start");
    env->time_penalty = (float)dict_get(kwargs, "time_penalty");
    env->novelty = (float)dict_get(kwargs, "novelty");
    env->novelty_episode = (float)dict_get(kwargs, "novelty_episode");
    env->novelty_cell = (float)dict_get(kwargs, "novelty_cell");
    env->respawn = (int)dict_get(kwargs, "respawn");
    env->go_explore = (float)dict_get(kwargs, "go_explore");
    env->go_explore_star = (float)dict_get(kwargs, "go_explore_star");
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
    dict_set(out, "novelty", log->novelty);
    dict_set(out, "ghost", log->ghost);
    dict_set(out, "cells", log->cells);
    dict_set(out, "explored", log->explored);
    dict_set(out, "from_start", log->from_start);
    dict_set(out, "start_perf", log->start_perf);
    dict_set(out, "archive", log->archive);
    dict_set(out, "replayed", log->replayed);
    dict_set(out, "star_cubes", log->star_cubes);
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
