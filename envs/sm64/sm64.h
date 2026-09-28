/* Super Mario 64, Peach's Secret Slide: get to the star.
 *
 * The game is the real cartridge, statically recompiled to native arm64 by
 * N64Bundler and run in a process of its own (see n64b_gym.h). This file is the
 * environment around it, written against PufferLib 5.0's env API and compiled
 * straight into its CUDA trainer (src/pufferl.cu) by build.sh.
 *
 * Which star is config: `star` in [env] names it the way the game numbers a
 * course's stars (pss-2 is Peach's Secret Slide's second, the one for reaching
 * the bottom inside 21 seconds). Episodes start where the course's painting
 * drops Mario, from a savestate made the first time the star is asked for (see
 * getting there) and kept in <SM64_DATA>/<star>/start.state.
 *
 * Observation: everything about Mario that can be read out of the console's
 * memory -- position, velocity, facing, spin, the stick as the game read it,
 * the floor, wall and ceiling around him, what he is doing and what he was
 * doing, his timers and input flags -- plus where the star is from him and how
 * much of the clock is left.
 *
 * Reward: the star's location, and nothing else about the course.
 *   - progress: every time Mario gets closer to the star than he has been this
 *     episode, he is paid `progress` times the distance gained over the
 *     distance he started at, so the whole way down is worth `progress`. Only
 *     the closest approach counts, so going back and forth pays nothing, and
 *     only while there is floor under him, so falling out of the world toward
 *     the star pays nothing either.
 *   - star: STAR_REWARD, the frame he touches it (its flag going on in the save
 *     file), which ends the episode.
 *   - clock: every frame costs time_penalty / max_ticks. A death or a warp out
 *     of the course pays for the rest of the clock at once and ends the
 *     episode, so dying is never a way to stop the clock early.
 *
 * Actions: four discrete heads
 *   stick: 0 none, 1-16 a direction relative to the way Mario is facing
 *          (1 straight ahead, 5 right, 9 back, 13 left)
 *   a, b, z: held or not
 */

#ifndef SM64_H
#define SM64_H

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
 * Checked against the recompiled C that N64Bundler generates from the cartridge
 * and against the running game. `mario_set_forward_vel` at 0x80251708 pins down
 * forwardVel (0x54), the velocity (0x48), the facing (0x2C) and slideVelX/Z
 * (0x58, 0x5C) in one go; the rest of struct MarioState is laid out as the
 * decompilation has it, and every offset below lines up with those.
 */
#define MARIO_STATE_PTR 0x8032D93C
#define CURR_LEVEL_NUM 0x8032DDF8 /* s16; 16 is the castle grounds */
#define CURR_COURSE_NUM 0x8033BAC6 /* s16 */
#define CURR_ACT_NUM 0x8033BAC8    /* s16 */

#define LEVEL_CASTLE_GROUNDS 16

#define STAR_REWARD 1.0f

/* gSaveBuffer.files[0][0].courseStars: a byte a course (course 1 first), a bit a
 * star (star 1 the lowest). Taking the slide's star turns on bit 1 of course
 * 19's byte the frame the star count goes up. */
#define SAVE_FILE_COURSE_STARS 0x8020770C

/* The castle grounds' warp nodes for the two halves of the front door (see
 * getting there), and where a course's painting puts Mario in every course. */
#define DOOR_WARP_NODE_LEFT 0x80196414
#define DOOR_WARP_NODE_RIGHT 0x80196420
#define DOOR_WARP_TO_CASTLE_LEFT 0x00060100
#define DOOR_WARP_TO_CASTLE_RIGHT 0x01060101
#define PAINTING_NODE 0x0A

/* struct MarioState */
#define M_INPUT 0x02            /* u16, INPUT_* flags */
#define M_FLAGS 0x04            /* u32, MARIO_* flags: caps, sounds, action flags */
#define M_ACTION 0x0C           /* u32 */
#define M_PREV_ACTION 0x10      /* u32 */
#define M_ACTION_STATE 0x18     /* u16 */
#define M_ACTION_TIMER 0x1A     /* u16 */
#define M_ACTION_ARG 0x1C       /* u32 */
#define M_INTENDED_MAG 0x20     /* f32, how far the stick is pushed, 0 to 32 */
#define M_INTENDED_YAW 0x24     /* s16, the world direction the stick asked for */
#define M_INVINC_TIMER 0x26     /* s16 */
#define M_FRAMES_SINCE_A 0x28   /* u8 */
#define M_FRAMES_SINCE_B 0x29   /* u8 */
#define M_WALL_KICK_TIMER 0x2A  /* u8 */
#define M_DOUBLE_JUMP_TIMER 0x2B /* u8 */
#define M_FACE_ANGLE 0x2C       /* Vec3s: pitch, yaw, roll */
#define M_ANGLE_VEL 0x32        /* Vec3s */
#define M_SLIDE_YAW 0x38        /* s16 */
#define M_TWIRL_YAW 0x3A        /* s16 */
#define M_POS 0x3C              /* Vec3f */
#define M_VEL 0x48              /* Vec3f */
#define M_FORWARD_VEL 0x54      /* f32 */
#define M_SLIDE_VEL_X 0x58      /* f32 */
#define M_SLIDE_VEL_Z 0x5C      /* f32 */
#define M_WALL 0x60             /* struct Surface* */
#define M_CEIL 0x64             /* struct Surface* */
#define M_FLOOR 0x68            /* struct Surface* */
#define M_CEIL_HEIGHT 0x6C      /* f32 */
#define M_FLOOR_HEIGHT 0x70     /* f32 */
#define M_FLOOR_ANGLE 0x74      /* s16 */
#define M_WATER_LEVEL 0x76      /* s16 */
#define M_HEALTH 0xAE           /* s16, 0x880 is full and under 0x100 is dead */
#define M_PEAK_HEIGHT 0xBC      /* f32, the top of the current jump or fall */

/* struct Surface */
#define SURFACE_TYPE 0x00       /* s16 */
#define SURFACE_NORMAL 0x1C     /* Vec3f */

/* An action is an index in its low six bits, a group in 0x1C0, and flags above. */
#define ACT_ID_MASK 0x3F
#define ACT_GROUP_MASK 0x1C0
#define ACT_GROUP_SHIFT 6
#define ACT_GROUPS 8
#define ACT_IDS 64
#define ACT_FLAG_FIRST 9
#define ACT_FLAGS 23 /* bits 9 to 31 */
#define ACT_FLAG_AIR (1 << 11)
#define ACT_IDLE 0x0C400201
#define ACT_WATER_IDLE 0x380022C0
#define ACT_FLYING 0x10880899

#define INPUT_BITS 16

/* libultra's button bits, as the pad reports them. */
#define BUTTON_A 0x8000
#define BUTTON_B 0x4000
#define BUTTON_Z 0x2000
#define BUTTON_START 0x1000

#define STICK_DIRECTIONS 16
#define ACTION_HEADS 4

/* --- the courses ------------------------------------------------------------ */

typedef struct {
    const char* name;  /* what a star's name starts with: pss in pss-2 */
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

typedef struct {
    const SM64Course* course;
    int star;       /* 1 to course->stars; 0 is any star in the course */
    int act;        /* the act it is entered for; 0 for a course with no act select */
    char name[32];  /* its folder under SM64_DATA */
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
        snprintf(error, size, "there is no course called '%.*s'", (int)length, star);
        return 0;
    }
    const SM64Course* course = goal->course;
    if (dash != NULL) {
        char* end;
        long number = strtol(dash + 1, &end, 10);
        if (end == dash + 1 || *end != '\0' || number < 1 || number > course->stars) {
            snprintf(error, size, "'%s': %s has stars 1 to %d", star, course->title, course->stars);
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
    int written = snprintf(goal->name, sizeof(goal->name), "%s", star);
    if (act != own && written > 0 && (size_t)written < sizeof(goal->name)) {
        snprintf(goal->name + written, sizeof(goal->name) - (size_t)written, "-act%d", act);
    }
    return 1;
}

/* --- the observation, in the order it is written ----------------------------- */
enum {
    OBS_POS_X, OBS_POS_Y, OBS_POS_Z,
    OBS_VEL_X, OBS_VEL_Y, OBS_VEL_Z,
    OBS_FORWARD_VEL,
    OBS_SPEED,               /* ground covered last step, per frame */
    OBS_FACE_SIN, OBS_FACE_COS,
    OBS_PITCH, OBS_ROLL,
    OBS_ANGLE_VEL_X, OBS_ANGLE_VEL_Y, OBS_ANGLE_VEL_Z,
    OBS_SLIDE_YAW_SIN, OBS_SLIDE_YAW_COS, /* relative to the facing */
    OBS_SLIDE_VEL_X, OBS_SLIDE_VEL_Z,
    OBS_TWIRL_YAW_SIN, OBS_TWIRL_YAW_COS,
    OBS_INTENDED_MAG,
    OBS_INTENDED_SIN, OBS_INTENDED_COS,   /* relative to the facing */
    OBS_CAMERA_SIN, OBS_CAMERA_COS,       /* relative to the facing */
    OBS_HEIGHT_ABOVE_FLOOR,
    OBS_FLOOR_HEIGHT,
    OBS_HEADROOM,
    OBS_FLOOR, OBS_FLOOR_NX, OBS_FLOOR_NY, OBS_FLOOR_NZ, OBS_FLOOR_ANGLE_SIN, OBS_FLOOR_ANGLE_COS, OBS_FLOOR_TYPE,
    OBS_WALL, OBS_WALL_NX, OBS_WALL_NY, OBS_WALL_NZ,
    OBS_CEIL,
    OBS_WATER,
    OBS_HEALTH,
    OBS_PEAK_ABOVE,          /* how far below the top of his jump or fall he is */
    OBS_FRAMES_SINCE_A, OBS_FRAMES_SINCE_B,
    OBS_WALL_KICK_TIMER, OBS_DOUBLE_JUMP_TIMER, OBS_INVINC_TIMER,
    OBS_ACTION_TIMER, OBS_ACTION_STATE, OBS_ACTION_ARG,
    OBS_ACTION_GROUP,                              /* one-hot, ACT_GROUPS */
    OBS_ACTION_ID = OBS_ACTION_GROUP + ACT_GROUPS, /* one-hot, ACT_IDS */
    OBS_ACTION_FLAGS = OBS_ACTION_ID + ACT_IDS,    /* bits 9 to 31 */
    OBS_PREV_GROUP = OBS_ACTION_FLAGS + ACT_FLAGS, /* one-hot */
    OBS_PREV_ID = OBS_PREV_GROUP + ACT_GROUPS,     /* one-hot */
    OBS_INPUT = OBS_PREV_ID + ACT_IDS,             /* INPUT_* bits */
    OBS_MARIO_FLAGS = OBS_INPUT + INPUT_BITS,      /* MARIO_* bits, the low 16 */
    OBS_STAR_DX = OBS_MARIO_FLAGS + 16,
    OBS_STAR_DY, OBS_STAR_DZ,
    OBS_STAR_DISTANCE,
    OBS_STAR_SIN, OBS_STAR_COS, /* the star's direction, relative to the facing */
    OBS_STAR_PROGRESS,          /* the closest he has been this episode, as a share of the way */
    OBS_TIME_LEFT,
    NUM_OBS
};

#define OBS_SIZE NUM_OBS
#define NUM_ATNS ACTION_HEADS
/* stick direction (none, then sixteen ways round), then A, B and Z */
#define ACT_SIZES {STICK_DIRECTIONS + 1, 2, 2, 2}

/* Required struct. Only use floats! */
struct Log {
    float perf;            /* fraction of episodes that took the star */
    float score;           /* fraction of the clock left when he took it, 0 if he did not */
    float episode_return;
    float episode_length;  /* agent steps */
    float frames;          /* frames to the star, the whole clock without it */
    float progress;        /* share of the way to the star covered at the closest */
    float closest;         /* the closest he got to the star, in units */
    float progress_reward; /* what getting closer paid this episode */
    float top_speed;
    float airborne;        /* fraction of frames off the ground */
    float ended_early;     /* fraction of episodes cut short by a death or a warp out */
    float n;               /* Required as the last field */
};

/* Required struct */
struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;     /* vecenv sets this to the env's index; one game per index */

    SM64Goal goal;
    int frameskip;        /* frames of the game per agent step */
    int max_ticks;        /* frames of the game per episode */
    int random_start;     /* at most this many frames of a random stick after loading */
    float time_penalty;   /* reward the whole clock costs */
    float progress;       /* reward the whole way to the star is worth */
    float star_x, star_y, star_z;
    int window;           /* draw the game, for watching a policy play */

    N64Gym gym;
    int opened;
    int star_flags_at_start;
    uint32_t mario;       /* where MarioState is */
    unsigned seed;
    int camera_offset;    /* between the stick and the world, worked out as we go */
    int last_stick_angle;
    int had_stick;
    int tick;             /* frames on the clock */
    int steps;
    float start_distance;
    float best_distance;
    float last_x, last_z;
    float speed;
    float episode_return;
    float progress_earned;
    float top_speed;
    int airborne_frames;
    double next_frame_time;
};
typedef Env SM64;

/* --- reading the game -------------------------------------------------------- */

static inline uint32_t sm64_action(SM64* env) { return n64_u32(&env->gym, env->mario + M_ACTION); }
static inline float sm64_pos(SM64* env, int axis) { return n64_f32(&env->gym, env->mario + M_POS + 4 * axis); }
static inline float sm64_vel(SM64* env, int axis) { return n64_f32(&env->gym, env->mario + M_VEL + 4 * axis); }
static inline int sm64_face_yaw(SM64* env) { return n64_s16(&env->gym, env->mario + M_FACE_ANGLE + 2); }
static inline int sm64_level(SM64* env) { return n64_s16(&env->gym, CURR_LEVEL_NUM); }
static inline float sm64_radians(int angle) { return (float)angle * (2.0f * (float)M_PI / 65536.0f); }
static inline float sm64_clamp(float x, float lo, float hi) { return x < lo ? lo : x > hi ? hi : x; }

/* Whether there is any floor under him at all: null only off the side of a
 * course, falling to the death plane. */
static inline int sm64_in_the_world(SM64* env) {
    return n64_is_ram(n64_u32(&env->gym, env->mario + M_FLOOR));
}

static inline int sm64_star_flags(SM64* env) {
    return n64_u8(&env->gym, SAVE_FILE_COURSE_STARS + (uint32_t)(env->goal.course->course - 1));
}

/* The star, the frame it is his. Any other star in the course is not it, and
 * taking one throws Mario out of the course, which ends the episode as a death does. */
static int sm64_goal_reached(SM64* env) {
    int wanted = env->goal.star > 0 ? 1 << (env->goal.star - 1) : 0x7F;
    return (sm64_star_flags(env) & ~env->star_flags_at_start & wanted) != 0;
}

static float sm64_star_distance(SM64* env) {
    float dx = env->star_x - sm64_pos(env, 0);
    float dy = env->star_y - sm64_pos(env, 1);
    float dz = env->star_z - sm64_pos(env, 2);
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

/* --- the controller ---------------------------------------------------------
 *
 * The game turns a stick direction into a direction to run in by adding the
 * camera's yaw to it, so an action names a direction relative to the way Mario
 * faces, and the camera is measured: whatever angle we asked for last frame,
 * the game wrote down what it understood by it in intendedYaw, and the
 * difference is the camera. One frame of lag, and no camera address needed.
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
    /* The game reads the stick as atan2s(-stickY, stickX). */
    *stick_x = sinf(sm64_radians(angle));
    *stick_y = -cosf(sm64_radians(angle));
}

static void sm64_watch_camera(SM64* env) {
    if (!env->had_stick || n64_f32(&env->gym, env->mario + M_INTENDED_MAG) <= 0.0f) {
        return;
    }
    int intended = n64_s16(&env->gym, env->mario + M_INTENDED_YAW);
    env->camera_offset = (int16_t)(intended - env->last_stick_angle);
}

/* --- getting there --------------------------------------------------------------
 *
 * Two presses of Start reach the file select and open the first file; mashing A
 * through Peach's letter and Lakitu takes about 1500 frames and ends with Mario
 * standing outside the castle. Then the castle's front door is pointed at the
 * star's course and Mario is put in front of it with the stick pushed forward.
 * He opens it, the act select comes up (A, with the star's act written over the
 * one a new save offers), and he lands where the painting would drop him. Done
 * once a star and saved; every episode starts from it.
 */
static int sm64_arrived(uint32_t action) {
    return action == ACT_IDLE || action == ACT_WATER_IDLE || action == ACT_FLYING;
}

/* Whether Mario answers the pad: under a course's opening camera he does not. */
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
    int in_grounds = 0;
    for (int i = 0; i < 4000 && !in_grounds; i++) {
        uint16_t buttons = 0;
        if (i == 200 || i == 260) {
            buttons = BUTTON_START;
        } else if (i > 300 && (i % 8) < 2) {
            buttons = BUTTON_A;
        }
        n64gym_pad(gym, buttons, 0.0f, 0.0f);
        if (!n64gym_step(gym, 1)) {
            snprintf(error, error_size, "the game stopped while it was being started up: %s", gym->error);
            return 0;
        }
        uint32_t mario = n64_u32(gym, MARIO_STATE_PTR);
        in_grounds = i > 400 && n64_is_ram(mario) && n64_s16(gym, CURR_LEVEL_NUM) == LEVEL_CASTLE_GROUNDS &&
                     n64_u32(gym, mario + M_ACTION) == ACT_IDLE;
    }
    if (!in_grounds) {
        snprintf(error, error_size, "the game never reached the castle grounds; it may be waiting on something");
        return 0;
    }
    n64gym_pad(gym, 0, 0.0f, 0.0f);
    if (!n64gym_step(gym, 30)) {
        snprintf(error, error_size, "%s", gym->error);
        return 0;
    }

    if (n64_u32(gym, DOOR_WARP_NODE_LEFT) != DOOR_WARP_TO_CASTLE_LEFT ||
        n64_u32(gym, DOOR_WARP_NODE_RIGHT) != DOOR_WARP_TO_CASTLE_RIGHT) {
        snprintf(error, error_size, "the castle door's warp nodes are not at %08x and %08x", DOOR_WARP_NODE_LEFT,
                 DOOR_WARP_NODE_RIGHT);
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
        n64gym_pad(gym, (selects_act && now_in == level && !spawned && (i % 16) < 2) ? BUTTON_A : 0, 0.0f,
                   now_in == LEVEL_CASTLE_GROUNDS ? 1.0f : 0.0f);
        if (!n64gym_step(gym, 1)) {
            snprintf(error, error_size, "the game stopped on the way to %s: %s", goal->course->title, gym->error);
            return 0;
        }
        if (selects_act && n64_s16(gym, CURR_LEVEL_NUM) == level) {
            n64_set_s16(gym, CURR_ACT_NUM, (int16_t)act);
        }
        mario = n64_u32(gym, MARIO_STATE_PTR);
        uint32_t action = n64_is_ram(mario) ? n64_u32(gym, mario + M_ACTION) : 0;
        spawned = spawned || (n64_s16(gym, CURR_LEVEL_NUM) == level && (action & (ACT_FLAG_AIR | (1 << 13))));
        landed = spawned && sm64_arrived(action);
    }
    if (!landed || n64_s16(gym, CURR_LEVEL_NUM) != level || n64_s16(gym, CURR_COURSE_NUM) != goal->course->course ||
        (selects_act && n64_s16(gym, CURR_ACT_NUM) != act)) {
        snprintf(error, error_size, "the door never left Mario standing in %s", goal->course->title);
        return 0;
    }
    /* The course's opening camera holds Mario still until a button is pressed,
     * and a state saved before then loads into a game where he never moves. */
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

/* --- where a star keeps its savestate ----------------------------------------- */

static const char* sm64_setting(const char* name, const char* fallback) {
    const char* found = getenv(name);
    return (found != NULL && found[0] != '\0') ? found : fallback;
}

static char sm64_state_file[1100];
static pthread_mutex_t sm64_state_lock = PTHREAD_MUTEX_INITIALIZER;

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

static void sm64_open(SM64* env) {
    N64GymOptions options;
    memset(&options, 0, sizeof(options));
    options.host = sm64_setting("SM64_HOST", SM64_HOST);
    options.module = sm64_setting("SM64_MODULE", SM64_MODULE);
    options.rom = sm64_setting("SM64_ROM", SM64_ROM);
    options.game_id = "NSME";
    options.config_dir = sm64_setting("SM64_CONFIG_DIR", SM64_CONFIG_DIR);
    options.log = sm64_setting("SM64_LOG", SM64_LOG);
    options.windowed = env->window;
    options.picture = 0;
    options.index = (int)env->rng;
    if (!n64gym_open(&env->gym, &options)) {
        fprintf(stderr, "sm64: %s\n      host   %s\n      module %s\n      rom    %s\n", env->gym.error,
                options.host, options.module, options.rom);
        exit(1);
    }
    env->opened = 1;
    n64gym_draw(&env->gym, N64B_GYM_DRAW_LAST_FRAME);
    env->seed = 0x9E3779B9u ^ (env->rng * 2654435761u);

    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/%s", sm64_setting("SM64_DATA", SM64_DATA), env->goal.name);
    pthread_mutex_lock(&sm64_state_lock);
    snprintf(sm64_state_file, sizeof(sm64_state_file), "%s/start.state", dir);
    struct stat ignored;
    if (stat(sm64_state_file, &ignored) != 0) {
        sm64_make_dirs(dir);
        char error[256];
        fprintf(stderr, "sm64: playing through the intro and into %s once, to make %s\n", env->goal.course->title,
                sm64_state_file);
        if (!sm64_make_state(&env->gym, sm64_state_file, &env->goal, error, sizeof(error))) {
            fprintf(stderr, "sm64: %s\n", error);
            exit(1);
        }
    }
    pthread_mutex_unlock(&sm64_state_lock);

    /* A game that has just booted is once in a while not yet ready to load a
     * state, and says so. A frame later it is. */
    int loaded = n64gym_load_state(&env->gym, sm64_state_file);
    for (int tries = 0; !loaded && tries < 30; tries++) {
        n64gym_step(&env->gym, 1);
        loaded = n64gym_load_state(&env->gym, sm64_state_file);
    }
    if (!loaded) {
        fprintf(stderr, "sm64: could not start from %s: %s\n", sm64_state_file, env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    env->star_flags_at_start = sm64_star_flags(env);
    if (sm64_level(env) != env->goal.course->level ||
        n64_s16(&env->gym, CURR_COURSE_NUM) != env->goal.course->course ||
        (env->goal.act > 0 && n64_s16(&env->gym, CURR_ACT_NUM) != env->goal.act)) {
        fprintf(stderr, "sm64: %s is not in %s; delete it to make it again\n", sm64_state_file,
                env->goal.course->title);
        exit(1);
    }
}

/* --- the observation ------------------------------------------------------------ */

static inline float sm64_bit(uint32_t word, int bit) { return (word >> bit) & 1 ? 1.0f : 0.0f; }

static void sm64_surface_normal(SM64* env, uint32_t surface, float* out) {
    for (int k = 0; k < 3; k++) {
        out[k] = n64_f32(&env->gym, surface + SURFACE_NORMAL + 4 * k);
    }
}

static void sm64_write_observations(SM64* env) {
    obs_t* obs = env->agents[0].observations;
    N64Gym* gym = &env->gym;
    uint32_t m = env->mario;
    memset(obs, 0, NUM_OBS * sizeof(obs_t));

    float x = sm64_pos(env, 0), y = sm64_pos(env, 1), z = sm64_pos(env, 2);
    obs[OBS_POS_X] = x / 8192.0f;
    obs[OBS_POS_Y] = y / 8192.0f;
    obs[OBS_POS_Z] = z / 8192.0f;
    obs[OBS_VEL_X] = sm64_clamp(sm64_vel(env, 0) / 100.0f, -4.0f, 4.0f);
    obs[OBS_VEL_Y] = sm64_clamp(sm64_vel(env, 1) / 100.0f, -4.0f, 4.0f);
    obs[OBS_VEL_Z] = sm64_clamp(sm64_vel(env, 2) / 100.0f, -4.0f, 4.0f);
    obs[OBS_FORWARD_VEL] = sm64_clamp(n64_f32(gym, m + M_FORWARD_VEL) / 100.0f, -4.0f, 4.0f);
    obs[OBS_SPEED] = sm64_clamp(env->speed / 100.0f, 0.0f, 4.0f);

    int yaw = sm64_face_yaw(env);
    obs[OBS_FACE_SIN] = sinf(sm64_radians(yaw));
    obs[OBS_FACE_COS] = cosf(sm64_radians(yaw));
    obs[OBS_PITCH] = (float)n64_s16(gym, m + M_FACE_ANGLE) / 32768.0f;
    obs[OBS_ROLL] = (float)n64_s16(gym, m + M_FACE_ANGLE + 4) / 32768.0f;
    for (int k = 0; k < 3; k++) {
        obs[OBS_ANGLE_VEL_X + k] = sm64_clamp((float)n64_s16(gym, m + M_ANGLE_VEL + 2 * k) / 4096.0f, -4.0f, 4.0f);
    }
    int slide_yaw = n64_s16(gym, m + M_SLIDE_YAW) - yaw;
    obs[OBS_SLIDE_YAW_SIN] = sinf(sm64_radians(slide_yaw));
    obs[OBS_SLIDE_YAW_COS] = cosf(sm64_radians(slide_yaw));
    obs[OBS_SLIDE_VEL_X] = sm64_clamp(n64_f32(gym, m + M_SLIDE_VEL_X) / 100.0f, -4.0f, 4.0f);
    obs[OBS_SLIDE_VEL_Z] = sm64_clamp(n64_f32(gym, m + M_SLIDE_VEL_Z) / 100.0f, -4.0f, 4.0f);
    int twirl = n64_s16(gym, m + M_TWIRL_YAW);
    obs[OBS_TWIRL_YAW_SIN] = sinf(sm64_radians(twirl));
    obs[OBS_TWIRL_YAW_COS] = cosf(sm64_radians(twirl));
    obs[OBS_INTENDED_MAG] = n64_f32(gym, m + M_INTENDED_MAG) / 32.0f;
    int intended = n64_s16(gym, m + M_INTENDED_YAW) - yaw;
    obs[OBS_INTENDED_SIN] = sinf(sm64_radians(intended));
    obs[OBS_INTENDED_COS] = cosf(sm64_radians(intended));
    obs[OBS_CAMERA_SIN] = sinf(sm64_radians(env->camera_offset - yaw));
    obs[OBS_CAMERA_COS] = cosf(sm64_radians(env->camera_offset - yaw));

    float floor_height = n64_f32(gym, m + M_FLOOR_HEIGHT);
    obs[OBS_HEIGHT_ABOVE_FLOOR] = sm64_clamp((y - floor_height) / 500.0f, -4.0f, 4.0f);
    obs[OBS_FLOOR_HEIGHT] = sm64_clamp(floor_height / 8192.0f, -2.0f, 2.0f);
    obs[OBS_HEADROOM] = sm64_clamp((n64_f32(gym, m + M_CEIL_HEIGHT) - y) / 1000.0f, -4.0f, 4.0f);
    uint32_t floor = n64_u32(gym, m + M_FLOOR);
    if (n64_is_ram(floor)) {
        obs[OBS_FLOOR] = 1.0f;
        sm64_surface_normal(env, floor, &obs[OBS_FLOOR_NX]);
        obs[OBS_FLOOR_TYPE] = (float)(n64_s16(gym, floor + SURFACE_TYPE) & 0xFF) / 64.0f;
    }
    int floor_angle = n64_s16(gym, m + M_FLOOR_ANGLE) - yaw;
    obs[OBS_FLOOR_ANGLE_SIN] = sinf(sm64_radians(floor_angle));
    obs[OBS_FLOOR_ANGLE_COS] = cosf(sm64_radians(floor_angle));
    uint32_t wall = n64_u32(gym, m + M_WALL);
    if (n64_is_ram(wall)) {
        obs[OBS_WALL] = 1.0f;
        sm64_surface_normal(env, wall, &obs[OBS_WALL_NX]);
    }
    obs[OBS_CEIL] = n64_is_ram(n64_u32(gym, m + M_CEIL)) ? 1.0f : 0.0f;
    obs[OBS_WATER] = sm64_clamp(((float)n64_s16(gym, m + M_WATER_LEVEL) - y) / 500.0f, -2.0f, 2.0f);
    obs[OBS_HEALTH] = (float)n64_s16(gym, m + M_HEALTH) / 2176.0f;
    obs[OBS_PEAK_ABOVE] = sm64_clamp((n64_f32(gym, m + M_PEAK_HEIGHT) - y) / 1000.0f, -4.0f, 4.0f);
    obs[OBS_FRAMES_SINCE_A] = (float)n64_u8(gym, m + M_FRAMES_SINCE_A) / 255.0f;
    obs[OBS_FRAMES_SINCE_B] = (float)n64_u8(gym, m + M_FRAMES_SINCE_B) / 255.0f;
    obs[OBS_WALL_KICK_TIMER] = (float)n64_u8(gym, m + M_WALL_KICK_TIMER) / 5.0f;
    obs[OBS_DOUBLE_JUMP_TIMER] = (float)n64_u8(gym, m + M_DOUBLE_JUMP_TIMER) / 5.0f;
    obs[OBS_INVINC_TIMER] = sm64_clamp((float)n64_s16(gym, m + M_INVINC_TIMER) / 60.0f, 0.0f, 4.0f);
    obs[OBS_ACTION_TIMER] = sm64_clamp((float)n64_u16(gym, m + M_ACTION_TIMER) / 60.0f, 0.0f, 4.0f);
    obs[OBS_ACTION_STATE] = sm64_clamp((float)n64_u16(gym, m + M_ACTION_STATE) / 8.0f, 0.0f, 4.0f);
    obs[OBS_ACTION_ARG] = sm64_clamp((float)n64_u32(gym, m + M_ACTION_ARG) / 8.0f, 0.0f, 4.0f);

    uint32_t action = sm64_action(env);
    obs[OBS_ACTION_GROUP + ((action & ACT_GROUP_MASK) >> ACT_GROUP_SHIFT)] = 1.0f;
    obs[OBS_ACTION_ID + (action & ACT_ID_MASK)] = 1.0f;
    for (int k = 0; k < ACT_FLAGS; k++) {
        obs[OBS_ACTION_FLAGS + k] = sm64_bit(action, ACT_FLAG_FIRST + k);
    }
    uint32_t prev = n64_u32(gym, m + M_PREV_ACTION);
    obs[OBS_PREV_GROUP + ((prev & ACT_GROUP_MASK) >> ACT_GROUP_SHIFT)] = 1.0f;
    obs[OBS_PREV_ID + (prev & ACT_ID_MASK)] = 1.0f;
    uint32_t input = n64_u16(gym, m + M_INPUT);
    uint32_t flags = n64_u32(gym, m + M_FLAGS);
    for (int k = 0; k < 16; k++) {
        obs[OBS_INPUT + k] = sm64_bit(input, k);
        obs[OBS_MARIO_FLAGS + k] = sm64_bit(flags, k);
    }

    float dx = env->star_x - x, dy = env->star_y - y, dz = env->star_z - z;
    float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    obs[OBS_STAR_DX] = sm64_clamp(dx / 8192.0f, -2.0f, 2.0f);
    obs[OBS_STAR_DY] = sm64_clamp(dy / 8192.0f, -2.0f, 2.0f);
    obs[OBS_STAR_DZ] = sm64_clamp(dz / 8192.0f, -2.0f, 2.0f);
    obs[OBS_STAR_DISTANCE] = sm64_clamp(distance / 8192.0f, 0.0f, 4.0f);
    /* The game's yaw 0 faces +z and 0x4000 faces +x, so atan2(dx, dz) is a yaw. */
    float to_star = atan2f(dx, dz) - sm64_radians(yaw);
    obs[OBS_STAR_SIN] = sinf(to_star);
    obs[OBS_STAR_COS] = cosf(to_star);
    obs[OBS_STAR_PROGRESS] = env->start_distance > 0.0f ? 1.0f - env->best_distance / env->start_distance : 0.0f;
    obs[OBS_TIME_LEFT] = 1.0f - (float)env->tick / (float)env->max_ticks;
}

/* --- episodes -------------------------------------------------------------------- */

/* Required function */
void puf_reset(SM64* env) {
    if (!n64gym_load_state(&env->gym, sm64_state_file)) {
        fprintf(stderr, "sm64: could not put the game back: %s\n", env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    env->camera_offset = 0;
    env->had_stick = 0;
    if (env->random_start > 0) {
        /* A few frames of some direction, so not every episode is the same one. */
        int frames = (int)(rand_r(&env->seed) % (unsigned)(env->random_start + 1));
        if (frames > 0) {
            float sx, sy;
            sm64_aim(env, 1 + (int)(rand_r(&env->seed) % STICK_DIRECTIONS), &sx, &sy);
            n64gym_pad(&env->gym, 0, sx, sy);
            n64gym_step(&env->gym, frames);
            sm64_watch_camera(env);
        }
    }
    env->tick = 0;
    env->steps = 0;
    env->episode_return = 0.0f;
    env->progress_earned = 0.0f;
    env->top_speed = 0.0f;
    env->speed = 0.0f;
    env->airborne_frames = 0;
    env->last_x = sm64_pos(env, 0);
    env->last_z = sm64_pos(env, 2);
    env->start_distance = sm64_star_distance(env);
    env->best_distance = env->start_distance;
    sm64_write_observations(env);
}

enum { ENDED_CLOCK, ENDED_STAR, ENDED_EARLY };

static void sm64_end_episode(SM64* env, int how) {
    int star = how == ENDED_STAR;
    env->log.perf += (float)star;
    env->log.score += star ? 1.0f - (float)env->tick / (float)env->max_ticks : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->steps;
    env->log.frames += star ? (float)env->tick : (float)env->max_ticks;
    env->log.progress += env->start_distance > 0.0f ? 1.0f - env->best_distance / env->start_distance : 0.0f;
    env->log.closest += env->best_distance;
    env->log.progress_reward += env->progress_earned;
    env->log.top_speed += env->top_speed;
    env->log.airborne += (float)env->airborne_frames / (float)(env->tick > 0 ? env->tick : 1);
    env->log.ended_early += (float)(how == ENDED_EARLY);
    env->log.n += 1.0f;
    env->agents[0].terminals[0] = 1.0f;
    puf_reset(env);
}

/* One decision: hold this on the pad for `frames` frames and pay for what came
 * of it. puf_step is this with the policy's action turned into a pad;
 * sm64_check plays demos through it too. Returns the reward. */
static float sm64_advance(SM64* env, uint16_t buttons, float stick_x, float stick_y, int frames) {
    Agent* agent = &env->agents[0];
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;
    env->steps++;
    n64gym_pad(&env->gym, buttons, stick_x, stick_y);
    if (!n64gym_step(&env->gym, frames)) {
        fprintf(stderr, "sm64: the game stopped: %s\n", env->gym.error);
        exit(1);
    }
    env->tick += frames;
    sm64_watch_camera(env);
    float step_cost = env->time_penalty * (float)frames / (float)env->max_ticks;

    /* The star, the frame he touches it. */
    if (sm64_goal_reached(env)) {
        float reward = STAR_REWARD - step_cost;
        env->episode_return += reward;
        agent->rewards[0] = reward;
        sm64_end_episode(env, ENDED_STAR);
        return reward;
    }
    /* A death, or any other way out of the course, pays the rest of the clock. */
    if (sm64_level(env) != env->goal.course->level || n64_s16(&env->gym, env->mario + M_HEALTH) < 0x100) {
        int left = env->max_ticks - env->tick;
        float reward = -step_cost - env->time_penalty * (float)(left > 0 ? left : 0) / (float)env->max_ticks;
        env->episode_return += reward;
        agent->rewards[0] = reward;
        sm64_end_episode(env, ENDED_EARLY);
        return reward;
    }

    float x = sm64_pos(env, 0), z = sm64_pos(env, 2);
    float moved = sqrtf((x - env->last_x) * (x - env->last_x) + (z - env->last_z) * (z - env->last_z));
    env->last_x = x;
    env->last_z = z;
    env->speed = moved / (float)frames;
    env->top_speed = fmaxf(env->top_speed, env->speed);
    if (sm64_action(env) & ACT_FLAG_AIR) {
        env->airborne_frames += frames;
    }

    float reward = -step_cost;
    if (sm64_in_the_world(env)) {
        float distance = sm64_star_distance(env);
        if (distance < env->best_distance) {
            float paid = env->progress * (env->best_distance - distance) / env->start_distance;
            env->best_distance = distance;
            env->progress_earned += paid;
            reward += paid;
        }
    }
    env->episode_return += reward;
    agent->rewards[0] = reward;

    if (env->tick >= env->max_ticks) {
        sm64_end_episode(env, ENDED_CLOCK);
        return reward;
    }
    sm64_write_observations(env);
    return reward;
}

/* Required function */
void puf_step(SM64* env) {
    Agent* agent = &env->agents[0];
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
    sm64_advance(env, buttons, stick_x, stick_y, env->frameskip);
}

/* Required function. The game draws itself in its own window when made with
 * window = 1; all there is to do here is not run faster than a television. */
void puf_render(SM64* env) {
    if (!env->window) {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double seconds = now.tv_sec + now.tv_nsec / 1e9;
    if (env->next_frame_time == 0.0 || seconds > env->next_frame_time + 1.0) {
        env->next_frame_time = seconds;
    }
    env->next_frame_time += (double)env->frameskip / 30.0;
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
}

/* Required function: read this env's settings from [env], and start its game */
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
    env->progress = (float)dict_get(kwargs, "progress");
    env->star_x = (float)dict_get(kwargs, "star_x");
    env->star_y = (float)dict_get(kwargs, "star_y");
    env->star_z = (float)dict_get(kwargs, "star_z");
    env->window = (int)dict_get(kwargs, "window");
    sm64_open(env);
}

/* Required function: the Log averaged over episodes, by name */
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "frames", log->frames);
    dict_set(out, "progress", log->progress);
    dict_set(out, "closest", log->closest);
    dict_set(out, "progress_reward", log->progress_reward);
    dict_set(out, "top_speed", log->top_speed);
    dict_set(out, "airborne", log->airborne);
    dict_set(out, "ended_early", log->ended_early);
}

#endif
