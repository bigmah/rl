/* Super Mario 64, as fast as Mario will go.
 *
 * The game is the real cartridge, statically recompiled to native arm64 by
 * N64Bundler and run in a process of its own (see n64b_gym.h). This file is the
 * environment around it: what an action does to the controller, what Mario's
 * state is worth, and where all of that lives in the console's memory.
 *
 * Reward is how far Mario actually moved, horizontally, per frame. Not his
 * forward velocity -- that is what the game *believes* about him, and it can be
 * enormous while he stands still grinding into a wall, which is a reward for
 * the wrong thing. Distance covered cannot be faked: it counts a long jump
 * (about 48 units a frame), it counts a slope he slides down, and it counts
 * whatever exploit turns out to beat both. `forward_vel` is logged beside it so
 * the two can be compared.
 *
 * Episodes start from a savestate of the castle grounds with Mario standing
 * outside the castle, so no episode spends its first thousand frames watching
 * Peach's letter. See sm64_make_state.
 *
 * Actions: four discrete heads
 *   stick: 0 none, 1-16 a direction relative to the way Mario is facing
 *          (1 straight ahead, 5 right, 9 back, 13 left)
 *   a, b, z: held or not
 */

#ifndef SM64_H
#define SM64_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

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

#define LEVEL_CASTLE_GROUNDS 16

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
#define M_HEALTH 0xAE      /* s16, 0x880 is full */
#define SURFACE_NORMAL_Y 0x20 /* f32 inside struct Surface */

/* An action is an index in its low bits, a group in 0x1C0, and flags above. */
#define ACT_GROUP_MASK 0x1C0
#define ACT_GROUP_CUTSCENE 0x100 /* dying, warping, star dances: not our episode */
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

/* Required struct. Only use floats! */
typedef struct {
    float perf;            /* average speed as a fraction of a long jump's 48/frame */
    float score;           /* average speed, units per frame */
    float episode_return;
    float episode_length;  /* agent steps */
    float distance;        /* units covered in the episode */
    float top_speed;       /* the best single frame of it */
    float forward_vel;     /* what the game thought his speed was, on average */
    float airborne;        /* fraction of frames off the ground */
    float ended_early;     /* fraction of episodes cut short by a death or a warp */
    float n;               /* Required as the last field */
} Log;

typedef struct {
    Log log;
    void* client;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;     /* vecenv sets this to the env's index; we spawn one game per index */

    int frameskip;        /* frames of the game per agent step */
    int max_ticks;        /* frames of the game per episode */
    int random_start;     /* frames of random stick held after loading the state */
    float speed_scale;    /* units per frame that is worth 1.0 of reward */
    int window;           /* draw the game, for watching a policy play */

    N64Gym gym;
    int opened;
    uint32_t mario;       /* where MarioState is, read once from its pointer */
    float last_x, last_z;
    int camera_offset;    /* between the stick and the world, worked out as we go */
    int last_stick_angle;
    int had_stick;
    int tick;             /* frames this episode */
    int steps;
    float episode_return;
    float distance;
    float top_speed;
    float forward_vel_sum;
    int airborne_frames;
    double next_frame_time; /* for watching it at the speed a television would */
} SM64;

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
static inline int sm64_face_yaw(SM64* env) {
    return n64_s16(&env->gym, env->mario + M_FACE_ANGLE + 2);
}
static inline int sm64_level(SM64* env) { return n64_s16(&env->gym, CURR_LEVEL_NUM); }

/* An angle the game's way: a signed 16-bit turn of the whole circle. */
static inline float sm64_radians(int angle) { return (float)angle * (2.0f * (float)M_PI / 65536.0f); }

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
    *stick_x = sinf(sm64_radians(angle));
    *stick_y = cosf(sm64_radians(angle));
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
 */
static int sm64_make_state(N64Gym* gym, const char* path, char* error, size_t error_size) {
    uint16_t buttons;
    for (int i = 0; i < 4000; i++) {
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
        if (i > 400 && n64_is_ram(mario) && n64_s16(gym, CURR_LEVEL_NUM) == LEVEL_CASTLE_GROUNDS &&
            n64_u32(gym, mario + M_ACTION) == ACT_IDLE) {
            /* Let him settle, with nothing held. */
            n64gym_pad(gym, 0, 0.0f, 0.0f);
            if (!n64gym_step(gym, 30)) {
                snprintf(error, error_size, "%s", gym->error);
                return 0;
            }
            if (!n64gym_save_state(gym, path)) {
                snprintf(error, error_size, "%s", gym->error);
                return 0;
            }
            return 1;
        }
    }
    snprintf(error, error_size,
             "the game never reached the castle grounds; it may be waiting on something");
    return 0;
}

/* --- the environment --------------------------------------------------------- */

static const char* sm64_setting(const char* name, const char* fallback) {
    const char* found = getenv(name);
    return (found != NULL && found[0] != '\0') ? found : fallback;
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
        .index = (int)env->rng,
    };
    if (!n64gym_open(&env->gym, &options)) {
        fprintf(stderr, "sm64: %s\n", env->gym.error);
        fprintf(stderr, "      host   %s\n      module %s\n      rom    %s\n", options.host,
                options.module, options.rom);
        exit(1);
    }
    env->opened = 1;
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);

    const char* state = sm64_setting("SM64_STATE", SM64_STATE);
    struct stat ignored;
    if (stat(state, &ignored) != 0) {
        /* Nobody has played through the intro yet. Do it once, here, and every
         * episode of every run from now on starts from it. */
        char error[256];
        fprintf(stderr, "sm64: playing through the intro once to make %s\n", state);
        if (!sm64_make_state(&env->gym, state, error, sizeof(error))) {
            fprintf(stderr, "sm64: %s\n", error);
            exit(1);
        }
    }
    if (!n64gym_load_state(&env->gym, state)) {
        fprintf(stderr, "sm64: could not start from %s: %s\n", state, env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
}

static void sm64_write_observations(SM64* env, float speed) {
    float* obs = env->observations;
    N64Gym* gym = &env->gym;
    uint32_t action = sm64_action(env);

    memset(obs, 0, NUM_OBS * sizeof(float));
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
    obs[OBS_TIME_LEFT] = 1.0f - (float)env->tick / (float)env->max_ticks;

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
void c_reset(SM64* env) {
    if (!n64gym_load_state(&env->gym, sm64_setting("SM64_STATE", SM64_STATE))) {
        fprintf(stderr, "sm64: could not put the game back: %s\n", env->gym.error);
        exit(1);
    }
    env->mario = n64_u32(&env->gym, MARIO_STATE_PTR);
    env->camera_offset = 0;
    env->had_stick = 0;

    /* A few frames of some direction, so that not every episode is the same
     * episode. Without it there is one starting state and the policy can learn
     * one trajectory through it. */
    if (env->random_start > 0) {
        int frames = (int)(rand() % (unsigned)(env->random_start + 1));
        if (frames > 0) {
            float x, y;
            sm64_aim(env, 1 + (int)(rand() % STICK_DIRECTIONS), &x, &y);
            n64gym_pad(&env->gym, 0, x, y);
            n64gym_step(&env->gym, frames);
            sm64_watch_camera(env);
        }
    }

    env->tick = 0;
    env->steps = 0;
    env->episode_return = 0.0f;
    env->distance = 0.0f;
    env->top_speed = 0.0f;
    env->forward_vel_sum = 0.0f;
    env->airborne_frames = 0;
    env->last_x = sm64_pos(env, 0);
    env->last_z = sm64_pos(env, 2);
    sm64_write_observations(env, 0.0f);
}

static void sm64_end_episode(SM64* env, int early) {
    float frames = (float)(env->tick > 0 ? env->tick : 1);
    float speed = env->distance / frames;
    env->log.perf += speed / 48.0f; /* a long jump, which is about as fast as he goes */
    env->log.score += speed;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->steps;
    env->log.distance += env->distance;
    env->log.top_speed += env->top_speed;
    env->log.forward_vel += env->forward_vel_sum / frames;
    env->log.airborne += (float)env->airborne_frames / frames;
    env->log.ended_early += (float)early;
    env->log.n += 1.0f;
    env->terminals[0] = 1.0f;
    c_reset(env);
}

/* Required function */
void c_step(SM64* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->steps++;

    int direction = (int)env->actions[0];
    if (direction < 0 || direction > STICK_DIRECTIONS) {
        direction = 0;
    }
    uint16_t buttons = 0;
    if ((int)env->actions[1] == 1) buttons |= BUTTON_A;
    if ((int)env->actions[2] == 1) buttons |= BUTTON_B;
    if ((int)env->actions[3] == 1) buttons |= BUTTON_Z;

    float stick_x, stick_y;
    sm64_aim(env, direction, &stick_x, &stick_y);
    n64gym_pad(&env->gym, buttons, stick_x, stick_y);
    if (!n64gym_step(&env->gym, env->frameskip)) {
        fprintf(stderr, "sm64: the game stopped: %s\n", env->gym.error);
        exit(1);
    }
    env->tick += env->frameskip;
    sm64_watch_camera(env);

    uint32_t action = sm64_action(env);
    float x = sm64_pos(env, 0);
    float z = sm64_pos(env, 2);
    float moved = sqrtf((x - env->last_x) * (x - env->last_x) + (z - env->last_z) * (z - env->last_z));
    env->last_x = x;
    env->last_z = z;

    /* A level change is a warp, and a warp is not running. Ending on it also
     * keeps its few thousand units of teleport out of the reward. */
    int warped = sm64_level(env) != LEVEL_CASTLE_GROUNDS ||
                 (action & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE;
    if (warped) {
        sm64_end_episode(env, 1);
        return;
    }

    float speed = moved / (float)env->frameskip;
    env->distance += moved;
    env->forward_vel_sum += fabsf(sm64_forward_vel(env)) * (float)env->frameskip;
    if (action & ACT_FLAG_AIR) {
        env->airborne_frames += env->frameskip;
    }
    if (speed > env->top_speed) {
        env->top_speed = speed;
    }
    env->rewards[0] = moved / env->speed_scale;
    env->episode_return += env->rewards[0];

    if (env->tick >= env->max_ticks) {
        sm64_end_episode(env, 0);
        return;
    }
    sm64_write_observations(env, speed);
}

/* Required function. The game draws itself in its own window when the env was
 * made with `window = 1`; all there is to do here is not run it faster than a
 * television would. */
void c_render(SM64* env) {
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

/* Required function. Do not free observations, actions, rewards, terminals */
void c_close(SM64* env) {
    if (env->opened) {
        n64gym_close(&env->gym);
        env->opened = 0;
    }
}

#endif
