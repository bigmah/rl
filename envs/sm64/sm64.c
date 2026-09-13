/* The env without the trainer: make the savestate, watch a game play, or find
 * out how fast this machine can run several of them.
 *
 *   sm64 state           play through the intro once and save the castle grounds
 *   sm64 watch [forward|random]   open a window and drive it
 *   sm64 probe           step a game headless and print what Mario is doing
 *   sm64 bench [games]   how many agent steps a second, with that many games
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sm64.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static SM64* make_env(int index, int window) {
    SM64* env = (SM64*)calloc(1, sizeof(SM64));
    env->num_agents = 1;
    env->frameskip = 2;
    env->max_ticks = 1800;
    env->random_start = 0;
    env->speed_scale = 100.0f;
    env->window = window;
    env->rng = (unsigned)index;
    env->observations = (float*)calloc(NUM_OBS, sizeof(float));
    env->actions = (float*)calloc(ACTION_HEADS, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    init(env);
    return env;
}

static void show(SM64* env, int step) {
    uint32_t action = sm64_action(env);
    printf("step %5d  away %+6.2f/frame  covered %6.0f  fwd %7.2f  pos (%8.1f %7.1f %8.1f)  action %08x%s\n",
           step, env->rewards[0] * env->speed_scale / (float)env->frameskip, env->covered,
           sm64_forward_vel(env), sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2), action,
           (action & ACT_FLAG_AIR) ? "  (in the air)" : "");
}

static int make_state(void) {
    const char* path = sm64_setting("SM64_STATE", SM64_STATE);
    N64GymOptions options = {
        .host = sm64_setting("SM64_HOST", SM64_HOST),
        .module = sm64_setting("SM64_MODULE", SM64_MODULE),
        .rom = sm64_setting("SM64_ROM", SM64_ROM),
        .game_id = "NSME",
        .config_dir = sm64_setting("SM64_CONFIG_DIR", SM64_CONFIG_DIR),
        .log = NULL, /* let the game speak, since this is the run that sets it up */
        .index = 0,  /* the first game keeps the runtime's copy of the cartridge */
    };
    N64Gym gym;
    printf("Starting %s\n", options.module);
    double started = now();
    if (!n64gym_open(&gym, &options)) {
        printf("could not start the game: %s\n", gym.error);
        return 1;
    }
    printf("booted in %.2fs; playing through the title, the file select and the intro\n",
           now() - started);

    char error[256];
    if (!sm64_make_state(&gym, path, error, sizeof(error))) {
        printf("%s\n", error);
        n64gym_close(&gym);
        return 1;
    }
    uint32_t mario = n64_u32(&gym, MARIO_STATE_PTR);
    printf("saved %s\n", path);
    printf("  %.1fs of wall clock, %llu frames of game, Mario at (%.0f %.0f %.0f) in level %d\n",
           now() - started, (unsigned long long)n64gym_frames(&gym), n64_f32(&gym, mario + M_POS),
           n64_f32(&gym, mario + M_POS + 4), n64_f32(&gym, mario + M_POS + 8),
           n64_s16(&gym, CURR_LEVEL_NUM));
    n64gym_close(&gym);
    return 0;
}

/* Hold the stick straight ahead and jump when he can: no policy at all, just
 * something to prove the controls reach the game and the reward moves. */
static void simple_policy(SM64* env, int random, int step) {
    if (random) {
        env->actions[0] = (float)(rand() % (STICK_DIRECTIONS + 1));
        env->actions[1] = (float)(rand() % 2);
        env->actions[2] = (float)(rand() % 2);
        env->actions[3] = (float)(rand() % 2);
        return;
    }
    env->actions[0] = 1.0f;                        /* straight ahead */
    env->actions[1] = (step % 8 == 0) ? 1.0f : 0.0f; /* and a jump now and then */
    env->actions[2] = 0.0f;
    env->actions[3] = 0.0f;
}

static int drive(int window, int random, int steps, int quiet) {
    SM64* env = make_env(0, window);
    c_reset(env);
    double started = now();
    float total = 0.0f;
    for (int step = 0; step < steps; step++) {
        simple_policy(env, random, step);
        c_step(env);
        c_render(env);
        total += env->rewards[0];
        if (!quiet && step % 30 == 0) {
            show(env, step);
        }
    }
    double elapsed = now() - started;
    printf("%d steps in %.2fs (%.0f steps/s, %.0f game frames/s), reward %.1f\n", steps, elapsed,
           steps / elapsed, steps * env->frameskip / elapsed, total);
    c_close(env);
    return 0;
}

static int bench(int games, int steps) {
    printf("starting %d games\n", games);
    double started = now();
    SM64** envs = (SM64**)calloc((size_t)games, sizeof(SM64*));
    for (int i = 0; i < games; i++) {
        envs[i] = make_env(i, 0);
        c_reset(envs[i]);
    }
    printf("all up in %.1fs; stepping\n", now() - started);

    started = now();
    for (int step = 0; step < steps; step++) {
#pragma omp parallel for schedule(static) num_threads(games)
        for (int i = 0; i < games; i++) {
            simple_policy(envs[i], 1, step);
            c_step(envs[i]);
        }
    }
    double elapsed = now() - started;
    printf("%d games x %d steps in %.2fs: %.0f agent steps/s, %.0f game frames/s\n", games, steps,
           elapsed, games * steps / elapsed, games * steps * envs[0]->frameskip / elapsed);
    for (int i = 0; i < games; i++) {
        c_close(envs[i]);
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* what = argc > 1 ? argv[1] : "probe";
    if (strcmp(what, "state") == 0) {
        return make_state();
    }
    if (strcmp(what, "watch") == 0) {
        int random = argc > 2 && strcmp(argv[2], "random") == 0;
        return drive(1, random, argc > 3 ? atoi(argv[3]) : 100000, 1);
    }
    if (strcmp(what, "probe") == 0) {
        int random = argc > 2 && strcmp(argv[2], "random") == 0;
        return drive(0, random, argc > 3 ? atoi(argv[3]) : 300, 0);
    }
    if (strcmp(what, "bench") == 0) {
        return bench(argc > 2 ? atoi(argv[2]) : 8, argc > 3 ? atoi(argv[3]) : 300);
    }
    printf("usage: sm64 [state | watch [forward|random] | probe [forward|random] | bench [games]]\n");
    return 2;
}
