/* The env without the trainer: make the savestate, watch a game play, or find
 * out how fast this machine can run several of them.
 *
 *   sm64 state           play through the intro once and save the castle grounds
 *   sm64 watch [forward|random|door]   open a window and drive it
 *   sm64 probe [forward|random|door]   step a game headless and print what Mario is doing
 *   sm64 bench [games]   how many agent steps a second, with that many games
 *   sm64 explore [games] [seconds]   Go-Explore with no policy, keeping the fastest door
 *   sm64 replay [watch]  play that door back and check it still opens
 */

#include <omp.h>
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
    env->max_ticks = 900;
    env->random_start = 0;
    env->time_penalty = 1.0f;
    env->novelty = 0.05f;
    env->novelty_episode = 0.02f;
    env->novelty_cell = 500.0f;
    env->go_explore = 0.0f;
    env->go_explore_door = 0.0f;
    env->backward = 0.0f;
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
    printf("step %5d  reward %+6.4f  closest %6.0f  fwd %7.2f  pos (%8.1f %7.1f %8.1f)  action %08x%s\n",
           step, env->rewards[0], env->closest, sm64_forward_vel(env), sm64_pos(env, 0),
           sm64_pos(env, 1), sm64_pos(env, 2), action, (action & ACT_FLAG_AIR) ? "  (in the air)" : "");
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

enum { POLICY_FORWARD, POLICY_RANDOM, POLICY_DOOR };

static int policy_named(const char* name) {
    if (name == NULL) return POLICY_FORWARD;
    if (strcmp(name, "random") == 0) return POLICY_RANDOM;
    if (strcmp(name, "door") == 0) return POLICY_DOOR;
    return POLICY_FORWARD;
}

/* No policy at all, just something to prove the controls reach the game and the
 * reward moves. Forward holds the stick straight ahead and jumps now and then.
 * Door walks to the foot of the bridge, then to the door, and mashes A through
 * Lakitu: the slow way there, and proof that the door ends an episode. */
static void simple_policy(SM64* env, int policy, int step) {
    if (policy == POLICY_RANDOM) {
        env->actions[0] = (float)(rand() % (STICK_DIRECTIONS + 1));
        env->actions[1] = (float)(rand() % 2);
        env->actions[2] = (float)(rand() % 2);
        env->actions[3] = (float)(rand() % 2);
        return;
    }
    env->actions[2] = 0.0f;
    env->actions[3] = 0.0f;
    if (policy == POLICY_FORWARD) {
        env->actions[0] = 1.0f;                        /* straight ahead */
        env->actions[1] = (step % 8 == 0) ? 1.0f : 0.0f; /* and a jump now and then */
        return;
    }
    float x = sm64_pos(env, 0);
    float z = sm64_pos(env, 2);
    float target_z = z > 100.0f ? 0.0f : DOOR_Z;
    int wanted = (int)lroundf(atan2f(DOOR_X - x, target_z - z) * (65536.0f / (2.0f * (float)M_PI)));
    int turn = (int)lroundf((float)(int16_t)(wanted - sm64_face_yaw(env)) / (65536.0f / STICK_DIRECTIONS));
    env->actions[0] = (float)(1 + ((turn % STICK_DIRECTIONS) + STICK_DIRECTIONS) % STICK_DIRECTIONS);
    int talking = (sm64_action(env) & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE;
    env->actions[1] = (talking && step % 2 == 0) ? 1.0f : 0.0f;
}

static int drive(int window, int policy, int steps, int quiet) {
    SM64* env = make_env(0, window);
    c_reset(env);
    double started = now();
    float total = 0.0f;
    int step = 0;
    for (; step < steps; step++) {
        simple_policy(env, policy, step);
        c_step(env);
        c_render(env);
        total += env->rewards[0];
        if (!quiet && step % 30 == 0) {
            show(env, step);
        }
        if (env->terminals[0] != 0.0f) {
            Log* log = &env->log;
            printf("episode over at step %d: %s after %.0f frames, return %.2f, %.0f%% of it in dialog\n",
                   step, log->perf > 0.0f ? "opened the door" : "door still shut",
                   log->frames, log->episode_return, 100.0f * log->dialog);
            memset(log, 0, sizeof(*log));
            total = 0.0f;
            if (policy == POLICY_DOOR && !window) {
                step++;
                break;
            }
        }
    }
    double elapsed = now() - started;
    printf("%d steps in %.2fs (%.0f steps/s, %.0f game frames/s), reward since the last episode %.1f\n",
           step, elapsed, step / elapsed, step * env->frameskip / elapsed, total);
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
            simple_policy(envs[i], POLICY_RANDOM, step);
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

/* Go-Explore's first phase as the paper ran it, with no policy: every episode
 * starts from a cube in the archive and plays at random from there. Any door
 * it opens is offered to the demo file, so this is a way to get a demo without
 * training anything.
 *
 * The stick, B and Z are held for a while rather than jittered, so Mario goes
 * somewhere. A is drawn fresh every step, because Lakitu is in the way. His
 * speech starts and ends inside one cube on the bridge, so the archive cannot
 * keep a place partway through it: an episode has to get from a cube before
 * the bridge to the next one after it, speech and all. With A held for a few
 * steps at a time and 300 frames an episode, none did in 2300 episodes, and
 * nothing came nearer the door than 1088. */
#define EXPLORE_TICKS 900

static void explore_policy(SM64* env) {
    if (rand_r(&env->seed) % 10 == 0) {
        env->actions[0] = (float)(rand_r(&env->seed) % (STICK_DIRECTIONS + 1));
    }
    env->actions[1] = (float)(rand_r(&env->seed) % 2);
    for (int head = 2; head < ACTION_HEADS; head++) {
        if (rand_r(&env->seed) % 10 == 0) {
            env->actions[head] = (float)(rand_r(&env->seed) % 2);
        }
    }
}

static int explore(int games, double seconds) {
    printf("starting %d games\n", games);
    SM64** envs = (SM64**)calloc((size_t)games, sizeof(SM64*));
    for (int i = 0; i < games; i++) {
        envs[i] = make_env(i, 0);
        envs[i]->go_explore = 1.0f;
        envs[i]->go_explore_door = 0.5f;
        envs[i]->max_ticks = EXPLORE_TICKS;
        envs[i]->random_start = 120;
    }
    unsigned episodes = 0, doors = 0, speeches = 0;
    float closest = 1e9f;
    double started = now();
    #pragma omp parallel num_threads(games)
    {
        SM64* env = envs[omp_get_thread_num()];
        c_reset(env);
        double next_report = started + 10.0;
        int talking = 0;
        while (now() < started + seconds) {
            explore_policy(env);
            c_step(env);
            float away = sm64_door_distance(sm64_pos(env, 0), sm64_pos(env, 2));
            if (away < closest) {
                closest = away; /* for the progress line only, so the race is harmless */
            }
            int now_talking = env->terminals[0] == 0.0f && (sm64_action(env) & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE;
            if (talking && !now_talking && env->terminals[0] == 0.0f) {
                __sync_add_and_fetch(&speeches, 1);
            }
            talking = now_talking;
            if (env->terminals[0] != 0.0f) {
                __sync_add_and_fetch(&episodes, 1);
                __sync_add_and_fetch(&doors, (unsigned)env->log.perf);
                memset(&env->log, 0, sizeof(env->log));
            }
            if (env == envs[0] && now() >= next_report) {
                next_report += 10.0;
                char fastest[32] = "none yet";
                if (sm64_demo_best > 0 && sm64_demo_best < INT_MAX) {
                    snprintf(fastest, sizeof(fastest), "%d frames", sm64_demo_best);
                }
                printf("%5.0fs  %5d cubes in the archive  %6u episodes  %5u cutscenes ended  %4u doors  closest %5.0f  fastest door %s\n",
                       now() - started, sm64_archive_size, episodes, speeches, doors, closest, fastest);
                fflush(stdout);
            }
        }
    }
    printf("%u episodes, %u doors; the fastest is in %s\n", episodes, doors, sm64_demo_path());
    for (int i = 0; i < games; i++) {
        c_close(envs[i]);
    }
    return doors > 0 ? 0 : 1;
}

/* Play the demo back from the savestate, and say whether the door opens when it
 * should: the check that a demo is still good for this savestate. */
static int replay(int window) {
    int count, frames;
    SM64Input* inputs = sm64_demo_read(sm64_demo_path(), &count, &frames);
    if (inputs == NULL) {
        printf("no demo in %s: explore first\n", sm64_demo_path());
        return 1;
    }
    printf("%s: %d inputs, the door at frame %d\n", sm64_demo_path(), count, frames);
    SM64* env = make_env(0, window);
    int frame = 0;
    int opened = -1;
    int talking = 0;
    for (int k = 0; k < count && opened < 0; k++) {
        n64gym_pad(&env->gym, inputs[k].buttons, inputs[k].stick_x, inputs[k].stick_y);
        if (!n64gym_step(&env->gym, inputs[k].frames)) {
            printf("the game stopped: %s\n", env->gym.error);
            return 1;
        }
        frame += inputs[k].frames;
        c_render(env);
        uint32_t action = sm64_action(env);
        if (action == ACT_PUSHING_DOOR || action == ACT_PULLING_DOOR || sm64_level(env) == LEVEL_CASTLE) {
            opened = frame;
        } else if (((action & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE) != talking) {
            talking = !talking;
            printf("  frame %4d (%3d before the door): %s at (%.0f %.0f %.0f)\n", frame, frames - frame,
                   talking ? "a cutscene starts" : "the cutscene ends", sm64_pos(env, 0), sm64_pos(env, 1),
                   sm64_pos(env, 2));
        }
    }
    if (opened == frames) {
        printf("the door opened at frame %d, as it should\n", opened);
    } else if (opened >= 0) {
        printf("the door opened at frame %d, not %d\n", opened, frames);
    } else {
        printf("the door never opened; Mario ended at (%.0f %.0f %.0f)\n", sm64_pos(env, 0), sm64_pos(env, 1),
               sm64_pos(env, 2));
    }
    c_close(env);
    free(inputs);
    return opened == frames ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* what = argc > 1 ? argv[1] : "probe";
    if (strcmp(what, "state") == 0) {
        return make_state();
    }
    if (strcmp(what, "watch") == 0) {
        return drive(1, policy_named(argc > 2 ? argv[2] : NULL), argc > 3 ? atoi(argv[3]) : 100000, 1);
    }
    if (strcmp(what, "probe") == 0) {
        return drive(0, policy_named(argc > 2 ? argv[2] : NULL), argc > 3 ? atoi(argv[3]) : 300, 0);
    }
    if (strcmp(what, "bench") == 0) {
        return bench(argc > 2 ? atoi(argv[2]) : 8, argc > 3 ? atoi(argv[3]) : 300);
    }
    if (strcmp(what, "explore") == 0) {
        return explore(argc > 2 ? atoi(argv[2]) : 8, argc > 3 ? atof(argv[3]) : 600.0);
    }
    if (strcmp(what, "replay") == 0) {
        return replay(argc > 2 && strcmp(argv[2], "watch") == 0);
    }
    printf("usage: sm64 [state | watch [forward|random|door] | probe [forward|random|door] | bench [games]\n"
           "            | explore [games] [seconds] | replay [watch]]\n");
    return 2;
}
