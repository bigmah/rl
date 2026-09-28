/* The env without the trainer or a GPU: check a machine runs the game and that
 * the reward does what it says, before spending GPU time on it.
 *
 *   sm64_check replay [file.demo]    play a run of pad inputs through the env's reward,
 *                                    and say whether and when it took the star
 *   sm64_check bench [games] [steps] random actions on that many games at once:
 *                                    agent steps a second, and what the episodes did
 *
 * The config is read the way the trainer reads it, from the directory it is run
 * in: config/default.ini, config/sm64.ini, then --section.key=value flags.
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

static Ini config;

static SM64* make_env(int index) {
    SM64* env = (SM64*)calloc(1, sizeof(SM64));
    Agent* agent = &env->agents[0];
    agent->observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    agent->actions = (float*)calloc(NUM_ATNS, sizeof(float));
    agent->rewards = (float*)calloc(1, sizeof(float));
    agent->terminals = (float*)calloc(1, sizeof(float));
    agent->action_mask = (unsigned char*)calloc(64, 1);
    env->rng = (unsigned)index;
    puf_init(env, puf_ini_section(&config, "env", 0));
    return env;
}

/* A demo: "SM64DEMO", the number of inputs, the frames to the star, then the
 * inputs, each a pad held for some frames. */
typedef struct {
    uint16_t buttons;
    uint16_t frames;
    float stick_x, stick_y;
} DemoInput;

static int replay(const char* path) {
    FILE* file = fopen(path, "rb");
    char magic[8];
    int32_t count, frames;
    if (file == NULL || fread(magic, 8, 1, file) != 1 || memcmp(magic, "SM64DEMO", 8) != 0 ||
        fread(&count, 4, 1, file) != 1 || fread(&frames, 4, 1, file) != 1 || count <= 0) {
        fprintf(stderr, "%s is not a demo\n", path);
        return 1;
    }
    DemoInput* inputs = (DemoInput*)malloc((size_t)count * sizeof(DemoInput));
    if (fread(inputs, sizeof(DemoInput), (size_t)count, file) != (size_t)count) {
        fprintf(stderr, "%s is cut short\n", path);
        return 1;
    }
    fclose(file);
    printf("%s: %d inputs, star at frame %d\n", path, count, frames);

    dict_set(puf_ini_section(&config, "env", 0), "random_start", 0);
    SM64* env = make_env(0);
    puf_reset(env);
    printf("observation: %d numbers; the star is %.0f units away at the start\n", OBS_SIZE, env->start_distance);
    float total = 0.0f, paid_before = 0.0f;
    for (int k = 0; k < count; k++) {
        int tick = env->tick;
        float reward = sm64_advance(env, inputs[k].buttons, inputs[k].stick_x, inputs[k].stick_y, inputs[k].frames);
        total += reward;
        if (env->agents[0].terminals[0] > 0.0f) {
            int ticks = tick + inputs[k].frames;
            printf("the episode ended at frame %d with a reward of %.3f on its last step: %s\n", ticks, reward,
                   env->log.perf > 0.0f ? "the star" : env->log.ended_early > 0.0f ? "a death or a warp" : "the clock");
            printf("return %.3f, of which getting closer paid %.3f; closest %.0f units\n", total,
                   env->log.progress_reward, env->log.closest);
            return env->log.perf > 0.0f && ticks == frames ? 0 : 1;
        }
        if ((k + 1) % 30 == 0) {
            printf("  frame %4d  pos (%7.0f %7.0f %7.0f)  star %6.0f away  return so far %.3f, progress +%.3f\n",
                   env->tick, sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2), sm64_star_distance(env), total,
                   env->progress_earned - paid_before);
            paid_before = env->progress_earned;
        }
    }
    printf("the demo ran out at frame %d without ending the episode; return %.3f\n", env->tick, total);
    return 1;
}

static int bench(int games, int steps) {
    SM64** envs = (SM64**)calloc((size_t)games, sizeof(SM64*));
    double started = now();
    for (int i = 0; i < games; i++) {
        envs[i] = make_env(i);
        puf_reset(envs[i]);
    }
    printf("%d games up in %.1fs; observation is %d numbers\n", games, now() - started, OBS_SIZE);
    int sizes[] = ACT_SIZES;
    started = now();
#pragma omp parallel for num_threads(games) schedule(static, 1)
    for (int i = 0; i < games; i++) {
        unsigned seed = 1234u + (unsigned)i;
        for (int s = 0; s < steps; s++) {
            for (int a = 0; a < NUM_ATNS; a++) {
                envs[i]->agents[0].actions[a] = (float)(rand_r(&seed) % (unsigned)sizes[a]);
            }
            puf_step(envs[i]);
        }
    }
    double seconds = now() - started;
    Log total;
    memset(&total, 0, sizeof(total));
    for (int i = 0; i < games; i++) {
        float* from = (float*)&envs[i]->log;
        float* into = (float*)&total;
        for (size_t f = 0; f < sizeof(Log) / sizeof(float); f++) {
            into[f] += from[f];
        }
    }
    printf("%d agent steps in %.1fs: %.0f a second\n", games * steps, seconds, games * steps / seconds);
    if (total.n > 0.0f) {
        printf("%.0f episodes: return %.3f, progress %.3f (closest %.0f), star %.3f, ended early %.3f, airborne %.2f\n",
               total.n, total.episode_return / total.n, total.progress / total.n, total.closest / total.n,
               total.perf / total.n, total.ended_early / total.n, total.airborne / total.n);
    } else {
        printf("no episode finished in %d steps a game\n", steps);
    }
    for (int i = 0; i < games; i++) {
        puf_close(envs[i]);
    }
    return 0;
}

int main(int argc, char** argv) {
    char* flags[64];
    char* positional[8];
    int nflags = 0, npositional = 0;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--", 2) == 0 && nflags < 64) {
            flags[nflags++] = argv[i];
        } else if (npositional < 8) {
            positional[npositional++] = argv[i];
        }
    }
    puf_ini_load_env(&config, "sm64", nflags, flags);
    const char* command = npositional > 0 ? positional[0] : "";
    if (strcmp(command, "replay") == 0) {
        return replay(npositional > 1 ? positional[1] : "demos/peach-slide/star-674-touch.demo");
    }
    if (strcmp(command, "bench") == 0) {
        return bench(npositional > 1 ? atoi(positional[1]) : 8, npositional > 2 ? atoi(positional[2]) : 1000);
    }
    fprintf(stderr, "usage: sm64_check replay [file.demo] | bench [games] [steps]  [--env.key=value ...]\n");
    return 2;
}
