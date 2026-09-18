/* The vecenv the trainers train on: PufferLib 5.0's CPU vecenv (env_setup and
 * the worker loop in src/pufferl.cu) without the CUDA around it, built into a
 * shared library per env, which the Rust trainer loads (src/vecenv.rs) and
 * mlx_pufferl.py drives through ctypes.
 *
 *   ./build.sh platformer    ->  build/vecenv_platformer.dylib (.so on Linux)
 *
 * One buffer. Envs are made in order until there are total_agents agents, each
 * told its index in rng, and step together on num_threads OpenMP threads. The
 * observations, actions, rewards, terminals and action masks of every agent are
 * flat arrays that the trainer reads and writes in place.
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include ENV_HEADER

#define VEC_API __attribute__((visibility("default")))
#define VEC_KEY PUF_DICT_MAX_KEY

typedef struct {
    Env* envs;
    int num_envs;
    int total_agents;
    int num_threads;
    int mask_size;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    Dict env_kwargs;
} VecEnv;

static const int act_sizes[] = ACT_SIZES;

VEC_API const char* vec_env_name(void) {
    return PUFFER_ENV_NAME;
}

VEC_API int vec_num_atns(void) {
    return NUM_ATNS;
}

VEC_API const int* vec_act_sizes(void) {
    return act_sizes;
}

// 1 for unsigned char observations, 4 for float
VEC_API int vec_obs_bytes(void) {
    return sizeof(obs_t);
}

// Only final once the envs are made: an env may size its observation from its config
VEC_API int vec_obs_size(void) {
    return OBS_SIZE;
}

// keys and values are the [env] section of the config, as written there
VEC_API VecEnv* vec_create(int total_agents, int num_threads, int num_keys,
        const char** keys, const char** values) {
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->env_kwargs.name = dict_strdup("env");
    for (int i = 0; i < num_keys; i++) {
        puf_ini_set(&vec->env_kwargs, keys[i], values[i]);
    }

    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    int agents = 0;
    int num_envs = 0;
    while (agents < total_agents) {
        envs[num_envs].rng = num_envs;
        puf_init(&envs[num_envs], &vec->env_kwargs);
        agents += envs[num_envs].num_agents;
        num_envs++;
    }
    if (agents != total_agents) {
        fprintf(stderr, "vecenv: %d envs make %d agents, not the %d asked for\n",
            num_envs, agents, total_agents);
        exit(1);
    }
    vec->envs = (Env*)realloc(envs, num_envs * sizeof(Env));
    vec->num_envs = num_envs;
    vec->total_agents = total_agents;
    vec->num_threads = num_threads > 0 ? num_threads : 1;

    for (int i = 0; i < NUM_ATNS; i++) {
        vec->mask_size += act_sizes[i];
    }
    vec->observations = (obs_t*)calloc((size_t)total_agents * OBS_SIZE, sizeof(obs_t));
    vec->actions = (float*)calloc((size_t)total_agents * NUM_ATNS, sizeof(float));
    vec->rewards = (float*)calloc(total_agents, sizeof(float));
    vec->terminals = (float*)calloc(total_agents, sizeof(float));
    vec->action_mask = (unsigned char*)malloc((size_t)total_agents * vec->mask_size);
    memset(vec->action_mask, 1, (size_t)total_agents * vec->mask_size);

    int slot = 0;
    for (int e = 0; e < num_envs; e++) {
        Env* env = &vec->envs[e];
        for (int s = 0; s < env->num_agents; s++, slot++) {
            Agent* agent = &env->agents[s];
            agent->observations = vec->observations + (size_t)slot * OBS_SIZE;
            agent->actions = vec->actions + (size_t)slot * NUM_ATNS;
            agent->rewards = vec->rewards + slot;
            agent->terminals = vec->terminals + slot;
            agent->action_mask = vec->action_mask + (size_t)slot * vec->mask_size;
        }
        env->tag = 0;
        env->boundary_reached = 0;
    }
    return vec;
}

VEC_API obs_t* vec_observations(VecEnv* vec) {
    return vec->observations;
}

VEC_API float* vec_actions(VecEnv* vec) {
    return vec->actions;
}

VEC_API float* vec_rewards(VecEnv* vec) {
    return vec->rewards;
}

VEC_API float* vec_terminals(VecEnv* vec) {
    return vec->terminals;
}

VEC_API unsigned char* vec_action_mask(VecEnv* vec) {
    return vec->action_mask;
}

// Fresh episodes everywhere, as env_restart starts a run
VEC_API void vec_reset(VecEnv* vec) {
    memset(vec->rewards, 0, vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, vec->total_agents * sizeof(float));
    #pragma omp parallel for schedule(static) num_threads(vec->num_threads)
    for (int i = 0; i < vec->num_envs; i++) {
        puf_reset(&vec->envs[i]);
    }
}

// One step of every env on the actions in place, as vec_thread_main steps a buffer
VEC_API void vec_step(VecEnv* vec) {
    memset(vec->rewards, 0, vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, vec->total_agents * sizeof(float));
    #pragma omp parallel for schedule(static) num_threads(vec->num_threads)
    for (int i = 0; i < vec->num_envs; i++) {
        puf_step(&vec->envs[i]);
    }
}

/* The Log averaged over the episodes that ended since it was last cleared, by
 * the names puf_log gives it, and n, how many episodes that was: vec_log's
 * aggregate. Writes up to max names (VEC_KEY bytes each) and values, and
 * returns how many it wrote. */
VEC_API int vec_log(VecEnv* vec, int clear, int max, char* keys, double* values) {
    Log aggregate = {0};
    float* acc = (float*)&aggregate;
    int fields = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->num_envs; i++) {
        Log* log = &vec->envs[i].log;
        if (log->n != 0.0f) {
            const float* el = (const float*)log;
            for (int j = 0; j < fields; j++) {
                acc[j] += el[j];
            }
        }
        if (clear) {
            memset(log, 0, sizeof(Log));
        }
    }

    float n = aggregate.n;
    Dict out = {0};
    if (n > 0.0f) {
        for (int j = 0; j < fields; j++) {
            acc[j] /= n;
        }
        puf_log(&aggregate, &out);
    }
    dict_set(&out, "n", n);
    int count = out.size < max ? out.size : max;
    for (int i = 0; i < count; i++) {
        snprintf(keys + (size_t)i * VEC_KEY, VEC_KEY, "%s", out.items[i].key);
        values[i] = out.items[i].value;
    }
    dict_clear(&out);
    return count;
}

VEC_API void vec_render(VecEnv* vec, int env_id) {
    puf_render(&vec->envs[env_id]);
}

// Whether a raylib window an env opened has been closed. An env that draws some other way never is.
VEC_API int vec_window_closed(void) {
    return IsWindowReady() && WindowShouldClose();
}

VEC_API void vec_close(VecEnv* vec) {
    for (int i = 0; i < vec->num_envs; i++) {
        puf_close(&vec->envs[i]);
    }
    free(vec->envs);
    free(vec->observations);
    free(vec->actions);
    free(vec->rewards);
    free(vec->terminals);
    free(vec->action_mask);
    dict_clear(&vec->env_kwargs);
    free(vec);
}
