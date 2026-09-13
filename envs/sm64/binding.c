#include "sm64.h"
#define OBS_SIZE NUM_OBS
#define NUM_ATNS ACTION_HEADS
// stick direction (none, then sixteen ways round), then A, B and Z
#define ACT_SIZES {STICK_DIRECTIONS + 1, 2, 2, 2}
#define OBS_TENSOR_T FloatTensor
#define MY_VEC_INIT

#define Env SM64
#include "vecenv.h"

// Every env is a game in a process of its own, and a step of one is a message
// to it and a wait for the answer. So the thread that is waiting is not doing
// any work, and the usual rule -- one thread per core -- would leave all but a
// handful of the games stepping one after another. One thread per game instead:
// they spend their time blocked on a socket, and the games run at once.
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < total_agents) {
        srand(num_envs);
        envs[num_envs].rng = num_envs;
        my_init(&envs[num_envs], env_kwargs);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }
    envs = (Env*)realloc(envs, num_envs * sizeof(Env));

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    omp_set_num_threads(num_envs);
    *num_envs_out = num_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = (int)dict_get(kwargs, "frameskip")->value;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    env->random_start = (int)dict_get(kwargs, "random_start")->value;
    env->speed_scale = (float)dict_get(kwargs, "speed_scale")->value;
    env->window = (int)dict_get(kwargs, "window")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "covered", log->covered);
    dict_set(out, "distance", log->distance);
    dict_set(out, "top_speed", log->top_speed);
    dict_set(out, "forward_vel", log->forward_vel);
    dict_set(out, "airborne", log->airborne);
    dict_set(out, "ended_early", log->ended_early);
}
