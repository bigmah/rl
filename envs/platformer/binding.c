#include "platformer.h"
#define OBS_SIZE NUM_OBS
#define NUM_ATNS 2
#define ACT_SIZES {3, 3}
#define OBS_TENSOR_T FloatTensor

#define Env Platformer
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->max_ticks = dict_get(kwargs, "max_ticks")->value;
    env->death_penalty = dict_get(kwargs, "death_penalty")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "timeouts", log->timeouts);
}
