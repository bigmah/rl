/* The env without the trainer: make a star's savestate, watch a game play, explore
 * without a policy, play a run back, or find out how fast this machine runs them.
 *
 *   sm64_tool courses                   the courses, and the names their stars go by
 *   sm64_tool state                     get into the star's course and save the game there
 *   sm64_tool watch [forward|random]    open a window and drive it
 *   sm64_tool probe [forward|random]    step a game headless and print what Mario is doing
 *   sm64_tool bench [games]             how many agent steps a second, with that many games
 *   sm64_tool explore [games] [seconds] Go-Explore with no policy, keeping the fastest star
 *   sm64_tool replay [watch] [file]     play the fastest star back and check it still gets there, or any run kept
 *   sm64_tool record [file] [out.mp4]   play a run back headless and write what the console showed, through ffmpeg
 *   sm64_tool trim [file...]            cut runs where the star is taken, for ones kept when it was measured later
 *
 * The star and everything else about the env come from the config, read the way
 * the trainer reads it: PufferLib's default.ini, envs/sm64/sm64.ini, each
 * --config FILE (a star's own settings: envs/sm64/stars/pss-2.ini), then
 * --section.key value flags. So `sm64_tool replay --env.star wf-1` is Whomp's
 * Fortress's first star. Each command then sets what it needs over that: no
 * archive and no demo for a replay, a window for watching.
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

/* --- the config ------------------------------------------------------------------ */

static Ini config;
static char* positional[16];
static int positionals;

/* A --config file over the config so far. Like a flag, it can only set keys the
 * config already has, so a misspelt one is an error rather than ignored. */
static void overlay(const char* path) {
    Ini other = {0};
    puf_ini_load_file(&other, path);
    for (int s = 0; s < other.num_sections; s++) {
        Dict* section = &other.sections[s];
        for (int k = 0; k < section->size; k++) {
            char key[512];
            snprintf(key, sizeof(key), "%s.%s", section->name, section->items[k].key);
            puf_ini_put(&config, key, section->items[k].str);
        }
    }
}

/* default.ini, sm64.ini, each --config in order, then every flag, whatever order
 * they came in: as the trainer does. Anything not a flag is the command's. */
static void read_config(int argc, char** argv) {
    char path[1200];
    const char* root = sm64_setting("PUFFERL_ROOT", PUFFERL_ROOT);
    snprintf(path, sizeof(path), "%s/vendor/PufferLib/config/default.ini", root);
    puf_ini_load_file(&config, path);
    snprintf(path, sizeof(path), "%s/envs/sm64/sm64.ini", root);
    puf_ini_load_file(&config, path);

    char* keys[64];
    char* values[64];
    int flags = 0;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--", 2) != 0) {
            if (positionals < (int)(sizeof(positional) / sizeof(positional[0]))) {
                positional[positionals++] = argv[i];
            }
            continue;
        }
        char* key = argv[i] + 2;
        char* value = "true";
        char* equals = strchr(key, '=');
        if (equals != NULL) {
            *equals = '\0';
            value = equals + 1;
        } else if (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
            value = argv[++i];
        }
        for (char* p = key; *p; p++) {
            *p = *p == '-' ? '_' : *p;
        }
        if (strcmp(key, "config") == 0) {
            overlay(value);
        } else if (flags < 64) {
            keys[flags] = key;
            values[flags++] = value;
        }
    }
    for (int k = 0; k < flags; k++) {
        char full[512];
        snprintf(full, sizeof(full), strchr(keys[k], '.') ? "%s" : "base.%s", keys[k]);
        puf_ini_put(&config, full, values[k]);
    }
}

static Dict* env_config(void) { return puf_ini_section(&config, "env", 0); }
static void env_set(const char* key, double value) { dict_set(env_config(), key, value); }
static double env_get(const char* key) { return dict_get(env_config(), key); }

/* The star the config names, and so where its things are kept, before any game
 * is started. */
static const SM64Goal* goal(void) {
    static SM64Goal goal;
    char error[512];
    if (!sm64_goal_parse(dict_get_str(env_config(), "star"), (int)env_get("act"), &goal, error, sizeof(error))) {
        fprintf(stderr, "sm64: [env] star: %s\n", error);
        exit(1);
    }
    sm64_use_goal(&goal);
    return &goal;
}

static SM64* make_env(int index, int window) {
    env_set("window", window);
    SM64* env = (SM64*)calloc(1, sizeof(SM64));
    env->rng = (unsigned)index;
    int picture = (int)env_get("picture_width") * (int)env_get("picture_height") * PICTURE_CHANNELS;
    env->agents[0] = (Agent){
        .observations = (obs_t*)calloc((size_t)(NUM_OBS + picture), sizeof(obs_t)),
        .actions = (float*)calloc(NUM_ATNS, sizeof(float)),
        .rewards = (float*)calloc(1, sizeof(float)),
        .terminals = (float*)calloc(1, sizeof(float)),
    };
    puf_init(env, env_config());
    return env;
}

/* --- the commands ------------------------------------------------------------------ */

static int courses(void) {
    printf("A star is <course>-<number>, as the act select numbers them (wf-1, pss-2), or a\n"
           "course alone for any star in it (bob). Stars 1 to 6 of a course with an act select\n"
           "are entered for their own act, and the rest for act 1; [env] act enters for another.\n\n");
    printf("  %-6s %-28s %6s %6s %7s\n", "", "", "stars", "level", "acts");
    for (int k = 0; k < SM64_COURSES; k++) {
        const SM64Course* course = &sm64_courses[k];
        printf("  %-6s %-28s %6d %6d %7s\n", course->name, course->title, course->stars, course->level,
               sm64_course_selects_act(course) ? "1 to 6" : "none");
    }
    return 0;
}

static void show(SM64* env, int step) {
    uint32_t action = sm64_action(env);
    printf("step %5d  reward %+6.4f  fwd %7.2f  pos (%8.1f %7.1f %8.1f)  action %08x%s\n", step,
           env->agents[0].rewards[0], sm64_forward_vel(env), sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2),
           action, (action & ACT_FLAG_AIR) ? "  (in the air)" : "");
}

static int make_state(void) {
    const SM64Goal* star = goal();
    const char* path = sm64_state_path();
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
    printf("booted in %.2fs; playing through the title, the file select and the intro, then into %s\n",
           now() - started, star->course->title);

    char error[256];
    if (!sm64_make_state(&gym, path, star, error, sizeof(error))) {
        printf("%s\n", error);
        n64gym_close(&gym);
        return 1;
    }
    uint32_t mario = n64_u32(&gym, MARIO_STATE_PTR);
    printf("saved %s\n", path);
    printf("  %.1fs of wall clock, %llu frames of game, Mario at (%.0f %.0f %.0f) in level %d course %d act %d\n",
           now() - started, (unsigned long long)n64gym_frames(&gym), n64_f32(&gym, mario + M_POS),
           n64_f32(&gym, mario + M_POS + 4), n64_f32(&gym, mario + M_POS + 8), n64_s16(&gym, CURR_LEVEL_NUM),
           n64_s16(&gym, CURR_COURSE_NUM), n64_s16(&gym, CURR_ACT_NUM));
    n64gym_close(&gym);
    return 0;
}

enum { POLICY_FORWARD, POLICY_RANDOM };

static int policy_named(const char* name) {
    return name != NULL && strcmp(name, "random") == 0 ? POLICY_RANDOM : POLICY_FORWARD;
}

/* No policy at all, just something to prove the controls reach the game and the
 * reward moves. Forward holds the stick straight ahead and jumps now and then. */
static void simple_policy(SM64* env, int policy, int step) {
    float* actions = env->agents[0].actions;
    if (policy == POLICY_RANDOM) {
        actions[0] = (float)(rand() % (STICK_DIRECTIONS + 1));
        actions[1] = (float)(rand() % 2);
        actions[2] = (float)(rand() % 2);
        actions[3] = (float)(rand() % 2);
        return;
    }
    actions[0] = 1.0f;                          /* straight ahead */
    actions[1] = (step % 8 == 0) ? 1.0f : 0.0f; /* and a jump now and then */
    actions[2] = 0.0f;
    actions[3] = 0.0f;
}

static int drive(int window, int policy, int steps, int quiet) {
    SM64* env = make_env(0, window);
    puf_reset(env);
    double started = now();
    float total = 0.0f;
    int step = 0;
    for (; step < steps; step++) {
        simple_policy(env, policy, step);
        puf_step(env);
        puf_render(env);
        total += env->agents[0].rewards[0];
        if (!quiet && step % 30 == 0) {
            show(env, step);
        }
        if (env->agents[0].terminals[0] != 0.0f) {
            Log* log = &env->log;
            printf("episode over at step %d: %s after %.0f frames, return %.2f, %.0f deaths\n", step,
                   log->perf > 0.0f ? "took the star" : "no star", log->frames, log->episode_return, log->deaths);
            memset(log, 0, sizeof(*log));
            total = 0.0f;
        }
    }
    double elapsed = now() - started;
    printf("%d steps in %.2fs (%.0f steps/s, %.0f game frames/s), reward since the last episode %.1f\n",
           step, elapsed, step / elapsed, step * env->frameskip / elapsed, total);
    puf_close(env);
    return 0;
}

static int bench(int games, int steps) {
    printf("starting %d games\n", games);
    double started = now();
    SM64** envs = (SM64**)calloc((size_t)games, sizeof(SM64*));
    for (int i = 0; i < games; i++) {
        envs[i] = make_env(i, 0);
        puf_reset(envs[i]);
    }
    printf("all up in %.1fs; stepping\n", now() - started);

    started = now();
    for (int step = 0; step < steps; step++) {
#pragma omp parallel for schedule(static) num_threads(games)
        for (int i = 0; i < games; i++) {
            simple_policy(envs[i], POLICY_RANDOM, step);
            puf_step(envs[i]);
        }
    }
    double elapsed = now() - started;
    printf("%d games x %d steps in %.2fs: %.0f agent steps/s, %.0f game frames/s\n", games, steps,
           elapsed, games * steps / elapsed, games * steps * envs[0]->frameskip / elapsed);
    for (int i = 0; i < games; i++) {
        puf_close(envs[i]);
    }
    return 0;
}

/* Go-Explore's first phase as the paper ran it, with no policy: every episode
 * starts from a cube in the archive and plays at random from there. Any star it
 * takes is offered to the demo file, so this is a way to get a demo without
 * training anything. Episodes are the config's max_ticks long, and a death ends
 * one, so that every episode starts from the archive.
 *
 * The stick, A, B and Z are held for a while rather than jittered, so Mario goes
 * somewhere, and half the time the stick is redrawn it goes straight ahead: a
 * course is covered by going somewhere, and on a slide that is what keeps him
 * sliding. A jump every step is the slowest way down a slide. (On the castle
 * grounds A was drawn fresh every step, to get through Lakitu's speech.) */
static void explore_policy(SM64* env) {
    float* actions = env->agents[0].actions;
    if (rand_r(&env->seed) % 10 == 0) {
        actions[0] = rand_r(&env->seed) % 2 == 0 ? 1.0f : (float)(rand_r(&env->seed) % (STICK_DIRECTIONS + 1));
    }
    for (int head = 1; head < ACTION_HEADS; head++) {
        if (rand_r(&env->seed) % 10 == 0) {
            actions[head] = (float)(rand_r(&env->seed) % 2);
        }
    }
}

static int explore(int games, double seconds) {
    env_set("go_explore", 1.0);
    env_set("go_explore_star", 0.5);
    env_set("random_start", 120);
    env_set("respawn", 0);
    SM64** envs = (SM64**)calloc((size_t)games, sizeof(SM64*));
    for (int i = 0; i < games; i++) {
        envs[i] = make_env(i, 0);
    }
    printf("%d games after %s, %d frames an episode\n", games, envs[0]->goal.name, envs[0]->max_ticks);
    unsigned episodes = 0, stars = 0;
    double started = now();
    #pragma omp parallel num_threads(games)
    {
        SM64* env = envs[omp_get_thread_num()];
        puf_reset(env);
        double next_report = started + 10.0;
        while (now() < started + seconds) {
            explore_policy(env);
            puf_step(env);
            if (env->agents[0].terminals[0] != 0.0f) {
                __sync_add_and_fetch(&episodes, 1);
                __sync_add_and_fetch(&stars, (unsigned)env->log.perf);
                memset(&env->log, 0, sizeof(env->log));
            }
            if (env == envs[0] && now() >= next_report) {
                next_report += 10.0;
                char fastest[32] = "none yet";
                if (sm64_demo_best > 0 && sm64_demo_best < INT_MAX) {
                    snprintf(fastest, sizeof(fastest), "%d frames", sm64_demo_best);
                }
                printf("%5.0fs  %5d cubes in the archive  %6u episodes  %4u stars  fastest %s\n",
                       now() - started, sm64_archive_size, episodes, stars, fastest);
                fflush(stdout);
            }
        }
    }
    printf("%u episodes, %u stars; the fastest is in %s\n", episodes, stars, sm64_demo_path());
    for (int i = 0; i < games; i++) {
        puf_close(envs[i]);
    }
    return stars > 0 ? 0 : 1;
}

/* Play a run back from the savestate, and say whether the star is taken when it
 * should be: the check that a demo is still good for this savestate. Any other
 * run of inputs plays back the same way, and says where it left Mario. */
static int replay(int window, const char* path) {
    int count, frames;
    SM64Input* inputs = sm64_demo_read(path, &count, &frames);
    if (inputs == NULL) {
        printf("no run in %s: explore first\n", path);
        return 1;
    }
    SM64* env = make_env(0, window);
    printf("%s: %d inputs, %d frames\n", path, count, frames);
    /* SM64_TRACE=30 prints where Mario is every thirty frames of the way. */
    int trace = atoi(sm64_setting("SM64_TRACE", "0"));
    int frame = 0;
    int taken = -1;
    int spawned = -1;
    int talking = 0;
    int traced = 0;
    double started = now();
    for (int k = 0; k < count && taken < 0; k++) {
        n64gym_pad(&env->gym, inputs[k].buttons, inputs[k].stick_x, inputs[k].stick_y);
        if (!n64gym_step(&env->gym, inputs[k].frames)) {
            printf("the game stopped: %s\n", env->gym.error);
            return 1;
        }
        frame += inputs[k].frames;
        if (trace > 0 && frame >= traced + trace) {
            traced = frame;
            uint32_t action = sm64_action(env);
            printf("  frame %4d  pos (%8.1f %7.1f %8.1f)  fwd %7.2f  action %08x%s%s\n", frame, sm64_pos(env, 0),
                   sm64_pos(env, 1), sm64_pos(env, 2), sm64_forward_vel(env), action,
                   (action & ACT_FLAG_AIR) ? "  (in the air)" : "", sm64_in_the_world(env) ? "" : "  (no floor)");
        }
        puf_render(env);
        /* In a window it is being watched, so it is played at the speed a
         * console ran it rather than the three times that it replays at. */
        if (window) {
            double late = started + frame / 30.0 - now();
            if (late > 0.0) {
                struct timespec rest = {(time_t)late, (long)((late - (double)(time_t)late) * 1e9)};
                nanosleep(&rest, NULL);
            }
        }
        uint32_t action = sm64_action(env);
        if (spawned < 0 && sm64_star_spawning(env)) {
            spawned = frame;
            printf("  frame %4d (%3d before the end): a star spawns\n", frame, frames - frame);
        }
        if (sm64_goal_reached(env)) {
            taken = frame;
        } else if (((action & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE) != talking) {
            talking = !talking;
            printf("  frame %4d (%3d before the end): %s at (%.0f %.0f %.0f)\n", frame, frames - frame,
                   talking ? "a cutscene starts" : "the cutscene ends", sm64_pos(env, 0), sm64_pos(env, 1),
                   sm64_pos(env, 2));
        }
    }
    if (taken == frames) {
        printf("%s at frame %d, as it should be\n", env->goal.name, taken);
    } else if (taken >= 0) {
        printf("%s at frame %d, not %d%s\n", env->goal.name, taken, frames,
               taken < frames ? " (sm64_tool trim cuts the run there)" : "");
    } else if (spawned == frames) {
        /* A run from when the spawn was the goal: it stops where the star
         * appears, and never takes it. */
        printf("a star spawned at frame %d, as this run says, and is not taken: the run ends at the spawn\n", spawned);
    } else {
        printf("never took %s; Mario ended at (%.0f %.0f %.0f)\n", env->goal.name, sm64_pos(env, 0), sm64_pos(env, 1),
               sm64_pos(env, 2));
    }
    /* Watched, the end is worth seeing too: the star coming down and the dance.
     * Five seconds more, with nothing held. */
    if (window && (taken >= 0 || spawned >= 0)) {
        n64gym_pad(&env->gym, 0, 0.0f, 0.0f);
        for (int k = 0; k < 75; k++) {
            n64gym_step(&env->gym, 2);
            puf_render(env);
        }
    }
    puf_close(env);
    free(inputs);
    return taken == frames || spawned == frames ? 0 : 1;
}

/* One frame of a recording: the picture the console's video interface would
 * show, handed to ffmpeg as raw RGBA. The first one says how big the video is,
 * and starts ffmpeg; the picture is small (the size the game renders headless),
 * so it is scaled up by whole pixels to at least 640 wide, sharp rather than
 * blurred. A blanked screen has no picture and is skipped. */
static int record_frame(SM64* env, FILE** video, int* width, int* height, const char* out) {
    int w, h;
    const uint8_t* picture = n64gym_picture(&env->gym, &w, &h);
    if (picture == NULL) {
        return 1;
    }
    if (*video == NULL) {
        *width = w;
        *height = h;
        int scale = w < 640 ? (640 + w - 1) / w : 1;
        char command[1200];
        snprintf(command, sizeof(command),
                 "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgba -s %dx%d -r 30 -i - "
                 "-vf scale=%d:%d:flags=neighbor -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \"%s\"",
                 w, h, w * scale, h * scale, out);
        *video = popen(command, "w");
        if (*video == NULL) {
            perror("ffmpeg");
            return 0;
        }
        printf("  %dx%d frames, written at %dx%d\n", w, h, w * scale, h * scale);
    } else if (w != *width || h != *height) {
        fprintf(stderr, "the picture changed from %dx%d to %dx%d; stopping there\n", *width, *height, w, h);
        return 0;
    }
    if (fwrite(picture, 4, (size_t)w * h, *video) != (size_t)w * h) {
        fprintf(stderr, "ffmpeg stopped taking frames\n");
        return 0;
    }
    return 1;
}

/* Record a run as a video: play it back headless with the game drawing every
 * frame, and hand each one to ffmpeg at thirty a second, with five seconds more
 * after the star, as watching it gives, so the star dance is in it. The game is
 * stepped a frame at a time so that every frame is drawn; the star is checked
 * where replay checks it, after each input. */
static int record(const char* path, const char* out) {
    int count, frames;
    SM64Input* inputs = sm64_demo_read(path, &count, &frames);
    if (inputs == NULL) {
        printf("no run in %s\n", path);
        return 1;
    }
    /* Only a game opened with a picture draws one. The size here is the
     * observation's, and does not matter: the frame handed over is the
     * console's own. */
    if (env_get("picture_width") <= 0) {
        env_set("picture_width", 80);
        env_set("picture_height", 60);
    }
    SM64* env = make_env(0, 0);
    printf("%s: %d inputs, %d frames -> %s\n", path, count, frames, out);
    FILE* video = NULL;
    int width = 0, height = 0, written = 0, frame = 0, taken = -1, spawned = -1, ok = 1;
    for (int k = 0; k < count && taken < 0 && ok; k++) {
        n64gym_pad(&env->gym, inputs[k].buttons, inputs[k].stick_x, inputs[k].stick_y);
        for (int f = 0; f < inputs[k].frames && ok; f++) {
            if (!n64gym_step(&env->gym, 1)) {
                printf("the game stopped: %s\n", env->gym.error);
                return 1;
            }
            frame++;
            ok = record_frame(env, &video, &width, &height, out);
            written += ok && video != NULL;
        }
        if (spawned < 0 && sm64_star_spawning(env)) {
            spawned = frame;
        }
        if (sm64_goal_reached(env)) {
            taken = frame;
        }
    }
    if (taken == frames) {
        printf("%s at frame %d, as it should be\n", env->goal.name, taken);
    } else if (taken >= 0) {
        printf("%s at frame %d, not %d\n", env->goal.name, taken, frames);
    } else if (spawned == frames) {
        printf("a star spawned at frame %d, as this run says: the run ends at the spawn\n", spawned);
    } else {
        printf("never took %s; Mario ended at (%.0f %.0f %.0f)\n", env->goal.name, sm64_pos(env, 0), sm64_pos(env, 1),
               sm64_pos(env, 2));
    }
    if ((taken >= 0 || spawned >= 0) && ok) {
        n64gym_pad(&env->gym, 0, 0.0f, 0.0f);
        for (int k = 0; k < 150 && ok; k++) {
            n64gym_step(&env->gym, 1);
            ok = record_frame(env, &video, &width, &height, out);
            written += ok && video != NULL;
        }
    }
    int status = 1;
    if (video != NULL) {
        status = pclose(video);
        printf("  %d frames, %.1f seconds, in %s%s\n", written, written / 30.0, out,
               status == 0 ? "" : " (ffmpeg failed)");
    } else {
        printf("  no frames: the game never drew a picture\n");
    }
    puf_close(env);
    free(inputs);
    return (taken == frames || spawned == frames) && ok && status == 0 ? 0 : 1;
}

/* Cut each run where the star is taken, and write it back. For runs kept
 * before the star was measured where it is now. A run whose star is where its
 * header says is left alone. */
static int trim(int count_paths, char** paths) {
    SM64* env = make_env(0, 0);
    int failures = 0;
    for (int p = 0; p < count_paths; p++) {
        int count, frames;
        SM64Input* inputs = sm64_demo_read(paths[p], &count, &frames);
        if (inputs == NULL) {
            printf("%s: not a run\n", paths[p]);
            failures++;
            continue;
        }
        puf_reset(env);
        int frame = 0, reached = -1, kept = count;
        for (int k = 0; k < count && reached < 0; k++) {
            n64gym_pad(&env->gym, inputs[k].buttons, inputs[k].stick_x, inputs[k].stick_y);
            if (!n64gym_step(&env->gym, inputs[k].frames)) {
                printf("the game stopped: %s\n", env->gym.error);
                return 1;
            }
            frame += inputs[k].frames;
            if (sm64_goal_reached(env)) {
                reached = frame;
                kept = k + 1;
            }
        }
        if (reached < 0) {
            printf("%s: never takes the star in %d frames, left as it is\n", paths[p], frames);
            failures++;
        } else if (reached == frames && kept == count) {
            printf("%s: the star is at frame %d as the header says\n", paths[p], frames);
        } else if (sm64_demo_write(paths[p], 0, inputs, kept, reached)) {
            printf("%s: cut at frame %d, from %d inputs and %d frames to %d and %d\n", paths[p], reached, count,
                   frames, kept, reached);
            /* A run in a fastest-runs folder is named for its frames and its
             * inputs (01326-4f16ec26.demo), so it is renamed for the cut. */
            const char* slash = strrchr(paths[p], '/');
            const char* name = slash != NULL ? slash + 1 : paths[p];
            int named_frames;
            unsigned named_hash;
            if (sscanf(name, "%5d-%8x.demo", &named_frames, &named_hash) == 2 && strlen(name) == 19) {
                char renamed[1200];
                snprintf(renamed, sizeof(renamed), "%.*s%05d-%08x.demo", (int)(name - paths[p]), paths[p], reached,
                         sm64_inputs_hash(inputs, kept));
                if (rename(paths[p], renamed) == 0) {
                    printf("  and renamed to %s\n", renamed);
                }
            }
        } else {
            printf("%s: could not write it back\n", paths[p]);
            failures++;
        }
        free(inputs);
    }
    puf_close(env);
    return failures > 0;
}

int main(int argc, char** argv) {
    read_config(argc, argv);
    const char* what = positionals > 0 ? positional[0] : "probe";
    const char* arg1 = positionals > 1 ? positional[1] : NULL;
    const char* arg2 = positionals > 2 ? positional[2] : NULL;
    if (strcmp(what, "courses") == 0) {
        return courses();
    }
    /* No command but explore uses the archive or the demo: a replay is the
     * savestate and the run's inputs, nothing else. */
    goal();
    if (strcmp(what, "explore") != 0) {
        env_set("go_explore", 0.0);
        env_set("go_explore_seed", 0);
    }
    env_set("backward", 0.0);
    env_set("random_start", 0);
    if (strcmp(what, "state") == 0) {
        return make_state();
    }
    if (strcmp(what, "watch") == 0) {
        return drive(1, policy_named(arg1), arg2 ? atoi(arg2) : 100000, 1);
    }
    if (strcmp(what, "probe") == 0) {
        return drive(0, policy_named(arg1), arg2 ? atoi(arg2) : 300, 0);
    }
    if (strcmp(what, "bench") == 0) {
        return bench(arg1 ? atoi(arg1) : 8, arg2 ? atoi(arg2) : 300);
    }
    if (strcmp(what, "explore") == 0) {
        return explore(arg1 ? atoi(arg1) : 8, arg2 ? atof(arg2) : 600.0);
    }
    if (strcmp(what, "replay") == 0) {
        int window = arg1 != NULL && strcmp(arg1, "watch") == 0;
        const char* file = positionals > 1 + window ? positional[1 + window] : sm64_demo_path();
        return replay(window, file);
    }
    if (strcmp(what, "record") == 0) {
        const char* demo = arg1 ? arg1 : sm64_demo_path();
        /* The video goes beside the run, as fastest.mp4 for fastest.demo, unless said. */
        char out[1200];
        if (arg2) {
            snprintf(out, sizeof(out), "%s", arg2);
        } else {
            size_t stem = strlen(demo);
            if (stem >= 5 && strcmp(demo + stem - 5, ".demo") == 0) {
                stem -= 5;
            }
            snprintf(out, sizeof(out), "%.*s.mp4", (int)stem, demo);
        }
        return record(demo, out);
    }
    if (strcmp(what, "trim") == 0) {
        char* demo = (char*)sm64_demo_path();
        return trim(positionals > 1 ? positionals - 1 : 1, positionals > 1 ? positional + 1 : &demo);
    }
    printf("usage: sm64_tool [courses | state | watch [forward|random] | probe [forward|random] | bench [games]\n"
           "                 | explore [games] [seconds] | replay [watch] [file] | record [file] [out.mp4]\n"
           "                 | trim [file...]] [--config FILE] [--env.key value ...]\n");
    return 2;
}
