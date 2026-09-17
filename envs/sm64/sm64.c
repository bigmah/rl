/* The env without the trainer: make the savestate, watch a game play, or find
 * out how fast this machine can run several of them.
 *
 *   sm64 state           play through the intro once and save the castle grounds
 *   sm64 watch [forward|random|door]   open a window and drive it
 *   sm64 probe [forward|random|door]   step a game headless and print what Mario is doing
 *   sm64 bench [games]   how many agent steps a second, with that many games
 *   sm64 explore [games] [seconds] [frames]   Go-Explore with no policy, keeping the fastest door
 *   sm64 replay [watch] [file]  play that door back and check it still opens, or any run exploring kept
 *   sm64 record [file] [out.mp4]  play a demo back headless and write what the console showed, through ffmpeg
 *   sm64 trim [file...]  cut demos where the goal is reached, for ones kept when it was measured later
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
    env->state = 1;
    /* SM64_PICTURE=80x60 draws the game and puts its picture in the observation,
     * as training with picture_width and picture_height does. */
    sscanf(sm64_setting("SM64_PICTURE", "0x0"), "%dx%d", &env->picture_width, &env->picture_height);
    env->rng = (unsigned)index;
    env->agents[0] = (Agent){
        .observations = (obs_t*)calloc(NUM_OBS + env->picture_width * env->picture_height * PICTURE_CHANNELS,
                                       sizeof(obs_t)),
        .actions = (float*)calloc(NUM_ATNS, sizeof(float)),
        .rewards = (float*)calloc(1, sizeof(float)),
        .terminals = (float*)calloc(1, sizeof(float)),
    };
    init(env);
    return env;
}

static void show(SM64* env, int step) {
    uint32_t action = sm64_action(env);
    printf("step %5d  reward %+6.4f  closest %6.0f  fwd %7.2f  pos (%8.1f %7.1f %8.1f)  action %08x%s\n",
           step, env->agents[0].rewards[0], env->closest, sm64_forward_vel(env), sm64_pos(env, 0),
           sm64_pos(env, 1), sm64_pos(env, 2), action, (action & ACT_FLAG_AIR) ? "  (in the air)" : "");
}

static int make_state(void) {
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
    printf("booted in %.2fs; playing through the title, the file select and the intro\n",
           now() - started);

    char error[256];
    if (!sm64_make_state(&gym, path, sm64_star_goal(), sm64_star_level(), sm64_star_act(), error,
                         sizeof(error))) {
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
    float* actions = env->agents[0].actions;
    if (policy == POLICY_RANDOM) {
        actions[0] = (float)(rand() % (STICK_DIRECTIONS + 1));
        actions[1] = (float)(rand() % 2);
        actions[2] = (float)(rand() % 2);
        actions[3] = (float)(rand() % 2);
        return;
    }
    actions[2] = 0.0f;
    actions[3] = 0.0f;
    if (policy == POLICY_FORWARD) {
        actions[0] = 1.0f;                          /* straight ahead */
        actions[1] = (step % 8 == 0) ? 1.0f : 0.0f; /* and a jump now and then */
        return;
    }
    float x = sm64_pos(env, 0);
    float z = sm64_pos(env, 2);
    float target_z = z > 100.0f ? 0.0f : DOOR_Z;
    int wanted = (int)lroundf(atan2f(DOOR_X - x, target_z - z) * (65536.0f / (2.0f * (float)M_PI)));
    int turn = (int)lroundf((float)(int16_t)(wanted - sm64_face_yaw(env)) / (65536.0f / STICK_DIRECTIONS));
    actions[0] = (float)(1 + ((turn % STICK_DIRECTIONS) + STICK_DIRECTIONS) % STICK_DIRECTIONS);
    int talking = (sm64_action(env) & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE;
    actions[1] = (talking && step % 2 == 0) ? 1.0f : 0.0f;
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
 * nothing came nearer the door than 1088.
 *
 * Episodes are EXPLORE_TICKS frames unless the command says otherwise. The
 * archive keeps runs up to three episodes long, so that scales with them. */
#define EXPLORE_TICKS 900

static void explore_policy(SM64* env) {
    float* actions = env->agents[0].actions;
    if (rand_r(&env->seed) % 10 == 0) {
        /* In a course, half the time the stick goes straight ahead: a course is
         * covered by going somewhere, and on a slide that is what keeps him
         * sliding. The grounds are explored as they were, every way alike. */
        actions[0] = (env->star && rand_r(&env->seed) % 2 == 0)
                         ? 1.0f
                         : (float)(rand_r(&env->seed) % (STICK_DIRECTIONS + 1));
    }
    /* A is drawn fresh every step on the way to the door, because Lakitu is in
     * the way. There is no Lakitu in a course, and a jump every step is the
     * slowest way down a slide, so there A is held and let go like the others. */
    if (!env->star) {
        actions[1] = (float)(rand_r(&env->seed) % 2);
    }
    for (int head = env->star ? 1 : 2; head < ACTION_HEADS; head++) {
        if (rand_r(&env->seed) % 10 == 0) {
            actions[head] = (float)(rand_r(&env->seed) % 2);
        }
    }
}

/* Where exploring keeps the run that came closest: beside the demo, star.demo -> star-closest.demo. */
static void closest_path(char* out, size_t size) {
    const char* demo = sm64_demo_path();
    size_t stem = strlen(demo);
    if (stem >= 5 && strcmp(demo + stem - 5, ".demo") == 0) {
        stem -= 5;
    }
    snprintf(out, size, "%.*s-closest.demo", (int)stem, demo);
}

static int explore(int games, double seconds, int ticks) {
    printf("starting %d games, %d frames an episode\n", games, ticks);
    SM64** envs = (SM64**)calloc((size_t)games, sizeof(SM64*));
    for (int i = 0; i < games; i++) {
        envs[i] = make_env(i, 0);
        envs[i]->go_explore = 1.0f;
        envs[i]->go_explore_door = 0.5f;
        envs[i]->max_ticks = ticks;
        envs[i]->random_start = 120;
    }
    unsigned episodes = 0, doors = 0, speeches = 0;
    float closest = 1e9f;
    SM64Input* nearest = NULL; /* the inputs, from the savestate, that came closest */
    int nearest_count = 0, nearest_frames = 0;
    double started = now();
    #pragma omp parallel num_threads(games)
    {
        SM64* env = envs[omp_get_thread_num()];
        puf_reset(env);
        double next_report = started + 10.0;
        int talking = 0;
        while (now() < started + seconds) {
            explore_policy(env);
            puf_step(env);
            /* The trail is every input since the savestate, so it is a way back to
             * here. Checked again under the lock, since every game is racing for it. */
            float away = sm64_goal_distance(env, sm64_pos(env, 0), sm64_pos(env, 2));
            if (away < closest && env->trail_count > 0) {
                #pragma omp critical(closest)
                if (away < closest) {
                    closest = away;
                    nearest = (SM64Input*)realloc(nearest, (size_t)env->trail_count * sizeof(SM64Input));
                    memcpy(nearest, env->trail, (size_t)env->trail_count * sizeof(SM64Input));
                    nearest_count = env->trail_count;
                    nearest_frames = env->trail_frames;
                }
            }
            int ended = env->agents[0].terminals[0] != 0.0f;
            int now_talking = !ended && (sm64_action(env) & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE;
            if (talking && !now_talking && !ended) {
                __sync_add_and_fetch(&speeches, 1);
            }
            talking = now_talking;
            if (ended) {
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
                printf("%5.0fs  %5d cubes in the archive  %6u episodes  %5u cutscenes ended  %4u %s  closest %5.0f  fastest %s\n",
                       now() - started, sm64_archive_size, episodes, speeches, doors, env->star ? "stars" : "doors",
                       closest, fastest);
                fflush(stdout);
            }
        }
    }
    printf("%u episodes, %u %s; the fastest is in %s\n", episodes, doors, envs[0]->star ? "stars" : "doors",
           sm64_demo_path());
    char path[1024];
    closest_path(path, sizeof(path));
    if (nearest_count > 0 && sm64_demo_write(path, 0, nearest, nearest_count, nearest_frames)) {
        printf("the run that came closest, %.0f units away after %d frames, is in %s\n", closest, nearest_frames,
               path);
    }
    free(nearest);
    for (int i = 0; i < games; i++) {
        puf_close(envs[i]);
    }
    return doors > 0 ? 0 : 1;
}

/* Play the demo back from the savestate, and say whether the door opens when it
 * should: the check that a demo is still good for this savestate. Any other run
 * of inputs plays back the same way, like the one exploring kept for coming
 * closest, and says where it left Mario. */
static int replay(int window, const char* path) {
    int count, frames;
    SM64Input* inputs = sm64_demo_read(path, &count, &frames);
    if (inputs == NULL) {
        printf("no demo in %s: explore first\n", path);
        return 1;
    }
    SM64* env = make_env(0, window);
    const char* goal = env->star ? "the star" : "the door";
    printf("%s: %d inputs, %d frames\n", path, count, frames);
    /* SM64_TRACE=30 prints where Mario is every thirty frames of the way. */
    int trace = atoi(sm64_setting("SM64_TRACE", "0"));
    int frame = 0;
    int opened = -1;
    int spawned = -1;
    int talking = 0;
    int traced = 0;
    double started = now();
    for (int k = 0; k < count && opened < 0; k++) {
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
        }
        if (sm64_goal_reached(env)) {
            opened = frame;
        } else if (((action & ACT_GROUP_MASK) == ACT_GROUP_CUTSCENE) != talking) {
            talking = !talking;
            printf("  frame %4d (%3d before the end): %s at (%.0f %.0f %.0f)\n", frame, frames - frame,
                   talking ? "a cutscene starts" : "the cutscene ends", sm64_pos(env, 0), sm64_pos(env, 1),
                   sm64_pos(env, 2));
        }
    }
    if (opened == frames) {
        printf("%s at frame %d, as it should be\n", goal, opened);
    } else if (opened >= 0) {
        printf("%s at frame %d, not %d%s\n", goal, opened, frames,
               opened < frames ? " (sm64_tool trim cuts the demo there)" : "");
    } else if (spawned == frames) {
        /* A demo from when the spawn was the goal: it stops where the star
         * appears, and never takes it. */
        printf("the star spawned at frame %d, as this demo says, and is not touched: the demo ends at the spawn\n",
               spawned);
    } else {
        printf("never reached %s; Mario ended at (%.0f %.0f %.0f), %.0f units from it across the ground\n", goal,
               sm64_pos(env, 0), sm64_pos(env, 1), sm64_pos(env, 2),
               sm64_goal_distance(env, sm64_pos(env, 0), sm64_pos(env, 2)));
        if (spawned >= 0) {
            printf("  (a star spawned at frame %d)\n", spawned);
        }
    }
    /* Watched, the episode's end is worth seeing too: the door swinging open, or
     * the star coming down and the dance. Five seconds more, with nothing held. */
    if (window && (opened >= 0 || spawned >= 0)) {
        n64gym_pad(&env->gym, 0, 0.0f, 0.0f);
        for (int k = 0; k < 75; k++) {
            n64gym_step(&env->gym, 2);
            puf_render(env);
        }
    }
    puf_close(env);
    free(inputs);
    return opened == frames || spawned == frames ? 0 : 1;
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

/* Record a demo as a video: play it back headless with the game drawing every
 * frame, and hand each one to ffmpeg at thirty a second, with five seconds more
 * after the goal, as watching it gives, so the door swings open or the star
 * comes down. The game is stepped a frame at a time so that every frame is
 * drawn; the goal is checked where replay checks it, after each input. */
static int record(const char* path, const char* out) {
    int count, frames;
    SM64Input* inputs = sm64_demo_read(path, &count, &frames);
    if (inputs == NULL) {
        printf("no demo in %s\n", path);
        return 1;
    }
    /* Only a game opened with a picture draws one. The size here is the
     * observation's, and does not matter: the frame handed over is the
     * console's own. */
    setenv("SM64_PICTURE", "80x60", 0);
    SM64* env = make_env(0, 0);
    const char* goal = env->star ? "the star" : "the door";
    printf("%s: %d inputs, %d frames -> %s\n", path, count, frames, out);
    FILE* video = NULL;
    int width = 0, height = 0, written = 0, frame = 0, opened = -1, spawned = -1, ok = 1;
    for (int k = 0; k < count && opened < 0 && ok; k++) {
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
            opened = frame;
        }
    }
    if (opened == frames) {
        printf("%s at frame %d, as it should be\n", goal, opened);
    } else if (opened >= 0) {
        printf("%s at frame %d, not %d\n", goal, opened, frames);
    } else if (spawned == frames) {
        printf("the star spawned at frame %d, as this demo says: the demo ends at the spawn\n", spawned);
    } else {
        printf("never reached %s; Mario ended at (%.0f %.0f %.0f)\n", goal, sm64_pos(env, 0), sm64_pos(env, 1),
               sm64_pos(env, 2));
    }
    if ((opened >= 0 || spawned >= 0) && ok) {
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
    return (opened == frames || spawned == frames) && ok && status == 0 ? 0 : 1;
}

/* Cut each demo where the goal is reached, and write it back. For demos kept
 * before the goal was measured earlier -- a star counted the frame it was
 * touched, and now the frame its spawn stops time, about a hundred frames
 * sooner. A demo whose goal is where its header says is left alone. */
static int trim(int count_paths, char** paths) {
    SM64* env = make_env(0, 0);
    int failures = 0;
    for (int p = 0; p < count_paths; p++) {
        int count, frames;
        SM64Input* inputs = sm64_demo_read(paths[p], &count, &frames);
        if (inputs == NULL) {
            printf("%s: not a demo\n", paths[p]);
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
            printf("%s: never reaches the goal in %d frames, left as it is\n", paths[p], frames);
            failures++;
        } else if (reached == frames && kept == count) {
            printf("%s: the goal is at frame %d as the header says\n", paths[p], frames);
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
        return explore(argc > 2 ? atoi(argv[2]) : 8, argc > 3 ? atof(argv[3]) : 600.0,
                       argc > 4 ? atoi(argv[4]) : EXPLORE_TICKS);
    }
    if (strcmp(what, "replay") == 0) {
        int window = argc > 2 && strcmp(argv[2], "watch") == 0;
        return replay(window, argc > 2 + window ? argv[2 + window] : sm64_demo_path());
    }
    if (strcmp(what, "record") == 0) {
        const char* demo = argc > 2 ? argv[2] : sm64_demo_path();
        /* The video goes beside the demo, as door.mp4 for door.demo, unless said. */
        char out[1200];
        if (argc > 3) {
            snprintf(out, sizeof(out), "%s", argv[3]);
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
        return trim(argc > 2 ? argc - 2 : 1, argc > 2 ? argv + 2 : &demo);
    }
    printf("usage: sm64 [state | watch [forward|random|door] | probe [forward|random|door] | bench [games]\n"
           "            | explore [games] [seconds] [frames] | replay [watch] [file] | record [file] [out.mp4]\n"
           "            | trim [file...]]\n");
    return 2;
}
