/* Platformer: a single-level 2D platformer. Run right, clear the pits and
 * spikes, and touch the flag. A PufferLib 5.0 env: the same puf_step drives
 * human play (platformer.c) and training (vecenv.c).
 *
 * Actions: two discrete heads
 *   move: 0 none, 1 left, 2 right
 *   vert: 0 none, 1 jump, 2 drop through one-way platforms
 *
 * Observations: a tile window around the player with 3 channels
 * (solid, one-way, hazard), followed by player state.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

typedef float obs_t;
#include "pufferenv.h"

#define LEVEL_W 100
#define LEVEL_H 18

// # solid   = one-way platform   ^ spikes   G goal flag   P player start
static const char LEVEL[LEVEL_H][LEVEL_W + 1] = {
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "....................................................................................................",
    "..........................................................##########........=======............G....",
    "...................................##....====............###########..##.......................G....",
    "..P...............^...######.......##.............^^....############..##......^^^..............G....",
    "############..##############...#########......######################..##..############...###########",
    "############..##############...#########......######################..##..############...###########",
    "############..##############...#########......######################..##..############...###########",
    "############..##############...#########......######################..##..############...###########",
};

// Physics, in tiles and seconds
#define DT (1.0f / 60.0f)
#define GRAVITY 50.0f
#define JUMP_SPEED 16.0f  // apex ~2.5 tiles
#define MAX_FALL 20.0f
#define RUN_SPEED 9.0f
#define GROUND_ACCEL 80.0f
#define AIR_ACCEL 40.0f
#define COYOTE_TICKS 6    // can still jump briefly after walking off a ledge
#define PLAYER_W 0.8f
#define PLAYER_H 0.9f
#define EPS 1e-3f

#define VIEW_W 15  // tile columns visible to the agent, centered on the player
#define VIEW_H 9
#define NUM_OBS (3 * VIEW_W * VIEW_H + 8)

#define OBS_SIZE NUM_OBS
#define NUM_ATNS 2
#define ACT_SIZES {3, 3}

#define TILE_PX 40
#define SCREEN_W 1280
#define SCREEN_H (LEVEL_H * TILE_PX)

enum { TICK_OK, TICK_DEATH, TICK_GOAL, TICK_TIMEOUT };

// Required struct. Only use floats!
struct Log {
    float perf;            // fraction of episodes that reach the flag
    float score;           // furthest progress toward the flag, 0-1
    float episode_return;
    float episode_length;  // agent steps
    float deaths;          // fraction of episodes ending on spikes or in a pit
    float timeouts;        // fraction of episodes that ran out of time
    float n;               // Required as the last field
};

typedef struct {
    float x, y;    // top-left of the hitbox, in tiles
    float vx, vy;  // tiles per second
    int on_ground;
    int coyote;
} Player;

typedef struct {
    int facing;  // 1 right, -1 left
} Client;

// Required struct
struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    Client* client;
    int num_agents;
    unsigned int rng;  // set by vecenv; the level is deterministic
    int frameskip;     // physics ticks per agent step
    int max_ticks;
    float fail_penalty;  // for dying or running out of time
    Player player;
    float spawn_x, spawn_y;
    float goal_x;
    float best_x;      // furthest player center x this episode
    int tick;          // physics ticks this episode
    int steps;         // agent steps this episode
    float episode_return;
    int wins, total_deaths;  // lifetime counters for the HUD
};
typedef Env Platformer;

static const Color PUFF_RED = {187, 0, 0, 255};
static const Color PUFF_CYAN = {0, 187, 187, 255};
static const Color PUFF_WHITE = {241, 241, 241, 241};
static const Color PUFF_BACKGROUND = {6, 24, 24, 255};
static const Color SOLID_COLOR = {24, 64, 64, 255};
static const Color PLATFORM_COLOR = {90, 140, 140, 255};
static const Color GOAL_COLOR = {255, 200, 40, 255};

static inline char tile_at(int tx, int ty) {
    if (tx < 0 || tx >= LEVEL_W) return '#';  // walls left and right of the level
    if (ty < 0 || ty >= LEVEL_H) return '.';  // open sky above, bottomless pit below
    return LEVEL[ty][tx];
}

void init(Platformer* env) {
    for (int ty = 0; ty < LEVEL_H; ty++) {
        for (int tx = 0; tx < LEVEL_W; tx++) {
            if (LEVEL[ty][tx] == 'P') {
                env->spawn_x = tx + (1.0f - PLAYER_W) / 2.0f;
                env->spawn_y = ty + 1.0f - PLAYER_H;
            } else if (LEVEL[ty][tx] == 'G') {
                env->goal_x = tx + 0.5f;
            }
        }
    }
}

void compute_observations(Platformer* env) {
    Player* p = &env->player;
    obs_t* obs = env->agents[0].observations;
    float cx = p->x + PLAYER_W / 2.0f;
    float cy = p->y + PLAYER_H / 2.0f;
    int px = (int)floorf(cx);
    int py = (int)floorf(cy);

    int i = 0;
    for (int dy = -VIEW_H / 2; dy <= VIEW_H / 2; dy++) {
        for (int dx = -VIEW_W / 2; dx <= VIEW_W / 2; dx++) {
            char c = tile_at(px + dx, py + dy);
            obs[i++] = c == '#';
            obs[i++] = c == '=';
            obs[i++] = c == '^' || py + dy >= LEVEL_H;
        }
    }
    obs[i++] = cx / LEVEL_W;
    obs[i++] = cy / LEVEL_H;
    obs[i++] = cx - px;
    obs[i++] = cy - py;
    obs[i++] = p->vx / RUN_SPEED;
    obs[i++] = p->vy / MAX_FALL;
    obs[i++] = p->on_ground;
    obs[i++] = (float)env->tick / env->max_ticks;
}

// Required function
void puf_reset(Platformer* env) {
    env->player = (Player){.x = env->spawn_x, .y = env->spawn_y};
    env->best_x = env->spawn_x + PLAYER_W / 2.0f;
    env->tick = 0;
    env->steps = 0;
    env->episode_return = 0.0f;
    compute_observations(env);
}

// Is the player overlapping a tile of this kind? Spikes only hurt in their lower-middle.
static int touching(Player* p, char kind) {
    int x0 = (int)floorf(p->x), x1 = (int)floorf(p->x + PLAYER_W);
    int y0 = (int)floorf(p->y), y1 = (int)floorf(p->y + PLAYER_H);
    for (int ty = y0; ty <= y1; ty++) {
        for (int tx = x0; tx <= x1; tx++) {
            if (tile_at(tx, ty) != kind) continue;
            if (kind != '^') return 1;
            if (p->x + PLAYER_W > tx + 0.15f && p->x < tx + 0.85f && p->y + PLAYER_H > ty + 0.45f) {
                return 1;
            }
        }
    }
    return 0;
}

static int physics_tick(Platformer* env, int move, int vert) {
    Player* p = &env->player;

    float target = move == 1 ? -RUN_SPEED : move == 2 ? RUN_SPEED : 0.0f;
    float accel = (p->on_ground ? GROUND_ACCEL : AIR_ACCEL) * DT;
    if (p->vx < target) p->vx = fminf(p->vx + accel, target);
    else if (p->vx > target) p->vx = fmaxf(p->vx - accel, target);

    p->coyote = p->on_ground ? COYOTE_TICKS : p->coyote - 1;
    if (vert == 1 && p->coyote > 0) {
        p->vy = -JUMP_SPEED;
        p->coyote = 0;
    }
    p->vy = fminf(p->vy + GRAVITY * DT, MAX_FALL);

    // Move and resolve collisions one axis at a time. EPS keeps tiles the
    // player is merely flush against out of the other axis' checks.
    p->x += p->vx * DT;
    int y0 = (int)floorf(p->y), y1 = (int)floorf(p->y + PLAYER_H - EPS);
    if (p->vx > 0) {
        int tx = (int)floorf(p->x + PLAYER_W);
        for (int ty = y0; ty <= y1; ty++) {
            if (tile_at(tx, ty) == '#') { p->x = tx - PLAYER_W; p->vx = 0; break; }
        }
    } else if (p->vx < 0) {
        int tx = (int)floorf(p->x);
        for (int ty = y0; ty <= y1; ty++) {
            if (tile_at(tx, ty) == '#') { p->x = tx + 1.0f; p->vx = 0; break; }
        }
    }

    float old_bottom = p->y + PLAYER_H;
    p->y += p->vy * DT;
    p->on_ground = 0;
    int x0 = (int)floorf(p->x + EPS), x1 = (int)floorf(p->x + PLAYER_W - EPS);
    if (p->vy > 0) {
        int ty = (int)floorf(p->y + PLAYER_H);
        for (int tx = x0; tx <= x1; tx++) {
            char c = tile_at(tx, ty);
            // One-way platforms only catch a player who was above them and isn't dropping
            int platform = c == '=' && old_bottom <= ty + EPS && vert != 2;
            if (c == '#' || platform) {
                p->y = ty - PLAYER_H;
                p->vy = 0;
                p->on_ground = 1;
                break;
            }
        }
    } else if (p->vy < 0) {
        int ty = (int)floorf(p->y);
        for (int tx = x0; tx <= x1; tx++) {
            if (tile_at(tx, ty) == '#') { p->y = ty + 1.0f; p->vy = 0; break; }
        }
    }

    if (p->y > LEVEL_H + 1.0f || touching(p, '^')) return TICK_DEATH;
    if (touching(p, 'G')) return TICK_GOAL;
    return TICK_OK;
}

static void end_episode(Platformer* env, int outcome) {
    Agent* agent = &env->agents[0];
    float start = env->spawn_x + PLAYER_W / 2.0f;
    env->episode_return += agent->rewards[0];
    env->log.perf += outcome == TICK_GOAL;
    env->log.score += outcome == TICK_GOAL ? 1.0f : (env->best_x - start) / (env->goal_x - start);
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->steps;
    env->log.deaths += outcome == TICK_DEATH;
    env->log.timeouts += outcome == TICK_TIMEOUT;
    env->log.n += 1;
    env->wins += outcome == TICK_GOAL;
    env->total_deaths += outcome == TICK_DEATH;
    agent->terminals[0] = 1.0f;
    puf_reset(env);
}

// Required function
void puf_step(Platformer* env) {
    Agent* agent = &env->agents[0];
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;
    env->steps++;

    float a_move = agent->actions[0];
    float a_vert = agent->actions[1];
    int move = (a_move >= 0.0f && a_move <= 2.0f) ? (int)a_move : 0;
    int vert = (a_vert >= 0.0f && a_vert <= 2.0f) ? (int)a_vert : 0;

    // Reward new ground covered toward the flag; the full run is worth 1
    float progress_scale = 1.0f / (env->goal_x - (env->spawn_x + PLAYER_W / 2.0f));
    for (int i = 0; i < env->frameskip; i++) {
        int outcome = physics_tick(env, move, vert);
        env->tick++;

        float cx = env->player.x + PLAYER_W / 2.0f;
        if (cx > env->best_x) {
            agent->rewards[0] += (cx - env->best_x) * progress_scale;
            env->best_x = cx;
        }

        if (outcome == TICK_DEATH) {
            agent->rewards[0] -= env->fail_penalty;
            end_episode(env, outcome);
            return;
        }
        if (outcome == TICK_GOAL) {
            agent->rewards[0] += 1.0f;
            end_episode(env, outcome);
            return;
        }
        if (env->tick >= env->max_ticks) {
            // Same cost as dying: running out the clock never scores better than a failed jump
            agent->rewards[0] -= env->fail_penalty;
            end_episode(env, TICK_TIMEOUT);
            return;
        }
    }
    env->episode_return += agent->rewards[0];
    compute_observations(env);
}

// Required function. Creates the window on first call
void puf_render(Platformer* env) {
    if (env->client == NULL) {
        InitWindow(SCREEN_W, SCREEN_H, "PufferLib Platformer");
        SetTargetFPS(60 / env->frameskip);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->facing = 1;
    }

    // Standard across puffer envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    Player* p = &env->player;
    Client* client = env->client;
    if (p->vx > 0.1f) client->facing = 1;
    if (p->vx < -0.1f) client->facing = -1;

    float cam = (p->x + PLAYER_W / 2.0f) * TILE_PX - SCREEN_W / 2.0f;
    cam = fminf(fmaxf(cam, 0.0f), (float)(LEVEL_W * TILE_PX - SCREEN_W));

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    int first_col = (int)(cam / TILE_PX);
    int last_col = first_col + SCREEN_W / TILE_PX + 1;
    if (last_col > LEVEL_W) last_col = LEVEL_W;
    for (int ty = 0; ty < LEVEL_H; ty++) {
        for (int tx = first_col; tx < last_col; tx++) {
            char c = LEVEL[ty][tx];
            float sx = tx * TILE_PX - cam;
            float sy = ty * TILE_PX;
            if (c == '#') {
                DrawRectangle(sx, sy, TILE_PX, TILE_PX, SOLID_COLOR);
                if (ty == 0 || LEVEL[ty - 1][tx] != '#') {
                    DrawRectangle(sx, sy, TILE_PX, 6, PUFF_CYAN);
                }
            } else if (c == '=') {
                DrawRectangle(sx, sy, TILE_PX, 8, PLATFORM_COLOR);
            } else if (c == '^') {
                for (int k = 0; k < 2; k++) {
                    float bx = sx + k * TILE_PX / 2.0f;
                    DrawTriangle(
                        (Vector2){bx + TILE_PX / 4.0f, sy + TILE_PX * 0.45f},
                        (Vector2){bx + 2, sy + TILE_PX},
                        (Vector2){bx + TILE_PX / 2.0f - 2, sy + TILE_PX},
                        PUFF_RED);
                }
            } else if (c == 'G') {
                DrawRectangle(sx + TILE_PX / 2 - 3, sy, 6, TILE_PX, PUFF_WHITE);
                if (ty == 0 || LEVEL[ty - 1][tx] != 'G') {
                    float fx = sx + TILE_PX / 2.0f + 3;
                    DrawTriangle((Vector2){fx, sy}, (Vector2){fx, sy + 24},
                        (Vector2){fx + 27, sy + 12}, GOAL_COLOR);
                }
            }
        }
    }

    int px = (int)(p->x * TILE_PX - cam);
    int py = (int)(p->y * TILE_PX);
    int pw = (int)(PLAYER_W * TILE_PX);
    int ph = (int)(PLAYER_H * TILE_PX);
    DrawRectangle(px, py, pw, ph, PUFF_CYAN);
    int eyes = px + pw / 2 + client->facing * pw / 5;
    DrawRectangle(eyes - 7, py + 8, 5, 9, PUFF_BACKGROUND);
    DrawRectangle(eyes + 2, py + 8, 5, 9, PUFF_BACKGROUND);

    float start = env->spawn_x + PLAYER_W / 2.0f;
    float progress = (env->best_x - start) / (env->goal_x - start);
    DrawText("A/D move   W/SPACE jump   S drop   R restart   ESC quit", 16, 12, 20, PUFF_WHITE);
    DrawText(TextFormat("Progress %3d%%   Time %4.1fs   Flags %d   Deaths %d",
        (int)(100 * progress), env->tick * DT, env->wins, env->total_deaths),
        16, 40, 20, PUFF_WHITE);

    EndDrawing();
    puf_web_vsync();
}

// Required function. Do not free the agent buffers: the vecenv owns them
void puf_close(Platformer* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

// Required function: read this env's settings from [env] in its config
void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip");
    env->max_ticks = dict_get(kwargs, "max_ticks");
    env->fail_penalty = dict_get(kwargs, "fail_penalty");
    init(env);
}

// Required function: the Log averaged over episodes, by name
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "timeouts", log->timeouts);
}
