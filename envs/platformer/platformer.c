// Human play: A/D move, W or Space jump, S drop through platforms, R restart, ESC quit

#include "platformer.h"

int main(void) {
    Platformer env = {.frameskip = 1, .max_ticks = 120 * 60};
    obs_t observations[OBS_SIZE] = {0};
    float actions[NUM_ATNS] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    env.agents[0] = (Agent){
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
    };

    init(&env);
    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        int left = IsKeyDown(KEY_A);
        int right = IsKeyDown(KEY_D);
        actions[0] = left == right ? 0 : left ? 1 : 2;
        actions[1] = (IsKeyDown(KEY_W) || IsKeyDown(KEY_SPACE)) ? 1 : IsKeyDown(KEY_S) ? 2 : 0;
        if (IsKeyPressed(KEY_R)) {
            puf_reset(&env);
        }
        puf_step(&env);
        puf_render(&env);
    }

    puf_close(&env);
    return 0;
}
