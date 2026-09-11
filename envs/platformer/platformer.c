// Human play: A/D move, W or Space jump, S drop through platforms, R restart, ESC quit

#include "platformer.h"

int main(void) {
    Platformer env = {.frameskip = 1, .max_ticks = 120 * 60};
    env.observations = (float*)calloc(NUM_OBS, sizeof(float));
    env.actions = (float*)calloc(2, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    init(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        int left = IsKeyDown(KEY_A);
        int right = IsKeyDown(KEY_D);
        env.actions[0] = left == right ? 0 : left ? 1 : 2;
        env.actions[1] = (IsKeyDown(KEY_W) || IsKeyDown(KEY_SPACE)) ? 1 : IsKeyDown(KEY_S) ? 2 : 0;
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }
        c_step(&env);
        c_render(&env);
    }

    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
