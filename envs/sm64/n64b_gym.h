// SPDX-License-Identifier: GPL-3.0-or-later
//
// One recompiled N64 game, in a process of its own, driven from here.
//
// N64Bundler statically recompiles a cartridge into native arm64 and runs it in
// `n64b-run`. Its `--gym` mode hands the console to another process: the game
// advances one frame when this asks for one, as fast as the machine manages,
// and stops between frames with every thread parked and every device answered.
// This is that other process's half.
//
// A game to a process, and not a choice. librecomp and ultramodern are a
// console, and a console is a pile of globals -- one memory, one scheduler, one
// video interface, one set of threads. Two games in one process would be two
// games on one console. So an environment here is a child process, and what
// crosses between them is the console's eight megabytes (mapped into both, so
// reading the game's variables is a load rather than a request) and a small
// block of control.
//
// Everything in here is about *a* game. What any particular game keeps where --
// where Mario is, how fast he is going -- is sm64.h's business.

#ifndef N64B_GYM_H
#define N64B_GYM_H

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gym.h" // from N64Bundler: the protocol both sides speak

extern char** environ;

typedef struct {
    const char* host;       // path to n64b-run
    const char* module;     // the recompiled game
    const char* rom;        // the cartridge it was recompiled from
    const char* game_id;    // four characters out of the ROM header, "NSME"
    const char* config_dir; // where librecomp keeps its copy of the ROM and the saves
    const char* log;        // where the game's own output goes, or NULL to inherit ours
    int windowed;           // draw it, for watching a policy play
    int picture;            // hand over every frame as a picture: see n64gym_picture
    int index;              // which of several games this is
} N64GymOptions;

typedef struct {
    pid_t pid;
    int socket;
    int shm;
    char shm_name[32];
    struct n64b_gym_block* block;
    uint8_t* memory; // the console's, mapped
    char error[256];
} N64Gym;

// --- the console's memory ----------------------------------------------------
//
// Words are stored so that an aligned 32-bit load is a load. Halfwords and
// bytes are not: the runtime keeps the cartridge's big-endian memory in
// native-endian words, so the two low bits of an address are flipped to reach
// the smaller pieces inside one. That is what `^ 2` and `^ 3` are, and they are
// the same two lines as MEM_H and MEM_B in the recompiler's own header.

static inline uint32_t n64_u32(const N64Gym* gym, uint32_t address) {
    uint32_t value;
    memcpy(&value, gym->memory + (address - N64B_GYM_RDRAM_BASE), sizeof(value));
    return value;
}

static inline int32_t n64_s32(const N64Gym* gym, uint32_t address) {
    return (int32_t)n64_u32(gym, address);
}

static inline float n64_f32(const N64Gym* gym, uint32_t address) {
    uint32_t bits = n64_u32(gym, address);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline uint16_t n64_u16(const N64Gym* gym, uint32_t address) {
    uint16_t value;
    memcpy(&value, gym->memory + ((address ^ 2) - N64B_GYM_RDRAM_BASE), sizeof(value));
    return value;
}

static inline int16_t n64_s16(const N64Gym* gym, uint32_t address) {
    return (int16_t)n64_u16(gym, address);
}

static inline uint8_t n64_u8(const N64Gym* gym, uint32_t address) {
    return gym->memory[(address ^ 3) - N64B_GYM_RDRAM_BASE];
}

// Writing is the same arithmetic the other way. It is for setting a game up --
// moving a warp, placing Mario to see what is there -- never for playing it.

static inline void n64_set_u32(N64Gym* gym, uint32_t address, uint32_t value) {
    memcpy(gym->memory + (address - N64B_GYM_RDRAM_BASE), &value, sizeof(value));
}

static inline void n64_set_s16(N64Gym* gym, uint32_t address, int16_t value) {
    memcpy(gym->memory + ((address ^ 2) - N64B_GYM_RDRAM_BASE), &value, sizeof(value));
}

static inline void n64_set_f32(N64Gym* gym, uint32_t address, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    n64_set_u32(gym, address, bits);
}

/// Whether an address is one the console has memory at. A pointer read out of
/// the game is worth checking before it is followed: a null one is ordinary
/// (Mario has no wall to his left), and a wild one means something read the
/// wrong field.
static inline int n64_is_ram(uint32_t address) {
    return address >= N64B_GYM_RDRAM_BASE && address < N64B_GYM_RDRAM_BASE + N64B_GYM_RDRAM_BYTES;
}

// --- starting one ------------------------------------------------------------

static void n64gym_fail(N64Gym* gym, const char* what) {
    snprintf(gym->error, sizeof(gym->error), "%s: %s", what, strerror(errno));
}

static int n64gym_write_all(int fd, const void* data, size_t bytes) {
    const uint8_t* from = (const uint8_t*)data;
    while (bytes > 0) {
        ssize_t wrote = write(fd, from, bytes);
        if (wrote <= 0) {
            if (wrote < 0 && errno == EINTR) continue;
            return 0;
        }
        from += wrote;
        bytes -= (size_t)wrote;
    }
    return 1;
}

static int n64gym_read_all(int fd, void* data, size_t bytes) {
    uint8_t* into = (uint8_t*)data;
    while (bytes > 0) {
        ssize_t got = read(fd, into, bytes);
        if (got <= 0) {
            if (got < 0 && errno == EINTR) continue;
            return 0;
        }
        into += got;
        bytes -= (size_t)got;
    }
    return 1;
}

/// Give this game its own directory for what librecomp writes.
///
/// The runtime keeps its own copy of the cartridge and a save file beside it,
/// and a machine running a dozen of these at once must not have a dozen games
/// writing one save file. The copy of the cartridge is the expensive part and it
/// is the same eight megabytes for every one of them, so the first game keeps it
/// -- the runtime writes it, byte order and all -- and the rest get a directory
/// of their own with a link to it and a save file that is theirs.
///
/// Games are started one at a time, so by the time any of the others look, the
/// first game's copy is there.
static void n64gym_prepare_config(const N64GymOptions* options, char* out, size_t out_size) {
    mkdir(options->config_dir, 0755);
    if (options->index == 0) {
        snprintf(out, out_size, "%s", options->config_dir);
        return;
    }
    snprintf(out, out_size, "%s/game%d", options->config_dir, options->index);
    mkdir(out, 0755);

    char shared_rom[1024];
    char linked_rom[1024];
    snprintf(shared_rom, sizeof(shared_rom), "%s/%s.z64", options->config_dir, options->game_id);
    snprintf(linked_rom, sizeof(linked_rom), "%s/%s.z64", out, options->game_id);
    struct stat ignored;
    if (stat(shared_rom, &ignored) == 0 && stat(linked_rom, &ignored) != 0) {
        symlink(shared_rom, linked_rom);
    }
}

/// Start a game and wait for it to boot.
///
/// Returns 0 with `gym->error` set if anything went wrong, including the game
/// failing to reach the point where it is waiting for a retrace -- which is what
/// a cartridge that does not run at all looks like from here.
static int n64gym_open(N64Gym* gym, const N64GymOptions* options) {
    memset(gym, 0, sizeof(*gym));
    gym->socket = -1;
    gym->shm = -1;

    // Named after this process and this game, because the name is a filesystem
    // path in all but spelling and macOS gives it 31 characters.
    snprintf(gym->shm_name, sizeof(gym->shm_name), "/n64b%d.%d", (int)getpid(), options->index);
    shm_unlink(gym->shm_name);
    gym->shm = shm_open(gym->shm_name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (gym->shm < 0) {
        n64gym_fail(gym, "could not make the shared block");
        return 0;
    }
    if (ftruncate(gym->shm, N64B_GYM_SHM_BYTES) != 0) {
        n64gym_fail(gym, "could not size the shared block");
        return 0;
    }
    void* mapped = mmap(NULL, N64B_GYM_SHM_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, gym->shm, 0);
    if (mapped == MAP_FAILED) {
        n64gym_fail(gym, "could not map the shared block");
        return 0;
    }
    gym->block = (struct n64b_gym_block*)mapped;
    gym->memory = (uint8_t*)mapped + N64B_GYM_RDRAM_OFFSET;
    memset(gym->block, 0, sizeof(*gym->block));
    gym->block->magic = N64B_GYM_MAGIC;
    gym->block->abi = N64B_GYM_ABI;

    int pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
        n64gym_fail(gym, "could not make the control socket");
        return 0;
    }

    char config[1024];
    n64gym_prepare_config(options, config, sizeof(config));

    char* argv[16];
    int argc = 0;
    argv[argc++] = (char*)options->host;
    argv[argc++] = (char*)"--module";
    argv[argc++] = (char*)options->module;
    argv[argc++] = (char*)"--rom";
    argv[argc++] = (char*)options->rom;
    argv[argc++] = (char*)"--gym";
    argv[argc++] = gym->shm_name;
    argv[argc++] = (char*)"--config-dir";
    argv[argc++] = config;
    if (!options->windowed) argv[argc++] = (char*)"--headless";
    if (options->picture) argv[argc++] = (char*)"--picture";
    argv[argc] = NULL;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    // The game expects its end of the socket on descriptor 3 and nothing else.
    posix_spawn_file_actions_adddup2(&actions, pair[1], N64B_GYM_SOCKET_FD);
    posix_spawn_file_actions_addclose(&actions, pair[0]);
    const char* log = options->log;
    if (log != NULL) {
        posix_spawn_file_actions_addopen(&actions, 1, log, O_WRONLY | O_CREAT | O_APPEND, 0644);
        posix_spawn_file_actions_adddup2(&actions, 1, 2);
    }

    int spawned = posix_spawn(&gym->pid, options->host, &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pair[1]);
    if (spawned != 0) {
        errno = spawned;
        n64gym_fail(gym, "could not start the game");
        close(pair[0]);
        return 0;
    }
    gym->socket = pair[0];

    // It rings once when it has booted and is waiting for a retrace.
    uint8_t ready = 0;
    if (!n64gym_read_all(gym->socket, &ready, 1)) {
        snprintf(gym->error, sizeof(gym->error),
                 "the game stopped before it finished booting (see %s)",
                 log != NULL ? log : "its output above");
        return 0;
    }
    return 1;
}

/// Ask for something and wait for it to be done.
static int n64gym_ask(N64Gym* gym, uint32_t command, uint32_t count) {
    gym->block->command = command;
    gym->block->count = count;
    uint8_t doorbell = 1;
    if (!n64gym_write_all(gym->socket, &doorbell, 1) ||
        !n64gym_read_all(gym->socket, &doorbell, 1)) {
        snprintf(gym->error, sizeof(gym->error), "the game is gone");
        return 0;
    }
    if (gym->block->status != N64B_GYM_OK) {
        snprintf(gym->error, sizeof(gym->error), "%s", gym->block->message);
        return 0;
    }
    return 1;
}

/// What the game will read the next time it looks at the controller. Buttons are
/// libultra's own bits; the stick is a fraction of full deflection.
static inline void n64gym_pad(N64Gym* gym, uint16_t buttons, float stick_x, float stick_y) {
    gym->block->pad[0].buttons = buttons;
    gym->block->pad[0].stick_x = stick_x;
    gym->block->pad[0].stick_y = stick_y;
}

static inline int n64gym_step(N64Gym* gym, int frames) {
    return n64gym_ask(gym, N64B_GYM_STEP, (uint32_t)frames);
}

static inline int n64gym_save_state(N64Gym* gym, const char* path) {
    snprintf(gym->block->path, sizeof(gym->block->path), "%s", path);
    return n64gym_ask(gym, N64B_GYM_SAVE_STATE, 0);
}

static inline int n64gym_load_state(N64Gym* gym, const char* path) {
    snprintf(gym->block->path, sizeof(gym->block->path), "%s", path);
    return n64gym_ask(gym, N64B_GYM_LOAD_STATE, 0);
}

static inline uint64_t n64gym_frames(const N64Gym* gym) { return gym->block->frames; }

/// Which frames of each step get drawn, from the next step on: every one, only
/// the last (the one the picture is of), or none, for getting somewhere with no
/// picture wanted. Only a headless game with a picture draws at all, and see
/// `n64b_gym_draw` for when skipping is wrong.
static inline void n64gym_draw(N64Gym* gym, enum n64b_gym_draw draw) {
    gym->block->draw = (uint32_t)draw;
}

/// What the game looks like: the frame its video interface will show next, as
/// rows of 8-bit RGBA, top to bottom. Only a game opened with `picture` has one,
/// and it is fresh after every step or load. Returns NULL, with the size zero,
/// while there is no picture -- a blanked screen, or a game not asked for one.
///
/// This is the one thing here that is the same for every game: it needs no
/// address out of any game's memory, only the console's own video interface.
static inline const uint8_t* n64gym_picture(const N64Gym* gym, int* width, int* height) {
    *width = (int)gym->block->picture_width;
    *height = (int)gym->block->picture_height;
    if (*width == 0 || *height == 0) {
        return NULL;
    }
    return (const uint8_t*)gym->block + N64B_GYM_PICTURE_OFFSET;
}

static void n64gym_close(N64Gym* gym) {
    if (gym->socket >= 0) {
        n64gym_ask(gym, N64B_GYM_QUIT, 0);
        close(gym->socket);
        gym->socket = -1;
    }
    if (gym->pid > 0) {
        // It ends itself when the socket closes; this is for the case where it
        // has stopped listening.
        int status = 0;
        for (int waited = 0; waited < 200; waited++) {
            pid_t done = waitpid(gym->pid, &status, WNOHANG);
            if (done == gym->pid || done < 0) break;
            usleep(10000);
            if (waited == 100) kill(gym->pid, SIGKILL);
        }
        gym->pid = 0;
    }
    if (gym->block != NULL) {
        munmap(gym->block, N64B_GYM_SHM_BYTES);
        gym->block = NULL;
        gym->memory = NULL;
    }
    if (gym->shm >= 0) {
        close(gym->shm);
        shm_unlink(gym->shm_name);
        gym->shm = -1;
    }
}

#endif
