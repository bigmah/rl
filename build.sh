#!/bin/bash
# Build an env for PufferLib 5.0, on a Mac or on Linux.
#
#   ./build.sh [platformer|sm64]
#
#       build/vecenv_<env>.dylib    the env in a vecenv, for the trainer to train on
#                                   (.so on Linux)
#       build/platformer            human play
#       build/sm64_tool             savestates, watching, benchmarking
#
# PufferLib's own build.sh compiles an env from its ocean/ into the CUDA trainer,
# which only an NVIDIA card runs, or into a CPU play binary. This compiles vecenv.c
# around an env from envs/ into a library of its own instead, so envs build side
# by side, and the trainer (src/) loads the one it is asked for.
set -euo pipefail
cd "$(dirname "$0")"

ENV=${1:-platformer}
ENV_DIR=envs/$ENV
PUFFER=vendor/PufferLib
BUILD=build

if [ ! -d "$ENV_DIR" ]; then
    echo "no env called $ENV in envs/" && exit 1
fi
if [ ! -f "$PUFFER/src/pufferenv.h" ]; then
    echo "Fetching PufferLib 5.0..."
    git submodule update --init "$PUFFER"
fi

mkdir -p "$BUILD"

# What a fresh Linux machine needs, on Debian or Ubuntu, for every env and the
# game under sm64. Named once, so every error that means "a package is missing"
# says the same thing.
APT_PACKAGES="build-essential clang cmake ninja-build git curl python3 libsdl2-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev"

# pufferenv.h includes raylib for every env, as it does in PufferLib's own build.
# RAYLIB names one that is already somewhere, with its include/ and lib/.
RAYLIB_FROM_SOURCE=0
case "$(uname -s)" in
    Darwin)
        PLATFORM=macos
        CC=${CC:-clang}
        LIB_EXT=dylib
        RAYLIB_RELEASE=raylib-5.5_macos
        RAYLIB_SYSTEM=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
        SYSTEM_LIBS=()
        # The vecenv steps envs on OpenMP threads, which Apple clang doesn't ship
        OMP="$(brew --prefix libomp)"
        if [ ! -f "$OMP/include/omp.h" ]; then
            echo "OpenMP missing: run 'brew install libomp'" && exit 1
        fi
        OMP_FLAGS=(-Xpreprocessor -fopenmp -I"$OMP/include" -L"$OMP/lib" -lomp -Wl,-rpath,"$OMP/lib")
        CORES=$(sysctl -n hw.ncpu)
        ;;
    Linux)
        PLATFORM=linux
        CC=${CC:-cc}
        LIB_EXT=so
        RAYLIB_SYSTEM=(-lGL -lpthread -ldl -lrt -lX11)
        # shm_open and the threads the games are stepped on, which older glibc
        # keeps out of libc
        SYSTEM_LIBS=(-lpthread -lrt)
        OMP_FLAGS=(-fopenmp)
        CORES=$(getconf _NPROCESSORS_ONLN)
        # raylib releases a Linux build for x86-64 only. Anywhere else, arm64
        # included, it is built from its source into the layout a release unpacks to.
        if [ "$(uname -m)" = x86_64 ]; then
            RAYLIB_RELEASE=raylib-5.5_linux_amd64
        else
            RAYLIB_RELEASE=raylib-5.5_linux_$(uname -m)
            RAYLIB_FROM_SOURCE=1
        fi
        ;;
    *)
        echo "build.sh knows macOS and Linux, not $(uname -s)" && exit 1
        ;;
esac
if [ -n "${RAYLIB:-}" ]; then
    : # one that is already somewhere
elif [ "$RAYLIB_FROM_SOURCE" = 1 ]; then
    RAYLIB=$BUILD/$RAYLIB_RELEASE
    if [ ! -f "$RAYLIB/lib/libraylib.a" ]; then
        echo "Building raylib 5.5 from its source, which releases no build for $(uname -m)..."
        RAYLIB_SRC=$BUILD/raylib-5.5-src
        rm -rf "$RAYLIB_SRC" && mkdir -p "$RAYLIB_SRC"
        curl -fsSL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz | tar xz -C "$RAYLIB_SRC" --strip-components 1
        # Position-independent, because it is linked into the env's shared library.
        # The window it opens is GLFW's, on X11, which is what the headers are for.
        if ! make -C "$RAYLIB_SRC/src" -j "$CORES" PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=STATIC \
                CUSTOM_CFLAGS=-fPIC >"$BUILD/raylib-build.log" 2>&1; then
            tail -20 "$BUILD/raylib-build.log"
            echo "raylib did not build. On Debian or Ubuntu, everything it and the rest of this want is:"
            echo "    sudo apt install $APT_PACKAGES"
            exit 1
        fi
        mkdir -p "$RAYLIB/include" "$RAYLIB/lib"
        cp "$RAYLIB_SRC"/src/{raylib.h,raymath.h,rlgl.h} "$RAYLIB/include/"
        cp "$RAYLIB_SRC/src/libraylib.a" "$RAYLIB/lib/"
    fi
else
    RAYLIB=$BUILD/$RAYLIB_RELEASE
    if [ ! -f "$RAYLIB/lib/libraylib.a" ]; then
        echo "Downloading raylib..."
        curl -fsSL "https://github.com/raysan5/raylib/releases/download/5.5/$RAYLIB_RELEASE.tar.gz" | tar xz -C "$BUILD"
    fi
fi
RAYLIB_LIBS=("$RAYLIB/lib/libraylib.a" "${RAYLIB_SYSTEM[@]}")

CFLAGS=(-O2 -Wall -DPLATFORM_DESKTOP -I"$PUFFER/src" -I"$RAYLIB/include" -I"$ENV_DIR")

if [ "$ENV" = sm64 ]; then
    # The game is a real cartridge, statically recompiled by N64Bundler and run
    # in a process of its own. N64Bundler is a submodule, built here the first
    # time. Three things come from it: the host that runs a game (n64b-run), the
    # protocol header the two sides speak, and the recompiler that turns your
    # cartridge into Super Mario 64 for this machine. N64BUNDLER names another
    # checkout of it instead.
    for tool in cmake ninja clang; do
        if ! command -v "$tool" >/dev/null; then
            echo "Building the game takes cmake, ninja and clang, and there is no $tool."
            if [ "$PLATFORM" = macos ]; then
                echo "    brew install cmake ninja   (clang is Xcode's)"
            else
                echo "    sudo apt install $APT_PACKAGES"
            fi
            exit 1
        fi
    done

    # Training never looks at the game, so the host it needs does not draw: no
    # GPU, graphics API or shader compiler. One that does, through RT64, is what
    # a picture in the observation and watching in a window need. It is the
    # default on a Mac, where it costs nothing to have; elsewhere it wants Vulkan,
    # and SM64_RENDERER=1 asks for it.
    if [ "$PLATFORM" = macos ]; then
        SM64_RENDERER=${SM64_RENDERER:-1}
    else
        SM64_RENDERER=${SM64_RENDERER:-0}
    fi
    N64B_FLAGS=(--no-window)
    [ "$SM64_RENDERER" = 1 ] || N64B_FLAGS+=(--no-renderer)
    if [ -n "${N64BUNDLER:-}" ]; then
        if [ ! -d "$N64BUNDLER/ModernReality" ]; then
            echo "N64BUNDLER is set, and there is no N64Bundler checkout at $N64BUNDLER" && exit 1
        fi
    else
        N64BUNDLER=$(pwd)/vendor/n64bundler
        if [ ! -d "$N64BUNDLER/ModernReality" ]; then
            echo "Fetching N64Bundler..."
            git submodule update --init vendor/n64bundler
        fi
    fi
    MR_BUILD="$N64BUNDLER/ModernReality/build"
    mkdir -p "$BUILD/sm64/game" "$BUILD/sm64/n64b"
    N64B_LOG="$BUILD/sm64/n64bundler-build.log"
    if [ -x "$MR_BUILD/host/n64b-run" ] && [ -x "$MR_BUILD/n64b-port" ]; then
        echo "Building N64Bundler..."
        "$N64BUNDLER/N64Bundler/build.sh" "${N64B_FLAGS[@]}" >"$N64B_LOG" 2>&1 || { cat "$N64B_LOG"; exit 1; }
    else
        echo "Building N64Bundler for the first time, which takes a few minutes..."
        "$N64BUNDLER/N64Bundler/build.sh" "${N64B_FLAGS[@]}" 2>&1 | tee "$N64B_LOG"
    fi

    # The one thing this needs from you: your own dump of the cartridge, Super
    # Mario 64 (USA). SM64_ROM names it; otherwise it is sm64.z64 at the top of
    # this checkout, or the one in N64Bundler's library if you have added it
    # there. It is read where it is: no game data is copied into this repository,
    # and everything derived from it lands in build/, which git ignores.
    if [ -z "${SM64_ROM:-}" ] && [ -f sm64.z64 ]; then
        SM64_ROM=$(pwd)/sm64.z64
    fi
    if [ "$PLATFORM" = macos ]; then
        LIBRARY="$HOME/Library/Application Support/N64Bundler/library.json"
    else
        LIBRARY="${XDG_DATA_HOME:-$HOME/.local/share}/N64Bundler/library.json"
    fi
    if [ -z "${SM64_ROM:-}" ] && [ -f "$LIBRARY" ] && command -v jq >/dev/null; then
        SM64_ROM=$(jq -r 'first(.games[]? | select(.game_id == "NSME")) | .rom // ""' "$LIBRARY")
    fi
    if [ ! -f "${SM64_ROM:-}" ]; then
        echo "No Super Mario 64 ROM. Put your own dump of the cartridge (USA) at"
        echo "    $(pwd)/sm64.z64"
        echo "or set SM64_ROM to where it is."
        exit 1
    fi
    SM64_ROM="$(cd "$(dirname "$SM64_ROM")" && pwd)/$(basename "$SM64_ROM")"

    # The env reads Mario out of the console's memory at the addresses one
    # cartridge keeps him at, and N64Bundler's record of where that cartridge's
    # code is was measured against one dump. Anything else would recompile into
    # a game this env cannot read, so it stops here instead.
    HEADER=$("$MR_BUILD/n64rip" inspect "$SM64_ROM" 2>&1) || {
        echo "$SM64_ROM is not an N64 ROM:"; echo "$HEADER"; exit 1; }
    if [ "$(echo "$HEADER" | sed -n 's/^ROM hash: *//p')" != 8a90daa33e09a265 ]; then
        echo "$SM64_ROM is not Super Mario 64 (USA) as it came off the cartridge:"
        echo "$HEADER" | sed 's/^/    /'
        echo "The env is written against that one dump (game id NSME, ROM hash 8a90daa33e09a265)."
        exit 1
    fi

    # Recompile it: recover where its code is (n64rip, a fraction of a second),
    # then translate that to native code (n64b-port, about twenty seconds). The
    # module is kept until the ROM, the analysis or the tools change, so every
    # build after the first finds it done.
    # Absolute: the analysis writes these paths into the recompiler's config,
    # which reads them relative to itself
    GAME="$(pwd)/$BUILD/sm64/game"
    SM64_MODULE="$GAME/NSME.$LIB_EXT"
    "$MR_BUILD/n64rip" analyze "$SM64_ROM" --out-dir "$GAME/analysis" \
        --runtime-provides "$MR_BUILD/runtime-provides.txt" \
        --titles "$N64BUNDLER/N64Bundler/titles" >"$GAME/n64rip.log" 2>&1 || {
        cat "$GAME/n64rip.log"; exit 1; }
    # A dump in another byte order is normalised by the analysis, which writes
    # the copy the recompiler reads beside it
    RECOMP_ROM="$SM64_ROM"
    [ -f "$GAME/analysis/NSME.z64" ] && RECOMP_ROM="$GAME/analysis/NSME.z64"
    if [ ! -f "$SM64_MODULE" ]; then
        echo "Recompiling Super Mario 64 for this machine..."
    fi
    "$MR_BUILD/n64b-port" build --analysis "$GAME/analysis" --rom "$RECOMP_ROM" \
        --out "$SM64_MODULE" >"$GAME/n64b-port.log" 2>&1 || {
        cat "$GAME/n64b-port.log"; exit 1; }

    cat > "$BUILD/sm64/sm64_paths.h" <<PATHS
/* Written by build.sh: where the game and the pieces around it are on this
 * machine. Every one of these can be overridden at run time by an environment
 * variable of the same name. No game data is here -- these are paths to the
 * player's own copy of it. */
#define SM64_HOST "$MR_BUILD/host/n64b-run"
#define SM64_MODULE "$SM64_MODULE"
#define SM64_ROM "$SM64_ROM"
#define SM64_CONFIG_DIR "$(pwd)/$BUILD/sm64/n64b"
#define SM64_DATA "$(pwd)/$BUILD/sm64"   /* a folder a star under it: its savestate, demos and seeds */
#define SM64_LOG "$(pwd)/$BUILD/sm64/game.log"
#define PUFFERL_ROOT "$(pwd)"            /* where sm64_tool reads the config the trainer does */
PATHS
    CFLAGS+=(-I"$BUILD/sm64" -I"$N64BUNDLER/ModernReality/include/modernreality")
fi

echo "Compiling the $ENV env..."
"$CC" -shared -fPIC -fvisibility=hidden "${CFLAGS[@]}" \
    -DENV_HEADER="\"$ENV_DIR/$ENV.h\"" -DPUFFER_ENV_NAME="\"$ENV\"" \
    vecenv.c "${RAYLIB_LIBS[@]}" "${OMP_FLAGS[@]}" -lm -o "$BUILD/vecenv_$ENV.$LIB_EXT"
echo "Built: $BUILD/vecenv_$ENV.$LIB_EXT"

if [ "$ENV" = platformer ]; then
    "$CC" "${CFLAGS[@]}" "$ENV_DIR/$ENV.c" "${RAYLIB_LIBS[@]}" -lm -o "$BUILD/$ENV"
    echo "Built: $BUILD/$ENV"
fi

if [ "$ENV" = sm64 ]; then
    "$CC" "${CFLAGS[@]}" "$ENV_DIR/$ENV.c" "${OMP_FLAGS[@]}" "${SYSTEM_LIBS[@]+"${SYSTEM_LIBS[@]}"}" -lm -o "$BUILD/sm64_tool"
    echo "Built: $BUILD/sm64_tool"
fi
