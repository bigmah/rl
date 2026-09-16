#!/bin/bash
# Build an env for PufferLib 5.0 on Apple Silicon.
#
#   ./build.sh [platformer|sm64]
#
#       build/vecenv_<env>.dylib    the env in a vecenv, for mlx_pufferl.py to train on
#       build/platformer            human play
#       build/sm64_tool             savestates, watching, benchmarking
#
# PufferLib's own build.sh compiles an env from its ocean/ into the CUDA trainer,
# which a Mac can't run, or into a CPU play binary. This compiles vecenv.c around
# an env from envs/ into a library of its own instead, so envs build side by side.
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
    echo "PufferLib 5.0 submodule missing: run 'git submodule update --init'" && exit 1
fi

mkdir -p "$BUILD"

# pufferenv.h includes raylib for every env, as it does in PufferLib's own build
RAYLIB=$BUILD/raylib-5.5_macos
if [ ! -d "$RAYLIB" ]; then
    echo "Downloading raylib..."
    curl -sL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_macos.tar.gz | tar xz -C "$BUILD"
fi
RAYLIB_LIBS=("$RAYLIB/lib/libraylib.a" -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)

# The vecenv steps envs on OpenMP threads, which Apple clang doesn't ship
OMP="$(brew --prefix libomp)"
if [ ! -f "$OMP/include/omp.h" ]; then
    echo "OpenMP missing: run 'brew install libomp'" && exit 1
fi
OMP_FLAGS=(-Xpreprocessor -fopenmp -I"$OMP/include" -L"$OMP/lib" -lomp -Wl,-rpath,"$OMP/lib")

CFLAGS=(-O2 -Wall -DPLATFORM_DESKTOP -I"$PUFFER/src" -I"$RAYLIB/include" -I"$ENV_DIR")

if [ "$ENV" = sm64 ]; then
    # The game is a real cartridge, statically recompiled by N64Bundler and run
    # in a process of its own. Three things come from that checkout: the host
    # that runs a game (n64b-run), the protocol header the two sides speak, and
    # the recompiled Super Mario 64 itself.
    N64BUNDLER=${N64BUNDLER:-$(cd .. && pwd)/static_recomp/n64bundler}
    if [ ! -d "$N64BUNDLER/ModernReality" ]; then
        echo "N64Bundler not found at $N64BUNDLER."
        echo "Set N64BUNDLER to your checkout of it."
        exit 1
    fi
    MR_BUILD="$N64BUNDLER/ModernReality/build"
    if [ ! -f "$MR_BUILD/build.ninja" ]; then
        echo "N64Bundler has not been built yet. Run its own build first:"
        echo "    $N64BUNDLER/N64Bundler/build.sh"
        exit 1
    fi

    echo "Building the game host (n64b-run)..."
    cmake --build "$MR_BUILD" -j "$(sysctl -n hw.ncpu)" --target n64b-run >/dev/null

    # Where the recompiled game and the cartridge it came from are. N64Bundler
    # writes both down when a ROM is added to its library; this only reads them.
    # No game data is copied here and none of it goes into git.
    LIBRARY="$HOME/Library/Application Support/N64Bundler/library.json"
    if [ -z "${SM64_MODULE:-}" ] || [ -z "${SM64_ROM:-}" ]; then
        if [ ! -f "$LIBRARY" ]; then
            echo "No N64Bundler library at $LIBRARY."
            echo "Drop your own sm64.z64 on N64Bundler to recompile it first,"
            echo "or set SM64_MODULE and SM64_ROM yourself."
            exit 1
        fi
        FOUND=$(/usr/bin/python3 -c '
import json, sys
library = json.load(open(sys.argv[1]))
for game in library.get("games", []):
    if game.get("game_id") == "NSME":
        print(game.get("module", ""))
        print(game.get("rom", ""))
        break
' "$LIBRARY")
        SM64_MODULE=${SM64_MODULE:-$(echo "$FOUND" | sed -n 1p)}
        SM64_ROM=${SM64_ROM:-$(echo "$FOUND" | sed -n 2p)}
    fi
    if [ ! -f "${SM64_MODULE:-}" ] || [ ! -f "${SM64_ROM:-}" ]; then
        echo "Super Mario 64 is not in the N64Bundler library (game id NSME)."
        echo "Drop your own sm64.z64 on N64Bundler to recompile it, then run this again."
        exit 1
    fi

    mkdir -p "$BUILD/sm64/n64b"
    cat > "$BUILD/sm64/sm64_paths.h" <<PATHS
/* Written by build.sh: where the game and the pieces around it are on this
 * machine. Every one of these can be overridden at run time by an environment
 * variable of the same name. No game data is here -- these are paths to the
 * player's own copy of it. */
#define SM64_HOST "$MR_BUILD/host/n64b-run"
#define SM64_MODULE "$SM64_MODULE"
#define SM64_ROM "$SM64_ROM"
#define SM64_CONFIG_DIR "$(pwd)/$BUILD/sm64/n64b"
#define SM64_STATE "$(pwd)/$BUILD/sm64/castle-grounds.state"
#define SM64_DEMO "$(pwd)/$BUILD/sm64/door.demo"
#define SM64_STAR_STATE "$(pwd)/$BUILD/sm64/bob-omb-battlefield.state"
#define SM64_STAR_DEMO "$(pwd)/$BUILD/sm64/star.demo"
#define SM64_LOG "$(pwd)/$BUILD/sm64/game.log"
PATHS
    CFLAGS+=(-I"$BUILD/sm64" -I"$N64BUNDLER/ModernReality/include/modernreality")
fi

echo "Compiling the $ENV env..."
clang -shared -fPIC -fvisibility=hidden "${CFLAGS[@]}" \
    -DENV_HEADER="\"$ENV_DIR/$ENV.h\"" -DPUFFER_ENV_NAME="\"$ENV\"" \
    vecenv.c "${RAYLIB_LIBS[@]}" "${OMP_FLAGS[@]}" -lm -o "$BUILD/vecenv_$ENV.dylib"
echo "Built: $BUILD/vecenv_$ENV.dylib"

if [ "$ENV" = platformer ]; then
    clang "${CFLAGS[@]}" "$ENV_DIR/$ENV.c" "${RAYLIB_LIBS[@]}" -lm -o "$BUILD/$ENV"
    echo "Built: $BUILD/$ENV"
fi

if [ "$ENV" = sm64 ]; then
    clang "${CFLAGS[@]}" "$ENV_DIR/$ENV.c" "${OMP_FLAGS[@]}" -lm -o "$BUILD/sm64_tool"
    echo "Built: $BUILD/sm64_tool"
fi
