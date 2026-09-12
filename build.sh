#!/bin/bash
# Build an env for PufferLib on Apple Silicon.
#
#   ./build.sh [platformer|sm64]
#
#       vendor/PufferLib/pufferlib/_C*.so   the training backend, for this env
#       build/platformer                    human play
#       build/sm64_tool                     savestates, watching, benchmarking
#
# PufferLib's own build.sh targets Linux/CUDA (-mavx2, gcc -fopenmp, nvcc) and
# only finds envs under ocean/, so this mirrors its --cpu mode for macOS. One env
# is compiled into the extension at a time, which is PufferLib's own shape:
# building a different one replaces it.
set -euo pipefail
cd "$(dirname "$0")"

ENV=${1:-platformer}
ENV_DIR=envs/$ENV
PUFFER=vendor/PufferLib
BUILD=build
PYTHON=.venv/bin/python

if [ ! -d "$ENV_DIR" ]; then
    echo "no env called $ENV in envs/" && exit 1
fi
if [ ! -f "$PUFFER/src/vecenv.h" ]; then
    echo "PufferLib submodule missing: run 'git submodule update --init'" && exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "Python env missing: run 'uv sync'" && exit 1
fi

mkdir -p "$BUILD"

# vecenv.h needs OpenMP, which Apple clang doesn't ship. Compile against Homebrew's
# libomp headers but link torch's bundled runtime: two OpenMP runtimes in one process abort.
OMP_INCLUDE="$(brew --prefix libomp)/include"
if [ ! -f "$OMP_INCLUDE/omp.h" ]; then
    echo "OpenMP headers missing: run 'brew install libomp'" && exit 1
fi
TORCH_LIB=$($PYTHON -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
PY_INCLUDE=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$($PYTHON -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# What this env needs beyond the binding itself.
ENV_INCLUDES=(-I"$ENV_DIR")
ENV_LIBS=()
ENV_FRAMEWORKS=()

if [ "$ENV" = platformer ]; then
    RAYLIB=$BUILD/raylib-5.5_macos
    if [ ! -d "$RAYLIB" ]; then
        echo "Downloading raylib..."
        curl -sL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_macos.tar.gz | tar xz -C "$BUILD"
    fi
    ENV_INCLUDES+=(-DPLATFORM_DESKTOP -I"$RAYLIB/include")
    ENV_LIBS+=("$RAYLIB/lib/libraylib.a")
    ENV_FRAMEWORKS+=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi

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
#define SM64_LOG "$(pwd)/$BUILD/sm64/game.log"
PATHS
    ENV_INCLUDES+=(-I"$BUILD/sm64" -I"$N64BUNDLER/ModernReality/include/modernreality")
fi

echo "Compiling the $ENV env..."
clang -c -O2 -DNDEBUG -Wall -fPIC -fvisibility=hidden \
    -Xpreprocessor -fopenmp -I"$OMP_INCLUDE" \
    -I"$PUFFER/src" "${ENV_INCLUDES[@]}" \
    "$ENV_DIR/binding.c" -o "$BUILD/binding.o"

# Only depends on PufferLib's sources and the env's name, so skip it unless one changed
BINDINGS_OBJ=$BUILD/bindings_cpu_$ENV.o
if [ ! "$BINDINGS_OBJ" -nt "$PUFFER/src/bindings_cpu.cpp" ] || [ ! "$BINDINGS_OBJ" -nt "$PUFFER/src/vecenv.h" ]; then
    echo "Compiling CPU training backend..."
    clang++ -c -O2 -std=c++17 -fPIC -DPRECISION_FLOAT -DENV_NAME=$ENV \
        -I"$PUFFER/src" -I"$PY_INCLUDE" -I"$PYBIND_INCLUDE" \
        "$PUFFER/src/bindings_cpu.cpp" -o "$BINDINGS_OBJ"
fi

OUTPUT="$PUFFER/pufferlib/_C$EXT_SUFFIX"
clang++ -shared "$BINDINGS_OBJ" "$BUILD/binding.o" \
    ${ENV_LIBS[@]+"${ENV_LIBS[@]}"} ${ENV_FRAMEWORKS[@]+"${ENV_FRAMEWORKS[@]}"} \
    -L"$TORCH_LIB" -lomp -Wl,-rpath,"$TORCH_LIB" \
    -undefined dynamic_lookup -o "$OUTPUT"
# torch's libomp.dylib carries a stale install name (/opt/llvm-openmp/...), so
# point the extension at the copy torch loads, then re-sign after editing
install_name_tool -change "$(otool -D "$TORCH_LIB/libomp.dylib" | tail -1)" @rpath/libomp.dylib "$OUTPUT" 2>/dev/null
codesign -f -s - "$OUTPUT" 2>/dev/null
echo "Built: $OUTPUT"

if [ "$ENV" = platformer ]; then
    clang -O2 -Wall "${ENV_INCLUDES[@]}" \
        "$ENV_DIR/$ENV.c" "${ENV_LIBS[@]}" "${ENV_FRAMEWORKS[@]}" -lm -o "$BUILD/$ENV"
    echo "Built: $BUILD/$ENV"
fi

if [ "$ENV" = sm64 ]; then
    # Homebrew's OpenMP rather than torch's, because nothing here loads torch:
    # the tool is the env without the trainer, and torch's copy carries an
    # install name that only resolves inside a process torch has already set up.
    OMP_LIB="$(brew --prefix libomp)/lib"
    clang -O2 -Wall -Xpreprocessor -fopenmp -I"$OMP_INCLUDE" "${ENV_INCLUDES[@]}" \
        "$ENV_DIR/$ENV.c" -L"$OMP_LIB" -lomp -Wl,-rpath,"$OMP_LIB" -lm -o "$BUILD/sm64_tool"
    echo "Built: $BUILD/sm64_tool"
fi

# puffer's CLI only reads configs from inside the PufferLib checkout. Link ours in
# and keep the link out of the submodule's git status.
ln -sfn "../../../$ENV_DIR/$ENV.ini" "$PUFFER/config/$ENV.ini"
EXCLUDE=$(git -C "$PUFFER" rev-parse --path-format=absolute --git-path info/exclude)
mkdir -p "$(dirname "$EXCLUDE")"
grep -qxF "config/$ENV.ini" "$EXCLUDE" 2>/dev/null || echo "config/$ENV.ini" >> "$EXCLUDE"
