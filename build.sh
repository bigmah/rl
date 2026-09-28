#!/bin/bash
# Build Super Mario 64 into PufferLib 5.0's CUDA trainer, on arm64 Linux.
#
#   ./build.sh
#
#       ./puffer              PufferLib's trainer (vendor/PufferLib/src/pufferl.cu)
#                             with envs/sm64/sm64.h compiled in: ./puffer train|eval
#       build/sm64_check      the env with no trainer and no GPU: replay a demo
#                             through the reward, or benchmark the games
#
# PufferLib's own build.sh compiles an env out of its ocean/ folder, downloads
# an x86-64 raylib and looks for NCCL where x86-64 Debian keeps it. This does
# the same compile around envs/sm64 instead, and on arm64 as well.
#
# On a Mac (no CUDA) it builds the game and sm64_check only.
set -euo pipefail
cd "$(dirname "$0")"

ENV=sm64
ENV_DIR=envs/$ENV
PUFFER=vendor/PufferLib
BUILD=build
if [ ! -f "$PUFFER/src/pufferenv.h" ]; then
    echo "Fetching PufferLib 5.0..."
    git submodule update --init "$PUFFER"
fi

mkdir -p "$BUILD"

# What a fresh Linux machine needs, on Debian or Ubuntu, besides CUDA and NCCL.
# Named once, so every error that means "a package is missing" says the same thing.
APT_PACKAGES="build-essential clang cmake ninja-build git curl python3 ccache libsdl2-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev"

# pufferenv.h includes raylib for every env, as it does in PufferLib's own build.
# RAYLIB names one that is already somewhere, with its include/ and lib/.
RAYLIB_FROM_SOURCE=0
case "$(uname -s)" in
    Darwin)
        PLATFORM=macos
        CC=${CC:-clang}
        LIB_EXT=dylib
        RAYLIB_RELEASE=raylib-5.5_macos
        SYSTEM_LIBS=()
        # sm64_check steps games on OpenMP threads, which Apple clang doesn't ship
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

CFLAGS=(-O2 -Wall -DPLATFORM_DESKTOP -I"$PUFFER/src" -I"$RAYLIB/include" -I"$ENV_DIR")

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
#define SM64_DATA "$(pwd)/$BUILD/sm64"   /* a folder a star under it, with its savestate */
#define SM64_LOG "$(pwd)/$BUILD/sm64/game.log"
PATHS
CFLAGS+=(-I"$BUILD/sm64" -I"$N64BUNDLER/ModernReality/include/modernreality")

# --- the env without the trainer ----------------------------------------------
echo "Compiling sm64_check..."
"$CC" "${CFLAGS[@]}" "$ENV_DIR/sm64_check.c" "${OMP_FLAGS[@]}" "${SYSTEM_LIBS[@]+"${SYSTEM_LIBS[@]}"}" -lm \
    -o "$BUILD/sm64_check"
echo "Built: $BUILD/sm64_check"

# --- PufferLib's CUDA trainer, with the env compiled in -------------------------
if ! command -v nvcc >/dev/null && [ -z "${CUDA_HOME:-}" ] && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    if [ "$PLATFORM" = macos ]; then
        echo "No CUDA on a Mac: built the game and sm64_check, not the trainer."
        exit 0
    fi
    echo "No nvcc. Put the CUDA toolkit's bin/ on PATH, or set CUDA_HOME (usually /usr/local/cuda)."
    exit 1
fi
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)")")}}
NVCC="$CUDA_HOME/bin/nvcc"
command -v ccache >/dev/null && NVCC="ccache $NVCC"
MULTIARCH=/usr/lib/$(uname -m)-linux-gnu

# NCCL: the trainer links it for multi-GPU even on one. From NVIDIA's apt repo
# (libnccl2, libnccl-dev), inside the toolkit, or from the nvidia-nccl-cu12 wheel.
NCCL_INCLUDE=""
NCCL_LIB=""
for dir in /usr/include "$CUDA_HOME/include" /usr/local/include; do
    [ -f "$dir/nccl.h" ] && { NCCL_INCLUDE=$dir; break; }
done
for lib in "$MULTIARCH/libnccl.so" "$CUDA_HOME/lib64/libnccl.so" /usr/lib64/libnccl.so /usr/local/lib/libnccl.so \
           "$MULTIARCH/libnccl.so.2" "$CUDA_HOME/lib64/libnccl.so.2"; do
    [ -f "$lib" ] && { NCCL_LIB=$lib; break; }
done
if [ -z "$NCCL_INCLUDE" ] || [ -z "$NCCL_LIB" ]; then
    WHEEL=$(python3 -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__) if nvidia.nccl.__file__ else list(nvidia.nccl.__path__)[0])" 2>/dev/null || true)
    if [ -n "$WHEEL" ] && [ -f "$WHEEL/include/nccl.h" ]; then
        NCCL_INCLUDE=$WHEEL/include
        NCCL_LIB=$(ls "$WHEEL"/lib/libnccl.so* 2>/dev/null | head -1)
    fi
fi
if [ -z "$NCCL_INCLUDE" ] || [ -z "$NCCL_LIB" ]; then
    echo "No NCCL. Either of:"
    echo "    sudo apt install libnccl2 libnccl-dev        (NVIDIA's CUDA apt repo)"
    echo "    pip install nvidia-nccl-cu12"
    exit 1
fi

# libnvidia-ml comes with the driver as .so.1 only; the toolkit's stubs/ has a
# .so to link against, and the driver's is what loads.
NVML_DIRS=()
[ -f "$MULTIARCH/libnvidia-ml.so" ] || NVML_DIRS=(-L"$CUDA_HOME/lib64/stubs" -L"$CUDA_HOME/targets/sbsa-linux/lib/stubs")

# bf16 by default, as PufferLib builds it; PRECISION=float for fp32.
PRECISION_FLAG=()
[ "${PRECISION:-}" = float ] && PRECISION_FLAG=(-DPRECISION_FLOAT)

echo "Compiling PufferLib's trainer around $ENV_DIR/$ENV.h (nvcc, a minute or two)..."
$NVCC -O2 --threads 0 -arch="${NVCC_ARCH:-native}" -std=c++17 \
    -I"$PUFFER" -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$ENV_DIR" -I"$RAYLIB/include" \
    -I"$BUILD/sm64" -I"$N64BUNDLER/ModernReality/include/modernreality" \
    -I"$CUDA_HOME/include" -I"$CUDA_HOME/include/cccl" -I"$NCCL_INCLUDE" \
    -DENV_HEADER="\"$(pwd)/$ENV_DIR/$ENV.h\"" -DENV_NAME=$ENV -DPUFFER_ENV_NAME="\"$ENV\"" \
    -DPUFFERLIB_BUILD_MAIN -DPUFFER_SM64 "${PRECISION_FLAG[@]+"${PRECISION_FLAG[@]}"}" \
    -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
    -Xcompiler=-Wno-narrowing -Xcompiler=-Wno-unused-function \
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 --diag-suppress=550 \
    "$PUFFER/src/pufferl.cu" \
    "$RAYLIB/lib/libraylib.a" \
    -L"$CUDA_HOME/lib64" "${NVML_DIRS[@]+"${NVML_DIRS[@]}"}" \
    "$NCCL_LIB" -Xlinker -rpath -Xlinker "$(dirname "$NCCL_LIB")" \
    -lcudart -lnvidia-ml -lcublas -lcusolver -lcurand \
    -lgomp -lGL -lX11 -lm -ldl -lrt -Xlinker=-lpthread \
    -o puffer
echo "Built: ./puffer"
