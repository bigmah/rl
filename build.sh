#!/bin/bash
# Build the platformer env for PufferLib on Apple Silicon.
#
#   ./build.sh  ->  vendor/PufferLib/pufferlib/_C*.so   CPU training backend (puffer ... --slowly)
#                   build/platformer                     human-playable binary
#
# PufferLib's own build.sh targets Linux/CUDA (-mavx2, gcc -fopenmp, nvcc) and
# only finds envs under ocean/, so this mirrors its --cpu mode for macOS.
set -euo pipefail
cd "$(dirname "$0")"

ENV=platformer
ENV_DIR=envs/$ENV
PUFFER=vendor/PufferLib
BUILD=build
PYTHON=.venv/bin/python

if [ ! -f "$PUFFER/src/vecenv.h" ]; then
    echo "PufferLib submodule missing: run 'git submodule update --init'" && exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "Python env missing: run 'uv sync'" && exit 1
fi

mkdir -p "$BUILD"
RAYLIB=$BUILD/raylib-5.5_macos
if [ ! -d "$RAYLIB" ]; then
    echo "Downloading raylib..."
    curl -sL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_macos.tar.gz | tar xz -C "$BUILD"
fi

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
FRAMEWORKS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)

echo "Compiling $ENV env..."
clang -c -O2 -DNDEBUG -Wall -fPIC -fvisibility=hidden -DPLATFORM_DESKTOP \
    -Xpreprocessor -fopenmp -I"$OMP_INCLUDE" \
    -I"$PUFFER/src" -I"$RAYLIB/include" -I"$ENV_DIR" \
    "$ENV_DIR/binding.c" -o "$BUILD/binding.o"

# Only depends on PufferLib sources, so skip it unless the submodule changed
if [ ! "$BUILD/bindings_cpu.o" -nt "$PUFFER/src/bindings_cpu.cpp" ] || [ ! "$BUILD/bindings_cpu.o" -nt "$PUFFER/src/vecenv.h" ]; then
    echo "Compiling CPU training backend..."
    clang++ -c -O2 -std=c++17 -fPIC -DPLATFORM_DESKTOP -DPRECISION_FLOAT -DENV_NAME=$ENV \
        -I"$PUFFER/src" -I"$PY_INCLUDE" -I"$PYBIND_INCLUDE" \
        "$PUFFER/src/bindings_cpu.cpp" -o "$BUILD/bindings_cpu.o"
fi

OUTPUT="$PUFFER/pufferlib/_C$EXT_SUFFIX"
clang++ -shared "$BUILD/bindings_cpu.o" "$BUILD/binding.o" "$RAYLIB/lib/libraylib.a" \
    "${FRAMEWORKS[@]}" -L"$TORCH_LIB" -lomp -Wl,-rpath,"$TORCH_LIB" \
    -undefined dynamic_lookup -o "$OUTPUT"
# torch's libomp.dylib carries a stale install name (/opt/llvm-openmp/...), so
# point the extension at the copy torch loads, then re-sign after editing
install_name_tool -change "$(otool -D "$TORCH_LIB/libomp.dylib" | tail -1)" @rpath/libomp.dylib "$OUTPUT" 2>/dev/null
codesign -f -s - "$OUTPUT" 2>/dev/null
echo "Built: $OUTPUT"

clang -O2 -Wall -DPLATFORM_DESKTOP -I"$RAYLIB/include" -I"$ENV_DIR" \
    "$ENV_DIR/$ENV.c" "$RAYLIB/lib/libraylib.a" "${FRAMEWORKS[@]}" -lm -o "$BUILD/$ENV"
echo "Built: $BUILD/$ENV"

# puffer's CLI only reads configs from inside the PufferLib checkout. Link ours in
# and keep the link out of the submodule's git status.
ln -sfn "../../../$ENV_DIR/$ENV.ini" "$PUFFER/config/$ENV.ini"
EXCLUDE=$(git -C "$PUFFER" rev-parse --path-format=absolute --git-path info/exclude)
mkdir -p "$(dirname "$EXCLUDE")"
grep -qxF "config/$ENV.ini" "$EXCLUDE" 2>/dev/null || echo "config/$ENV.ini" >> "$EXCLUDE"
