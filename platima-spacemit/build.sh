#!/usr/bin/env bash
#
# build.sh
# --------
# Native build of the SpacemiT llama.cpp fork on the K3 (RISC-V, A100 cores).
#
# Uses cmake/riscv64-spacemit-linux-gnu-gcc.cmake as the toolchain file. The
# file detects a riscv host and skips setting cross-compiler paths, but still
# applies the SpacemiT-recommended compile flags:
#
#   -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d
#       Explicit A100 ISA target (matches K3 A100 hart ISA).
#   -fno-tree-vectorize -fno-tree-loop-vectorize
#       Disable GCC auto-vectorizer. The SpacemiT IME2 kernels are
#       hand-written; auto-vec would either redo the work suboptimally or
#       emit instructions that conflict with the IME2 pipeline.
#   -latomic linker flag (riscv64 atomic ops).
#
# These are NOT picked up by default when CMAKE_TOOLCHAIN_FILE is omitted -
# system gcc applies its own (less SpacemiT-tuned) defaults instead.
#
# Usage:
#   ./build.sh              # configure (if needed) + build all targets
#   ./build.sh clean        # remove build/ first, then full build
#   ./build.sh <target>     # configure + build a single CMake target
#   ./build.sh clean <tgt>  # clean + build single target
#
# Runtime reminder (the single most important finding from the bring-up log):
# always invoke binaries with LD_LIBRARY_PATH pointing at the SpacemiT libs
# first, otherwise Debian's system libggml shadows them and compute silently
# falls back to the X100 general cores (~3x lower tg, ~17x lower pg). Verified
# good runs look like:
#
#   LD_LIBRARY_PATH=$PWD/build/bin ./build/bin/llama-cli --version
#   ... CPU_RISCV64_SPACEMIT: ... use_ime2: 1 ...
#
# A startup line without 'use_ime2: 1' means the wrong libggml was loaded.

set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
    echo ">>> removing build/"
    rm -rf build
    shift
fi

TARGET="${1:-}"

if [[ ! -f build/CMakeCache.txt ]]; then
    echo ">>> configuring (native riscv64, SpacemiT A100 / IME2 / hand-kernel-friendly flags)"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="${PWD}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake" \
        -DGGML_CPU_RISCV64_SPACEMIT=ON \
        -DGGML_CPU_REPACK=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DGGML_RVV=ON \
        -DGGML_RV_ZVFH=ON \
        -DGGML_RV_ZFH=ON \
        -DGGML_RV_ZICBOP=ON \
        -DGGML_RV_ZIHINTPAUSE=ON \
        -DGGML_RV_ZBA=ON
fi

if [[ -n "$TARGET" ]]; then
    echo ">>> building target: $TARGET"
    cmake --build build --parallel "$(nproc)" --config Release --target "$TARGET"
else
    echo ">>> building all targets"
    cmake --build build --parallel "$(nproc)" --config Release
fi

echo
echo ">>> done. quick verify:"
echo "    LD_LIBRARY_PATH=\$PWD/build/bin \$PWD/build/bin/llama-cli --version"
echo "    expect: 'use_ime2: 1' in the startup banner."
