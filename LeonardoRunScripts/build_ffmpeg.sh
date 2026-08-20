#!/usr/bin/env bash
# One-time build of a static ffmpeg+libx264 for trainer.video_validation=True.

set -euo pipefail

OPT=/leonardo/home/userexternal/sshamsi0/.local/opt
VENV=/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1

NASM_VER=3.02
FFMPEG_VER=9.0.1

NASM_PREFIX="$OPT/nasm-${NASM_VER}"
X264_PREFIX="$OPT/x264"          # no tagged releases upstream, git snapshot
FFMPEG_PREFIX="$OPT/ffmpeg-${FFMPEG_VER}"

mkdir -p "$NASM_PREFIX"/{source,build} "$X264_PREFIX"/{source,build} "$FFMPEG_PREFIX"/{source,build}

module purge
module load gcc/12.2.0

ARCH_FLAGS="-march=icelake-server -mtune=icelake-server -O3"

# NASM: x264's hand-written AVX-512/VNNI asm kernels need it
# without it x264 falls back to plain C
if [ ! -x "$NASM_PREFIX/bin/nasm" ]; then
  curl -L "https://www.nasm.us/pub/nasm/releasebuilds/${NASM_VER}/nasm-${NASM_VER}.tar.gz" \
    | tar xz -C "$NASM_PREFIX/source" --strip-components=1
  cd "$NASM_PREFIX/build"
  "$NASM_PREFIX/source/configure" --prefix="$NASM_PREFIX" CFLAGS="$ARCH_FLAGS"
  make -j"$(nproc)"
  make install
fi
export PATH="$NASM_PREFIX/bin:$PATH"

# x264: encoder We need. We make it static so no LD_LIBRARY_PATH juggling by FFMPEG
if [ ! -f "$X264_PREFIX/lib/libx264.a" ]; then
  git clone --depth 1 https://code.videolan.org/videolan/x264.git "$X264_PREFIX/source"
  cd "$X264_PREFIX/build"
  "$X264_PREFIX/source/configure" \
    --prefix="$X264_PREFIX" \
    --enable-static --disable-shared \
    --enable-opencl \
    --enable-lto \
    --extra-cflags="$ARCH_FLAGS"
  make -j"$(nproc)"
  make install
fi

# --- ffmpeg itself. Deliberately NOT --disable-everything: the only thing we
# actually need beyond ffmpeg's defaults is --enable-gpl --enable-libx264, and
# leaving the rest at defaults means configure auto-skips anything requiring
# libraries we don't have rather than us hand-maintaining an allowlist of
# muxers/filters/protocols that breaks the next time the_well tweaks its
# ffmpeg invocation (codec, preset, filter graph, etc). --enable-hardcoded-tables
# precomputes codec tables at compile time instead of on first use; there's no
# GPU hardware encoder to reach for here -- the A100 die (unlike T4/A10/L40/H100)
# has no NVENC block, so h264_nvenc isn't an option on this cluster.
if [ ! -x "$FFMPEG_PREFIX/bin/ffmpeg" ]; then
  curl -L "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VER}.tar.bz2" \
    | tar xj -C "$FFMPEG_PREFIX/source" --strip-components=1
  cd "$FFMPEG_PREFIX/build"
  PKG_CONFIG_PATH="$X264_PREFIX/lib/pkgconfig" "$FFMPEG_PREFIX/source/configure" \
    --prefix="$FFMPEG_PREFIX" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I$X264_PREFIX/include $ARCH_FLAGS" \
    --extra-ldflags="-L$X264_PREFIX/lib" \
    --extra-libs="-lpthread -lm" \
    --enable-gpl \
    --enable-libx264 \
    --enable-hardcoded-tables \
    --disable-shared --enable-static \
    --disable-doc --disable-debug --disable-ffplay
  make -j"$(nproc)"
  make install
fi
  