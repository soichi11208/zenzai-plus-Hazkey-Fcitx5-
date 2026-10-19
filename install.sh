#!/usr/bin/env bash
# SPDX-License-Identifier: Unlicense
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prefix=${PREFIX:-/usr}
backend=${BACKEND:-cuda}
jobs=${JOBS:-4}
if [[ $prefix != /* || $prefix == / ]]; then echo 'PREFIX must be an absolute installation prefix, e.g. /usr or /usr/local' >&2; exit 1; fi
prefix=${prefix%/}
case $backend in
 cuda) cuda=ON; vulkan=OFF ;;
 vulkan) cuda=OFF; vulkan=ON ;;
 cpu) cuda=OFF; vulkan=OFF ;;
 *) echo 'BACKEND=cuda|vulkan|cpu' >&2; exit 1 ;;
esac
build="$root/.build-distribution/$backend"
cmake -S "$root" -B "$build" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
 -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_LIBEXECDIR=libexec \
 -DFCITX_INSTALL_LIBDIR="$prefix/lib" -DFCITX_INSTALL_PKGDATADIR="$prefix/share/fcitx5" \
 -DGGML_CUDA="$cuda" -DGGML_VULKAN="$vulkan" "${@}"
cmake --build "$build" --parallel "$jobs"
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
DESTDIR="$stage" cmake --install "$build"
if [[ -w $prefix ]]; then
 python3 "$root/distribution/deploy.py" install "$prefix" "$stage"
else
 sudo python3 "$root/distribution/deploy.py" install "$prefix" "$stage"
fi
printf '\nInstalled zenzai-plus-Hazkey-Fcitx5. Log out and back in, then select it in Fcitx5.\nEnable Zenzai in settings to download the FT model; select your GPU in the device setting.\n'
