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
cuda_host_args=()
if [[ $cuda == ON ]]; then
 explicit_host=0
 for arg in "$@"; do
  if [[ $arg == -DCMAKE_CUDA_HOST_COMPILER=* ]]; then explicit_host=1; break; fi
 done
 if [[ $explicit_host == 0 ]]; then
  nvcc_path=${CUDACXX:-$(command -v nvcc || true)}
  if [[ -z $nvcc_path ]]; then echo 'CUDA backend selected but nvcc was not found.' >&2; exit 1; fi
  nvcc_path=$(readlink -f "$nvcc_path")
  toolkit_root=$(cd -- "$(dirname -- "$nvcc_path")/.." && pwd)
  host_config="$toolkit_root/include/crt/host_config.h"
  if [[ ! -r $host_config ]]; then host_config="$toolkit_root/targets/x86_64-linux/include/crt/host_config.h"; fi
  max_gcc=$(grep -Eo '__GNUC__[[:space:]]*>[[:space:]]*[0-9]+' "$host_config" 2>/dev/null | grep -Eo '[0-9]+' | tail -1 || true)
  if [[ -z $max_gcc ]]; then
   max_gcc=$(grep -Eio 'gcc versions later than [0-9]+' "$host_config" 2>/dev/null | grep -Eo '[0-9]+' | tail -1 || true)
  fi
  if [[ -z $max_gcc ]]; then echo "Cannot determine nvcc's supported GCC version from $host_config; set -DCMAKE_CUDA_HOST_COMPILER explicitly." >&2; exit 1; fi
  for ((version=max_gcc; version>=10; version--)); do
   if [[ -x /usr/bin/g++-$version ]]; then cuda_host_args+=("-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-$version"); break; fi
  done
  if [[ ${#cuda_host_args[@]} == 0 ]]; then echo "CUDA supports GCC up to $max_gcc; install a matching g++ or pass -DCMAKE_CUDA_HOST_COMPILER=PATH." >&2; exit 1; fi
  selected_host=${cuda_host_args[0]#*=}
  cached_host=$(sed -n 's/^CMAKE_CUDA_HOST_COMPILER:FILEPATH=//p' "$build/CMakeCache.txt" 2>/dev/null | head -1)
  if [[ -n $cached_host && $cached_host != "$selected_host" ]]; then rm -rf -- "$build"; fi
 fi
fi
cmake -S "$root" -B "$build" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
 -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_LIBEXECDIR=libexec \
 -DFCITX_INSTALL_LIBDIR="$prefix/lib" -DFCITX_INSTALL_PKGDATADIR="$prefix/share/fcitx5" \
 -DGGML_CUDA="$cuda" -DGGML_VULKAN="$vulkan" "${cuda_host_args[@]}" "${@}"
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
