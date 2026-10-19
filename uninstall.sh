#!/usr/bin/env bash
# SPDX-License-Identifier: Unlicense
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prefix=${PREFIX:-/usr}
if [[ -w $prefix ]]; then
 python3 "$root/distribution/deploy.py" uninstall "${prefix%/}"
else
 sudo python3 "$root/distribution/deploy.py" uninstall "${prefix%/}"
fi
