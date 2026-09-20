#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /path/to/TCOD" >&2
    exit 2
fi

target_root=$1
target_workflows="$target_root/trinity/common/workflows/envs/TCOD"

if [ ! -d "$target_workflows" ]; then
    echo "error: target does not look like a TCOD checkout: $target_root" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
release_root=$(CDPATH= cd -- "$script_dir/.." && pwd)

cp -R "$release_root/trinity/." "$target_root/trinity/"
echo "FutureBridge-OPD workflows installed into $target_workflows"
