#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {g_opd|fire_opd|ablation} UPSTREAM_REPOSITORY" >&2
  exit 64
fi

variant=$1
target=$(realpath "$2")
release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -d "$target/verl/verl/trainer" ]]; then
  echo "not a supported G-OPD/FiRe-OPD checkout: $target" >&2
  exit 66
fi

copy_tree() {
  local source=$1
  if [[ ! -d "$source" ]]; then
    echo "missing overlay: $source" >&2
    exit 66
  fi
  cp -a "$source/." "$target/"
}

if [[ -d "$target/.git" ]] && [[ -n "$(git -C "$target" status --porcelain)" ]]; then
  echo "warning: applying overlay to a checkout that already has local changes" >&2
fi

copy_tree "$release_root/overlays/common"
case "$variant" in
  g_opd)
    copy_tree "$release_root/overlays/g_opd"
    ;;
  fire_opd)
    copy_tree "$release_root/overlays/fire_opd"
    ;;
  ablation)
    copy_tree "$release_root/overlays/fire_opd"
    copy_tree "$release_root/overlays/ablation"
    ;;
  *)
    echo "unknown overlay: $variant" >&2
    exit 64
    ;;
esac

echo "applied $variant overlay to $target"

