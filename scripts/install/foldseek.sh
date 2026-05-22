#!/usr/bin/env bash
# Install foldseek into tools/foldseek/bin/ for aarch64 (GH200) or AVX2.
# Used by the ecstasy_v1 build pipeline (scripts/build_ecstasy_v1/04_run_foldseek.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
DEST="$HERE/tools/foldseek/bin"
mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap "rm -rf $TMP" EXIT

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64) PKG="foldseek-linux-arm64.tar.gz" ;;
  x86_64)        PKG="foldseek-linux-avx2.tar.gz"  ;;
  *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
esac

URL="https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/$PKG"
echo "downloading $URL"
curl -fsSL -o "$TMP/$PKG" "$URL"
tar -xzf "$TMP/$PKG" -C "$TMP"
cp "$TMP/foldseek/bin/foldseek" "$DEST/foldseek"
chmod +x "$DEST/foldseek"
echo "installed: $("$DEST/foldseek" version)"
