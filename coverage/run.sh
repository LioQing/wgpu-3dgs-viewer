#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TARGET_DIR="$SCRIPT_DIR/../target/coverage"

mkdir -p "$TARGET_DIR"

OUT_BIN="$TARGET_DIR/coverage"

cleanup() {
  rm -f "$OUT_BIN"
}
trap cleanup 0

rustc "$SCRIPT_DIR/coverage.rs" -o "$OUT_BIN"
"$OUT_BIN"
