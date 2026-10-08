#!/usr/bin/env bash
# Builds the C and Java implementations with the fixed flags used in the study.
set -euo pipefail
cd "$(dirname "$0")"
CFLAGS="-O2 -std=c11 -ffp-contract=off"
mkdir -p c/build java/build
echo "[build] gcc $CFLAGS"
gcc $CFLAGS -o c/build/matmul c/matmul.c
echo "[build] javac --release 17"
javac --release 17 -d java/build java/MatMul.java
echo "[build] done"
