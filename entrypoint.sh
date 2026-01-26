#!/bin/bash

if [ -z "$1" ]; then
    echo "No algorithm specified"
    exit 1
fi

ALGO=$1
MODE=$2

case "$MODE" in
    "repro" | "r")
        TARGET="run_reproducibility"
        shift 2
        ;;
    "run" | "normal" | "n" | "")
        TARGET="run"
        [ -z "$MODE" ] && shift 1 || shift 2
        ;;
    *)
        TARGET="run"
        shift 1
        ;;
esac

echo "Running $ALGO -> $TARGET"
exec python -m "algorithms.$ALGO.$TARGET" "$@"