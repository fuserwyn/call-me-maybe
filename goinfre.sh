#!/bin/sh
# Run a command with the heavy caches moved to goinfre.
#
# The model (~1.4 GB), torch and the uv package cache do not fit in the home
# quota of 42 machines. When a writable goinfre directory exists, the uv
# cache, the virtual environment and the Hugging Face cache are placed there.
# Otherwise the usual locations are kept, so the command runs unchanged.
#
# Usage: sh goinfre.sh <command> [args...]
# Override the location with GOINFRE=/path sh goinfre.sh ...

GOINFRE="${GOINFRE:-/goinfre/$USER}"

if [ -d "$GOINFRE" ] && [ -w "$GOINFRE" ]; then
    project="$(basename "$(pwd)")"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$GOINFRE/.uv-cache}"
    export HF_HOME="${HF_HOME:-$GOINFRE/.huggingface}"
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$GOINFRE/venvs/$project}"
    echo "goinfre: caches and venv in $GOINFRE" >&2
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: sh goinfre.sh <command> [args...]" >&2
    exit 2
fi

exec "$@"
