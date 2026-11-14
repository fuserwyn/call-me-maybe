#!/bin/sh
# Remove everything that make run, make runner and the tests create.
#
# In the project: .venv, .venv-cpu, data/output and Python tool caches.
# In goinfre (when goinfre.sh used it): this project's venv, the uv cache
# and the Hugging Face cache with the downloaded model.
# Caches in the home directory are shared with other projects and are kept.
#
# Usage: sh clean.sh   (override the location with GOINFRE=/path)

GOINFRE="${GOINFRE:-/goinfre/${USER:-}}"
project="$(basename "$(pwd)")"

remove() {
    if [ -e "$1" ]; then
        chmod -R u+w "$1" 2>/dev/null
        rm -rf -- "$1" && echo "removed $1"
    fi
}

remove .venv
remove .venv-cpu
remove data/output
remove .coverage
find . -path ./.git -prune -o -type d \
    \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) \
    -prune -exec rm -rf {} +

if [ -n "${USER:-}" ] && [ -n "$project" ] && [ -d "$GOINFRE" ] \
    && [ -w "$GOINFRE" ]; then
    remove "$GOINFRE/venvs/$project"
    rmdir "$GOINFRE/venvs" 2>/dev/null
    remove "$GOINFRE/.uv-cache"
    remove "$GOINFRE/.huggingface"
fi

echo "clean: done"
