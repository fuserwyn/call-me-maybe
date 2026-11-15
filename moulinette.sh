#!/bin/sh
# Grade the project with the 42 moulinette.
#
# Put moulinette.zip (from the evaluation page) in the project root, then:
#   sh moulinette.sh [private|public]
# It unpacks the moulinette into .moulinette/, generates the exercise set,
# runs this project on it and grades the answers. Every uv command goes
# through goinfre.sh, so on 42 machines the venvs and the model stay in
# goinfre.
#
# PYTHON chooses how the project runs (default: the uv venv), e.g.
#   PYTHON=.venv-cpu/bin/python sh moulinette.sh     (CPU-only venv)
# MOULINETTE_ZIP overrides the archive path.

set -e

SET="${1:-private}"
case "$SET" in
    private|public) ;;
    *) echo "Usage: sh moulinette.sh [private|public]" >&2; exit 2 ;;
esac

ROOT="$(pwd)"
ZIP="${MOULINETTE_ZIP:-$ROOT/moulinette.zip}"
WORK="$ROOT/.moulinette"

if [ ! -f "$ZIP" ]; then
    echo "moulinette: $ZIP not found." >&2
    echo "Download moulinette.zip from the evaluation page and put it in" \
        "the project root." >&2
    exit 1
fi

# A venv activated in this shell would make uv warn and ignore it.
unset VIRTUAL_ENV

echo "== unpack $ZIP"
rm -rf "$WORK"
mkdir -p "$WORK"
unzip -q -o "$ZIP" -d "$WORK"
M="$WORK/moulinette"
if [ ! -d "$M" ]; then
    echo "moulinette: no moulinette/ folder inside $ZIP" >&2
    exit 1
fi

echo "== generate the $SET set"
(cd "$M" && sh "$ROOT/goinfre.sh" uv sync -q \
    && sh "$ROOT/goinfre.sh" uv run python -m moulinette \
        prepare_exercises --set "$SET")

echo "== run the project"
if [ -n "${PYTHON:-}" ]; then
    RUN="$PYTHON"
else
    sh goinfre.sh uv sync -q
    RUN="uv run python"
fi
# shellcheck disable=SC2086  # RUN is a command with arguments
sh goinfre.sh $RUN -m src \
    --functions_definition "$M/data/input/functions_definition.json" \
    --input "$M/data/input/function_calling_tests.json" \
    --output "$M/student_answers.json"

echo "== grade"
(cd "$M" && sh "$ROOT/goinfre.sh" uv run python -m moulinette \
    grade_student_answers --set "$SET" \
    --student_answer_path student_answers.json)
