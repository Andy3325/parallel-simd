#!/bin/sh
set -eu

PROJECT_DIR=/home/s2411560/guess
cd "$PROJECT_DIR"

LATEST_OUT=$(ls -t test.sh.o* 2>/dev/null | head -n 1 || true)
if [ -z "$LATEST_OUT" ]; then
  echo "No test.sh.o* output file found." >&2
  exit 1
fi

LATEST_RESULT_DIR=$(ls -dt results/qsub_final_* 2>/dev/null | head -n 1 || true)
if [ -z "$LATEST_RESULT_DIR" ]; then
  echo "No results/qsub_final_* directory found." >&2
  exit 1
fi

cp "$LATEST_OUT" "$LATEST_RESULT_DIR/final_qsub_output.txt"
{
  echo "source_output=$LATEST_OUT"
  grep -E '^(Guess time|Hash time|Train time|Cracked):' "$LATEST_OUT" || true
} > "$LATEST_RESULT_DIR/summary.txt"

echo "Copied $LATEST_OUT to $LATEST_RESULT_DIR/final_qsub_output.txt"
echo "Summary:"
cat "$LATEST_RESULT_DIR/summary.txt"
