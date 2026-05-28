#!/bin/sh
set -eu

PROJECT_DIR=/home/s2411560/guess
cd "$PROJECT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/qsub_final_$TS"
SNAPSHOT_DIR="$RESULT_DIR/src_snapshot"
mkdir -p "$SNAPSHOT_DIR"

cp PCFG.h guessing.cpp correctness_guess.cpp md5.cpp md5.h train.cpp main.cpp test.sh "$SNAPSHOT_DIR"/
[ -f README_final.md ] && cp README_final.md "$SNAPSHOT_DIR"/

echo "Building final test binary..."
g++ -O2 -std=c++17 \
  -DENABLE_OPENMP_GENERATE \
  -DENABLE_RELAXED_HEAP_PRIORITY \
  -DENABLE_LAZY_GUESS_BLOCK \
  -DENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH \
  -fopenmp \
  correctness_guess.cpp train.cpp guessing.cpp md5.cpp \
  -o test

echo "Submitting PBS job..."
JOB_ID=$(qsub test.sh)
echo "$JOB_ID" > "$RESULT_DIR/job_id.txt"
echo "job id: $JOB_ID"
echo "result dir: $RESULT_DIR"
echo "Use these commands after the job finishes:"
echo "  qstat"
echo "  ls -lt test.sh.o* test.sh.e*"
echo "  cat test.sh.o${JOB_ID%%.*}"
