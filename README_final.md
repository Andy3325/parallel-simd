# Final Lazy Block Parallel Hash Version

This is the final submission route for `/home/s2411560/guess`.

## Final build command

```sh
g++ -O2 -std=c++17 \
  -DENABLE_OPENMP_GENERATE \
  -DENABLE_RELAXED_HEAP_PRIORITY \
  -DENABLE_LAZY_GUESS_BLOCK \
  -DENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH \
  -fopenmp \
  correctness_guess.cpp train.cpp guessing.cpp md5.cpp \
  -o test
```

Runtime settings for the final PBS run:

```sh
OMP_NUM_THREADS=4 LAZY_BLOCK_HASH_CHUNK_SIZE=8192 /home/s2411560/guess/test
```

## qsub flow

```sh
cd /home/s2411560/guess
chmod +x run_final_qsub.sh collect_qsub_result.sh
./run_final_qsub.sh
qstat
ls -lt test.sh.o* test.sh.e*
cat test.sh.oJOBID
./collect_qsub_result.sh
```

`run_final_qsub.sh` creates `results/qsub_final_TIMESTAMP/`, saves a source snapshot, builds `/home/s2411560/guess/test`, and submits `qsub test.sh`. It does not run `./test` directly.

## Final macros

- `ENABLE_OPENMP_GENERATE`: parallelizes candidate generation with OpenMP.
- `ENABLE_RELAXED_HEAP_PRIORITY`: uses heap-based priority queue maintenance instead of sorted-vector insertion.
- `ENABLE_LAZY_GUESS_BLOCK`: stores generated candidates as lazy blocks instead of materialized strings.
- `ENABLE_SIMD_HASH_BATCH`: generic `MD5Hash4` batch hash path for serial SIMD hash and lazy-block generic SIMD ablations.
- `ENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH`: consumes lazy blocks in parallel hash chunks and uses `MD5Hash4` for SIMD batches.

## Correctness boundary

`correctness_guess.cpp` is the final test entry. It preserves the official driver behavior and is not an algorithm-core shortcut.

Unchanged:

- training path: `/guessdata/Rockyou-singleLined-full.txt`
- `test_count=1000000`
- `generate_n=10000000`
- `Cracked` membership logic
- MD5 execution
- output format
- candidate count limit

Changed only:

- candidate representation
- hash-stage consumption method
- OpenMP/SIMD execution strategy
