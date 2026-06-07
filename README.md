# Lab4 MPI Password Guessing

This repository contains the Lab4 MPI password-guessing implementations and
experiment artifacts. The final submission path is a single MPI mainline built
as `main`.

## TA Check Entry

`qsub_mpi.sh` is the default TA-check submission script. It runs the executable
named `main`.

The final default checking entry is compiled from:

```text
correctness_guess_mpi.cpp
```

The final mainline uses:

- MPI distributed training
- rank-0 PT priority scheduling
- PT-internal value index range splitting across ranks
- MPI broadcast/reduce for each PT in basic mode
- strict PT batch broadcast/reduce in batch mode

The previous successful experiment sources remain available for comparison:

- `correctness_guess_mpi_static.cpp`
- `correctness_guess_mpi_master_worker.cpp`
- `correctness_guess_mpi_master_worker_nonblocking.cpp`
- `correctness_guess_mpi_hybrid_openmp.cpp`

The experimental `correctness_guess_mpi_fast.cpp` path is not the final
mainline.

## Build

Build the final executable on the server:

```sh
mpic++ -O2 -std=c++17 \
  correctness_guess_mpi.cpp train_mpi.cpp guessing_mpi.cpp \
  train.cpp guessing.cpp md5.cpp \
  -o main
```

## Run

The server cannot reliably execute binaries directly from `/home/${USER}/guess`
on compute nodes. `qsub_mpi.sh` preserves the working node-copy flow:

1. copy `/home/${USER}/guess/main` from `master_ubss1` to each allocated node;
2. run from `/home/${USER}/mpi_final_work`;
3. launch with `/usr/local/bin/mpiexec`.

Run correctness checks in this order:

```sh
qsub -v NP=1,MODE=basic,BATCH_SIZE=1 qsub_mpi.sh
qsub -v NP=2,MODE=basic,BATCH_SIZE=1 qsub_mpi.sh
qsub -v NP=4,MODE=basic,BATCH_SIZE=1 qsub_mpi.sh
qsub -v NP=8,MODE=basic,BATCH_SIZE=1 qsub_mpi.sh
```

Batch mode uses the same model and PT range split, but broadcasts up to
`BATCH_SIZE` PTs per synchronization round. Rank 0 uses strict queue evolution:
after selecting each PT, it immediately inserts that PT's successors before
selecting the next PT.

Batch test examples:

```sh
qsub -v NP=1,MODE=batch,BATCH_SIZE=2 qsub_mpi.sh
qsub -v NP=8,MODE=batch,BATCH_SIZE=2 qsub_mpi.sh
qsub -v NP=8,MODE=batch,BATCH_SIZE=4 qsub_mpi.sh
qsub -v NP=8,MODE=batch,BATCH_SIZE=8 qsub_mpi.sh
qsub -v NP=8,MODE=batch,BATCH_SIZE=16 qsub_mpi.sh
```

Final submitted run command:

```sh
qsub -v NP=8,MODE=batch,BATCH_SIZE=16 qsub_mpi.sh
```

Direct executable interface:

```sh
./main --mode basic
./main --mode batch --batch-size 8
```

Historical note: a performance-oriented `loose-batch` experiment was tested.
It popped a batch before inserting successors, which caused `Generated` /
`Cracked` drift, inflated PT task counts, and did not outperform strict batch.
It is not part of the final supported mainline.

## Correctness Target

The strict target from the previously verified final run remains:

```text
Generated: 9528822
Cracked: 358217
```

Verified final batch result:

```text
MPI mode: batch_pt_range_distributed_training
MPI size: 8
Batch size: 16
Generated: 9528822
Cracked: 358217
Train time: 11.017928
Guess time: 0.429608
Hash time: 0.809533
MPI total Guess+Hash time: 1.239141
MPI generate only time: 0.090232
MPI compute time: 0.895253
MPI overhead/non-compute time: 0.343888
Total wall time: 14.533796
Total PT tasks: 515
Total batch rounds: 38
```

If distributed training causes drift, inspect raw-count merge, deterministic
ordering, PT serialization, and range coverage before accepting changed totals.

## Output Fields

Rank 0 prints grep-friendly lines:

```text
MPI mode: basic_pt_range_distributed_training
MPI size: <N>
Batch size: <B>   # batch mode only
Generated: <...>
Cracked: <...>
Train time: <...>
Guess time: <...>
Hash time: <...>
MPI total Guess+Hash time: <...>
MPI generate only time: <...>
MPI compute time: <...>
MPI overhead/non-compute time: <...>
Total wall time: <...>
Total PT tasks: <...>
Total batch rounds: <...>   # batch mode only
```

Pipeline, OpenMP hybrid, and SIMD changes are intentionally out of scope for
this final mainline phase.
