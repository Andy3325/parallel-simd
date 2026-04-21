# parallel-simd

`parallel-simd` is a SIMD parallel programming lab project. The current focus is MD5 NEON 4-way parallelization within the existing password-guessing experiment codebase.

This repository is being prepared conservatively for ongoing lab work, so existing source files remain in place unless there is a clear need to reorganize them later.

## Current Focus

- SIMD parallel programming lab setup
- MD5 NEON 4-way parallelization
- Preserving the current baseline implementation while preparing for later optimization work

## Project Tree

```text
parallel-simd/
├── README.md
├── .gitignore
├── include/
├── report/
├── scripts/
├── src/
├── tests/
├── PCFG.h
├── md5.h
├── md5.cpp
├── main.cpp
├── guessing.cpp
├── train.cpp
├── correctness.cpp
├── qsub.sh
└── test.sh
```

## Existing Core Files

- `md5.h`
- `md5.cpp`
- `main.cpp`
- `guessing.cpp`
- `train.cpp`

## Build And Run

Exact long-term build commands may change as the project is reorganized for lab work.

Current baseline build example:

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp -O2 -o main
```

Current baseline run placeholder:

```bash
./main
```

## Notes

- This is a lab repository setup stage, not the final optimized SIMD implementation.
- Existing code and history are preserved for incremental NEON-focused development.
