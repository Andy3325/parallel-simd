# parallel

`parallel` is the current-stage repository for a Parallel Programming experiment focused on the SIMD password guessing subtask.

## Background

This project comes from a parallel programming lab / experiment on password guessing. The current codebase is organized around an MD5-based 4-lane batch processing framework, and the main flow has already been connected to the `MD5Hash4` interface for batched invocation.

This is a current development-stage version rather than the final optimized release.

## Current Version Scope

- MD5-based 4-lane batch hashing entry point via `MD5Hash4`
- Main guessing flow updated to call the batch hashing interface in groups of four
- Training and guessing code kept as the current experiment baseline

## Core Files

- `md5.h`
- `md5.cpp`
- `main.cpp`
- `guessing.cpp`
- `train.cpp`

## Build Example

The current code can be compiled with a standard `g++` toolchain and the existing project sources:

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp -O2 -o main
```

If you want a correctness-oriented build without extra optimization, you can also use:

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
```

## Notes

- This repository keeps the source code, scripts, and lightweight documentation for the current stage.
- Large datasets, generated outputs, caches, binaries, and intermediate experiment artifacts should not be committed.
