#!/bin/sh
set -e

mpic++ -O2 -std=c++17 -fopenmp correctness_guess_mpi_hybrid_openmp.cpp train.cpp guessing.cpp md5.cpp -o mpi_hybrid_openmp

mpiexec -n 1 ./mpi_hybrid_openmp 8
mpiexec -n 2 ./mpi_hybrid_openmp 4
mpiexec -n 4 ./mpi_hybrid_openmp 2
mpiexec -n 8 ./mpi_hybrid_openmp 1
