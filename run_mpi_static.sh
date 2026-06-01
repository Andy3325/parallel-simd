#!/bin/sh
set -e

mpicxx -O2 -std=c++17 correctness_guess_mpi_static.cpp train.cpp guessing.cpp md5.cpp -o mpi_static

mpiexec -n 1 ./mpi_static
mpiexec -n 2 ./mpi_static
mpiexec -n 4 ./mpi_static
mpiexec -n 8 ./mpi_static
