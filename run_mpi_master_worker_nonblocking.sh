#!/bin/sh
set -e

mpic++ -O2 -std=c++17 correctness_guess_mpi_master_worker_nonblocking.cpp train.cpp guessing.cpp md5.cpp -o mpi_master_worker_nonblocking

mpiexec -n 2 ./mpi_master_worker_nonblocking 64
mpiexec -n 8 ./mpi_master_worker_nonblocking 8
mpiexec -n 8 ./mpi_master_worker_nonblocking 64
