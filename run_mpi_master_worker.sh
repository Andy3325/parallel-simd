#!/bin/sh
set -e

mpic++ -O2 -std=c++17 correctness_guess_mpi_master_worker.cpp train.cpp guessing.cpp md5.cpp -o mpi_master_worker

mpiexec -n 2 ./mpi_master_worker 100000
mpiexec -n 4 ./mpi_master_worker 100000
mpiexec -n 8 ./mpi_master_worker 100000
