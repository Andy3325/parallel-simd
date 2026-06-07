#ifndef TRAIN_MPI_H
#define TRAIN_MPI_H

#include <mpi.h>
#include <string>

class model;

void train_mpi(model& m, const std::string& path, int rank, int world_size);

#endif
