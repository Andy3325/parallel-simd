#ifndef GUESSING_MPI_H
#define GUESSING_MPI_H

#include <mpi.h>
#include <string>
#include <unordered_set>
#include <vector>

class model;
class PT;
class PriorityQueue;

struct MPILocalResult
{
    unsigned long long generated = 0;
    unsigned long long cracked = 0;
    double generate_time = 0.0;
    double hash_time = 0.0;
    double compute_time = 0.0;
};

void BcastPT(PT& pt, int rank);
void BcastPTBatch(std::vector<PT>& pts, int rank);
void AdvanceFrontWithoutGenerate(PriorityQueue& q);

MPILocalResult GenerateAndHashPTMPI(model& m,
                                    const PT& pt,
                                    const std::unordered_set<std::string>& test_set,
                                    int rank,
                                    int world_size);

#endif
