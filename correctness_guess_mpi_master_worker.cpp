#include "PCFG.h"
#include "md5.h"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace chrono;

static const int REQUEST_TAG = 100;
static const int TASK_TAG = 101;
static const int RESULT_TAG = 102;
static const int STOP_TAG = 103;

static const unsigned long long GENERATE_LIMIT = 10000000ULL;
static const unsigned long long DEFAULT_BLOCK_SIZE = 100000ULL;

struct WorkerResult
{
    unsigned long long guesses;
    unsigned long long cracked;
    double worker_time;
    double hash_time;
    double generate_time;
};

class NullBuffer : public std::streambuf
{
protected:
    int overflow(int ch) override
    {
        return ch;
    }
};

static double SecondsSince(const steady_clock::time_point& start,
                           const steady_clock::time_point& end)
{
    return duration<double>(end - start).count();
}

static unsigned long long ParseBlockSize(int argc, char** argv)
{
    if (argc < 2)
    {
        return DEFAULT_BLOCK_SIZE;
    }

    char* end = nullptr;
    unsigned long long parsed = strtoull(argv[1], &end, 10);
    if (end == argv[1] || parsed == 0)
    {
        return DEFAULT_BLOCK_SIZE;
    }
    return parsed;
}

static unordered_set<string> LoadTestSet()
{
    unordered_set<string> test_set;
    test_set.reserve(2000000);
    test_set.max_load_factor(0.5);

    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while (test_data >> pw)
    {
        test_count += 1;
        test_set.insert(pw);
        if (test_count >= 1000000)
        {
            break;
        }
    }

    return test_set;
}

static WorkerResult ProcessBlock(const model& trained_model,
                                 const unordered_set<string>& test_set,
                                 unsigned long long block_start,
                                 unsigned long long block_end)
{
    WorkerResult result = {0, 0, 0.0, 0.0, 0.0};
    const auto worker_start = steady_clock::now();

    PriorityQueue q;
    q.m = trained_model;
    q.init();

    int curr_num = 0;
    unsigned long long history = 0;

    while (!q.priority.empty())
    {
        const auto generate_start = steady_clock::now();
        q.PopNext();
        const auto generate_end = steady_clock::now();
        result.generate_time += SecondsSince(generate_start, generate_end);

        q.total_guesses = static_cast<int>(q.guesses.size());

        if (q.total_guesses - curr_num >= 100000)
        {
            curr_num = q.total_guesses;

            if (history + static_cast<unsigned long long>(q.total_guesses) >
                GENERATE_LIMIT)
            {
                break;
            }
        }

        if (curr_num > 1000000)
        {
            const unsigned long long buffer_start = history;
            const unsigned long long buffer_end =
                history + static_cast<unsigned long long>(q.guesses.size());
            const unsigned long long overlap_start =
                max(block_start, buffer_start);
            const unsigned long long overlap_end =
                min(block_end, buffer_end);

            if (overlap_start < overlap_end)
            {
                const auto hash_start = steady_clock::now();

                bit32 state[4];
                for (unsigned long long global_idx = overlap_start;
                     global_idx < overlap_end; ++global_idx)
                {
                    const size_t local_idx =
                        static_cast<size_t>(global_idx - buffer_start);
                    const string& candidate = q.guesses[local_idx];

                    result.guesses += 1;
                    if (test_set.find(candidate) != test_set.end())
                    {
                        result.cracked += 1;
                    }

                    MD5Hash(candidate, state);
                }

                const auto hash_end = steady_clock::now();
                result.hash_time += SecondsSince(hash_start, hash_end);
            }

            history += static_cast<unsigned long long>(curr_num);
            curr_num = 0;
            q.guesses.clear();

            if (history >= block_end)
            {
                break;
            }
        }
    }

    const auto worker_end = steady_clock::now();
    result.worker_time = SecondsSince(worker_start, worker_end);
    return result;
}

static void RunMaster(int size, unsigned long long block_size)
{
    const auto wall_start = steady_clock::now();

    const int worker_count = size - 1;
    const unsigned long long total_tasks =
        (GENERATE_LIMIT + block_size - 1) / block_size;

    unsigned long long next_task_start = 0;
    int stopped_workers = 0;

    unsigned long long total_guesses = 0;
    unsigned long long total_cracked = 0;
    vector<double> worker_time_by_rank(size, 0.0);
    vector<double> hash_time_by_rank(size, 0.0);
    vector<double> generate_time_by_rank(size, 0.0);

    while (stopped_workers < worker_count)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        const int source = status.MPI_SOURCE;
        const int tag = status.MPI_TAG;

        if (tag == REQUEST_TAG)
        {
            MPI_Recv(nullptr, 0, MPI_BYTE, source, REQUEST_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (next_task_start < GENERATE_LIMIT)
            {
                unsigned long long task[2];
                task[0] = next_task_start;
                task[1] = min(GENERATE_LIMIT, next_task_start + block_size);
                next_task_start = task[1];

                MPI_Send(task, 2, MPI_UNSIGNED_LONG_LONG, source, TASK_TAG,
                         MPI_COMM_WORLD);
            }
            else
            {
                MPI_Send(nullptr, 0, MPI_BYTE, source, STOP_TAG,
                         MPI_COMM_WORLD);
                stopped_workers += 1;
            }
        }
        else if (tag == RESULT_TAG)
        {
            WorkerResult result;
            MPI_Recv(&result, static_cast<int>(sizeof(result)), MPI_BYTE,
                     source, RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            total_guesses += result.guesses;
            total_cracked += result.cracked;
            worker_time_by_rank[source] += result.worker_time;
            hash_time_by_rank[source] += result.hash_time;
            generate_time_by_rank[source] += result.generate_time;
        }
    }

    const double max_worker_time =
        *max_element(worker_time_by_rank.begin(), worker_time_by_rank.end());
    const double max_hash_time =
        *max_element(hash_time_by_rank.begin(), hash_time_by_rank.end());
    const double max_generate_time =
        *max_element(generate_time_by_rank.begin(), generate_time_by_rank.end());
    const double total_wall_time = SecondsSince(wall_start, steady_clock::now());

    cout << fixed << setprecision(6);
    cout << "MPI mode: master_worker_blocking" << endl;
    cout << "MPI size: " << size << endl;
    cout << "Block size: " << block_size << endl;
    cout << "Total guesses: " << total_guesses << endl;
    cout << "Total cracked: " << total_cracked << endl;
    cout << "Max worker time: " << max_worker_time << endl;
    cout << "Max hash time: " << max_hash_time << endl;
    cout << "Max generate time: " << max_generate_time << endl;
    cout << "Total wall time: " << total_wall_time << endl;
    cout << "Total tasks: " << total_tasks << endl;
}

static void RunWorker()
{
    NullBuffer null_buffer;
    streambuf* original_cout = cout.rdbuf(&null_buffer);

    model trained_model;
    trained_model.train("/guessdata/Rockyou-singleLined-full.txt");
    trained_model.order();
    unordered_set<string> test_set = LoadTestSet();

    cout.rdbuf(original_cout);

    while (true)
    {
        MPI_Send(nullptr, 0, MPI_BYTE, 0, REQUEST_TAG, MPI_COMM_WORLD);

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == STOP_TAG)
        {
            MPI_Recv(nullptr, 0, MPI_BYTE, 0, STOP_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            break;
        }

        unsigned long long task[2] = {0, 0};
        MPI_Recv(task, 2, MPI_UNSIGNED_LONG_LONG, 0, TASK_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout.rdbuf(&null_buffer);
        WorkerResult result =
            ProcessBlock(trained_model, test_set, task[0], task[1]);
        cout.rdbuf(original_cout);

        MPI_Send(&result, static_cast<int>(sizeof(result)), MPI_BYTE, 0,
                 RESULT_TAG, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const unsigned long long block_size = ParseBlockSize(argc, argv);

    if (size < 2)
    {
        if (rank == 0)
        {
            cerr << "mpi_master_worker requires at least 2 MPI processes"
                 << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        RunMaster(size, block_size);
    }
    else
    {
        RunWorker();
    }

    MPI_Finalize();
    return 0;
}
