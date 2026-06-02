#include "PCFG.h"
#include "md5.h"

#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace chrono;

static const unsigned long long GENERATE_LIMIT = 10000000ULL;

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

static int ParseOpenMPThreadCount(int argc, char** argv)
{
    if (argc < 2)
    {
        return 1;
    }

    char* end = nullptr;
    long parsed = strtol(argv[1], &end, 10);
    if (end == argv[1] || parsed < 1)
    {
        return 1;
    }

    return static_cast<int>(parsed);
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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const int omp_threads = ParseOpenMPThreadCount(argc, argv);
    omp_set_num_threads(omp_threads);

    const auto total_start = steady_clock::now();

    double local_generate_time = 0.0;
    double local_hash_time = 0.0;
    unsigned long long local_guesses = 0;
    unsigned long long local_cracked = 0;

    NullBuffer null_buffer;
    streambuf* original_cout = cout.rdbuf(&null_buffer);

    PriorityQueue q;
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    unordered_set<string> test_set = LoadTestSet();

    cout.rdbuf(original_cout);

    q.init();

    int curr_num = 0;
    unsigned long long history = 0;
    const auto search_start = steady_clock::now();

    while (!q.priority.empty())
    {
        const auto generate_start = steady_clock::now();
        q.PopNext();
        q.total_guesses = static_cast<int>(q.guesses.size());

        if (q.total_guesses - curr_num >= 100000)
        {
            curr_num = q.total_guesses;

            if (history + static_cast<unsigned long long>(q.total_guesses) >
                GENERATE_LIMIT)
            {
                const auto generate_end = steady_clock::now();
                local_generate_time += SecondsSince(generate_start, generate_end);
                break;
            }
        }

        if (curr_num > 1000000)
        {
            vector<string> owned_guesses;
            owned_guesses.reserve(
                q.guesses.size() / static_cast<size_t>(mpi_size) + 1);

            for (size_t i = 0; i < q.guesses.size(); ++i)
            {
                const unsigned long long global_idx =
                    history + static_cast<unsigned long long>(i);
                if (global_idx % static_cast<unsigned long long>(mpi_size) ==
                    static_cast<unsigned long long>(rank))
                {
                    owned_guesses.emplace_back(q.guesses[i]);
                }
            }

            const auto generate_end = steady_clock::now();
            local_generate_time += SecondsSince(generate_start, generate_end);

            const auto hash_start = steady_clock::now();

            unsigned long long block_cracked = 0;
            const long long owned_count =
                static_cast<long long>(owned_guesses.size());
            const unordered_set<string>& readonly_test_set = test_set;

#pragma omp parallel for schedule(static) reduction(+:block_cracked)
            for (long long i = 0; i < owned_count; ++i)
            {
                bit32 state[4];
                const string& candidate =
                    owned_guesses[static_cast<size_t>(i)];
                if (readonly_test_set.find(candidate) != readonly_test_set.end())
                {
                    block_cracked += 1;
                }

                MD5Hash(candidate, state);
            }

            const auto hash_end = steady_clock::now();
            local_hash_time += SecondsSince(hash_start, hash_end);

            local_guesses += static_cast<unsigned long long>(owned_guesses.size());
            local_cracked += block_cracked;

            history += static_cast<unsigned long long>(curr_num);
            curr_num = 0;
            q.guesses.clear();
        }
        else
        {
            const auto generate_end = steady_clock::now();
            local_generate_time += SecondsSince(generate_start, generate_end);
        }
    }

    const auto total_end = steady_clock::now();
    const double local_total_time = SecondsSince(total_start, total_end);
    const double local_guess_time =
        SecondsSince(search_start, total_end);

    unsigned long long total_guesses = 0;
    unsigned long long total_cracked = 0;
    double max_generate_time = 0.0;
    double max_hash_time = 0.0;
    double max_guess_time = 0.0;
    double max_total_time = 0.0;

    MPI_Reduce(&local_guesses, &total_guesses, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cracked, &total_cracked, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_generate_time, &max_generate_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_time, &max_hash_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_guess_time, &max_guess_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << fixed << setprecision(6);
        cout << "MPI mode: hybrid_openmp_static_cyclic" << endl;
        cout << "MPI size: " << mpi_size << endl;
        cout << "OpenMP threads per rank: " << omp_threads << endl;
        cout << "Total parallelism: " << mpi_size * omp_threads << endl;
        cout << "Total guesses: " << total_guesses << endl;
        cout << "Total cracked: " << total_cracked << endl;
        cout << "Max guess time: " << max_guess_time << endl;
        cout << "Max hash time: " << max_hash_time << endl;
        cout << "Max generate time: " << max_generate_time << endl;
        cout << "Total wall time: " << max_total_time << endl;
    }

    MPI_Finalize();
    return 0;
}
