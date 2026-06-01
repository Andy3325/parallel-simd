#include "PCFG.h"
#include "md5.h"

#include <mpi.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>

using namespace std;
using namespace chrono;

static double SecondsSince(const steady_clock::time_point& start,
                           const steady_clock::time_point& end)
{
    return duration<double>(end - start).count();
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto total_start = steady_clock::now();

    double local_hash_time = 0.0;
    double local_generate_time = 0.0;
    unsigned long long local_guesses = 0;
    unsigned long long local_cracked = 0;

    PriorityQueue q;

    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();

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

    q.init();

    const auto search_start = steady_clock::now();
    int curr_num = 0;
    unsigned long long history = 0;
    const int generate_n = 10000000;

    while (!q.priority.empty())
    {
        const auto generate_start = steady_clock::now();
        q.PopNext();
        const auto generate_end = steady_clock::now();
        local_generate_time += SecondsSince(generate_start, generate_end);

        q.total_guesses = static_cast<int>(q.guesses.size());

        if (q.total_guesses - curr_num >= 100000)
        {
            curr_num = q.total_guesses;

            if (history + static_cast<unsigned long long>(q.total_guesses) >
                static_cast<unsigned long long>(generate_n))
            {
                break;
            }
        }

        if (curr_num > 1000000)
        {
            const auto hash_start = steady_clock::now();

            bit32 state[4];
            for (size_t i = 0; i < q.guesses.size(); ++i)
            {
                const unsigned long long global_idx =
                    history + static_cast<unsigned long long>(i);
                if (global_idx % static_cast<unsigned long long>(size) !=
                    static_cast<unsigned long long>(rank))
                {
                    continue;
                }

                const string& candidate = q.guesses[i];
                local_guesses += 1;
                if (test_set.find(candidate) != test_set.end())
                {
                    local_cracked += 1;
                }

                MD5Hash(candidate, state);
            }

            const auto hash_end = steady_clock::now();
            local_hash_time += SecondsSince(hash_start, hash_end);

            history += static_cast<unsigned long long>(curr_num);
            curr_num = 0;
            q.guesses.clear();
        }
    }

    const auto total_end = steady_clock::now();
    const double local_total_time = SecondsSince(total_start, total_end);
    const double local_search_time = SecondsSince(search_start, total_end);
    const double local_guess_time = local_search_time - local_hash_time;

    unsigned long long total_guesses = 0;
    unsigned long long total_cracked = 0;
    double max_total_time = 0.0;
    double max_hash_time = 0.0;
    double max_generate_time = 0.0;
    double max_guess_time = 0.0;

    MPI_Reduce(&local_guesses, &total_guesses, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cracked, &total_cracked, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_time, &max_hash_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_generate_time, &max_generate_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_guess_time, &max_guess_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << fixed << setprecision(6);
        cout << "MPI mode: static_cyclic" << endl;
        cout << "MPI size: " << size << endl;
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
