#include "PCFG.h"
#include "guessing_mpi.h"
#include "train_mpi.h"

#include <mpi.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

namespace
{
static const unsigned long long GENERATE_LIMIT = 10000000ULL;

struct GlobalStats
{
    unsigned long long total_guesses = 0;
    unsigned long long total_cracked = 0;
    unsigned long long total_pt_tasks = 0;
    unsigned long long total_batch_rounds = 0;
    double local_generate_time = 0.0;
    double local_hash_time = 0.0;
};

enum RunMode
{
    MODE_BASIC,
    MODE_BATCH
};

struct RunConfig
{
    RunMode mode = MODE_BASIC;
    int batch_size = 1;
};

static RunConfig ParseArgs(int argc, char** argv)
{
    RunConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            const char* mode = argv[++i];
            if (strcmp(mode, "batch") == 0)
            {
                cfg.mode = MODE_BATCH;
            }
            else
            {
                cfg.mode = MODE_BASIC;
            }
        }
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
        {
            cfg.batch_size = atoi(argv[++i]);
        }
    }

    if (cfg.batch_size < 1)
    {
        cfg.batch_size = 1;
    }
    return cfg;
}

static unordered_set<string> LoadTestSet()
{
    unordered_set<string> test_set;
    test_set.reserve(2000000);
    test_set.max_load_factor(0.5);

    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    string pw;
    int test_count = 0;
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

static void ReduceRoundStats(const MPILocalResult& local,
                             unsigned long long& round_generated,
                             unsigned long long& round_cracked,
                             double& round_generate_time,
                             double& round_hash_time,
                             double& round_compute_time)
{
    unsigned long long local_generated = local.generated;
    unsigned long long local_cracked = local.cracked;
    double local_generate = local.generate_time;
    double local_hash = local.hash_time;
    double local_compute = local.compute_time;

    MPI_Reduce(&local_generated, &round_generated, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cracked, &round_cracked, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_generate, &round_generate_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash, &round_hash_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute, &round_compute_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

static GlobalStats RunBasicPTBroadcast(PriorityQueue& q,
                                       int rank,
                                       int world_size,
                                       const unordered_set<string>& test_set)
{
    GlobalStats stats;

    unsigned long long history = 0;
    unsigned long long buffered_generated = 0;
    unsigned long long curr_num = 0;
    unsigned long long pending_guesses = 0;
    unsigned long long pending_cracked = 0;

    while (true)
    {
        int active = 0;
        PT current_pt;

        if (rank == 0)
        {
            if (!q.priority.empty())
            {
                current_pt = q.priority.front();
                const unsigned long long generated_count =
                    static_cast<unsigned long long>(
                        q.CountGeneratedGuesses(current_pt));
                const unsigned long long preview_buffered =
                    buffered_generated + generated_count;

                unsigned long long next_curr_num = curr_num;
                if (preview_buffered - curr_num >= 100000ULL)
                {
                    next_curr_num = preview_buffered;
                    if (history + preview_buffered > GENERATE_LIMIT)
                    {
                        active = 0;
                    }
                    else
                    {
                        active = 1;
                    }
                }
                else
                {
                    active = 1;
                }

                if (active)
                {
                    buffered_generated = preview_buffered;
                    curr_num = next_curr_num;
                }
            }
        }

        MPI_Bcast(&active, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!active)
        {
            break;
        }

        BcastPT(current_pt, rank);

        MPILocalResult local_result =
            GenerateAndHashPTMPI(q.m, current_pt, test_set, rank, world_size);
        stats.local_generate_time += local_result.generate_time;
        stats.local_hash_time += local_result.hash_time;

        unsigned long long round_generated = 0;
        unsigned long long round_cracked = 0;
        double round_generate_time = 0.0;
        double round_hash_time = 0.0;
        double round_compute_time = 0.0;
        ReduceRoundStats(local_result,
                         round_generated,
                         round_cracked,
                         round_generate_time,
                         round_hash_time,
                         round_compute_time);

        if (rank == 0)
        {
            pending_guesses += round_generated;
            pending_cracked += round_cracked;
            stats.total_pt_tasks += 1;

            AdvanceFrontWithoutGenerate(q);

            if (curr_num > 1000000ULL)
            {
                stats.total_guesses += pending_guesses;
                stats.total_cracked += pending_cracked;
                history += curr_num;
                curr_num = 0;
                buffered_generated = 0;
                pending_guesses = 0;
                pending_cracked = 0;
            }
        }
    }

    return stats;
}

static GlobalStats RunBatchPTBroadcast(PriorityQueue& q,
                                       int rank,
                                       int world_size,
                                       int batch_size,
                                       const unordered_set<string>& test_set)
{
    GlobalStats stats;

    unsigned long long history = 0;
    unsigned long long buffered_generated = 0;
    unsigned long long curr_num = 0;
    unsigned long long pending_guesses = 0;
    unsigned long long pending_cracked = 0;

    while (true)
    {
        int active = 0;
        vector<PT> batch_pts;
        bool should_flush_after_batch = false;

        if (rank == 0)
        {
            while (static_cast<int>(batch_pts.size()) < batch_size &&
                   !q.priority.empty())
            {
                PT current_pt = q.priority.front();
                const unsigned long long generated_count =
                    static_cast<unsigned long long>(
                        q.CountGeneratedGuesses(current_pt));
                const unsigned long long preview_buffered =
                    buffered_generated + generated_count;

                unsigned long long next_curr_num = curr_num;
                if (preview_buffered - curr_num >= 100000ULL)
                {
                    next_curr_num = preview_buffered;
                    if (history + preview_buffered > GENERATE_LIMIT)
                    {
                        break;
                    }
                }

                batch_pts.emplace_back(current_pt);
                buffered_generated = preview_buffered;
                curr_num = next_curr_num;

                AdvanceFrontWithoutGenerate(q);

                if (curr_num > 1000000ULL)
                {
                    should_flush_after_batch = true;
                    break;
                }
            }

            active = batch_pts.empty() ? 0 : 1;
        }

        MPI_Bcast(&active, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!active)
        {
            break;
        }

        vector<PT> local_batch = batch_pts;
        BcastPTBatch(local_batch, rank);

        MPILocalResult local_batch_result;
        for (const PT& pt : local_batch)
        {
            MPILocalResult local_result =
                GenerateAndHashPTMPI(q.m, pt, test_set, rank, world_size);
            local_batch_result.generated += local_result.generated;
            local_batch_result.cracked += local_result.cracked;
            local_batch_result.generate_time += local_result.generate_time;
            local_batch_result.hash_time += local_result.hash_time;
            local_batch_result.compute_time += local_result.compute_time;
        }

        stats.local_generate_time += local_batch_result.generate_time;
        stats.local_hash_time += local_batch_result.hash_time;

        unsigned long long round_generated = 0;
        unsigned long long round_cracked = 0;
        double round_generate_time = 0.0;
        double round_hash_time = 0.0;
        double round_compute_time = 0.0;
        ReduceRoundStats(local_batch_result,
                         round_generated,
                         round_cracked,
                         round_generate_time,
                         round_hash_time,
                         round_compute_time);

        if (rank == 0)
        {
            pending_guesses += round_generated;
            pending_cracked += round_cracked;
            stats.total_pt_tasks +=
                static_cast<unsigned long long>(batch_pts.size());
            stats.total_batch_rounds += 1;

            if (should_flush_after_batch)
            {
                stats.total_guesses += pending_guesses;
                stats.total_cracked += pending_cracked;
                history += curr_num;
                curr_num = 0;
                buffered_generated = 0;
                pending_guesses = 0;
                pending_cracked = 0;
            }
        }
    }

    return stats;
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const RunConfig cfg = ParseArgs(argc, argv);

    if (rank != 0)
    {
        cout.setstate(ios_base::failbit);
    }

    const auto wall_start = chrono::steady_clock::now();
    PriorityQueue q;

    const double train_start = MPI_Wtime();
    train_mpi(q.m, "/guessdata/Rockyou-singleLined-full.txt", rank, world_size);
    const double train_end = MPI_Wtime();

    double local_train_time = train_end - train_start;
    double train_time = 0.0;
    MPI_Reduce(&local_train_time, &train_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unordered_set<string> test_set = LoadTestSet();

    if (rank == 0)
    {
        cout.clear();
        q.init();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double guess_start = MPI_Wtime();
    GlobalStats stats;
    if (cfg.mode == MODE_BATCH)
    {
        stats = RunBatchPTBroadcast(q, rank, world_size, cfg.batch_size,
                                    test_set);
    }
    else
    {
        stats = RunBasicPTBroadcast(q, rank, world_size, test_set);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double guess_end = MPI_Wtime();

    double local_guess_hash_time = guess_end - guess_start;
    double local_generate_time = stats.local_generate_time;
    double local_hash_time = stats.local_hash_time;
    double local_compute_time = local_generate_time + local_hash_time;

    double max_guess_hash_time = 0.0;
    double max_generate_time = 0.0;
    double max_hash_time = 0.0;
    double max_compute_time = 0.0;

    MPI_Reduce(&local_guess_hash_time, &max_guess_hash_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_generate_time, &max_generate_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_time, &max_hash_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_time, &max_compute_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const auto wall_end = chrono::steady_clock::now();
    const double total_wall_time =
        chrono::duration<double>(wall_end - wall_start).count();

    if (rank == 0)
    {
        double overhead = max_guess_hash_time - max_compute_time;
        if (overhead < 0.0)
        {
            overhead = 0.0;
        }

        double guess_time = max_guess_hash_time - max_hash_time;
        if (guess_time < 0.0)
        {
            guess_time = 0.0;
        }

        cout << fixed << setprecision(6);
        if (cfg.mode == MODE_BATCH)
        {
            cout << "MPI mode: batch_pt_range_distributed_training" << endl;
        }
        else
        {
            cout << "MPI mode: basic_pt_range_distributed_training" << endl;
        }
        cout << "MPI size: " << world_size << endl;
        if (cfg.mode == MODE_BATCH)
        {
            cout << "Batch size: " << cfg.batch_size << endl;
        }
        cout << "Generated: " << stats.total_guesses << endl;
        cout << "Cracked: " << stats.total_cracked << endl;
        cout << "Train time: " << train_time << endl;
        cout << "Guess time: " << guess_time << endl;
        cout << "Hash time: " << max_hash_time << endl;
        cout << "MPI total Guess+Hash time: " << max_guess_hash_time << endl;
        cout << "MPI generate only time: " << max_generate_time << endl;
        cout << "MPI compute time: " << max_compute_time << endl;
        cout << "MPI overhead/non-compute time: " << overhead << endl;
        cout << "Total wall time: " << total_wall_time << endl;
        cout << "Total PT tasks: " << stats.total_pt_tasks << endl;
        if (cfg.mode == MODE_BATCH)
        {
            cout << "Total batch rounds: " << stats.total_batch_rounds << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
