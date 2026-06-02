#include "PCFG.h"
#include "md5.h"

#include <mpi.h>

#include <algorithm>
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

static const int REQUEST_TAG = 100;
static const int TASK_TAG = 101;
static const int RESULT_TAG = 102;
static const int STOP_TAG = 103;

static const unsigned long long GENERATE_LIMIT = 10000000ULL;
static const unsigned long long DEFAULT_PT_BLOCK_SIZE = 64ULL;

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

static unsigned long long ParsePTBlockSize(int argc, char** argv)
{
    if (argc < 2)
    {
        return DEFAULT_PT_BLOCK_SIZE;
    }

    char* end = nullptr;
    unsigned long long parsed = strtoull(argv[1], &end, 10);
    if (end == argv[1] || parsed == 0)
    {
        return DEFAULT_PT_BLOCK_SIZE;
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

static void InsertPTSorted(PriorityQueue& q, const PT& pt)
{
    if (q.priority.empty())
    {
        q.priority.emplace_back(pt);
        return;
    }

    for (auto iter = q.priority.begin(); iter != q.priority.end(); ++iter)
    {
        if (iter != q.priority.end() - 1 && iter != q.priority.begin())
        {
            if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
            {
                q.priority.emplace(iter + 1, pt);
                return;
            }
        }
        if (iter == q.priority.end() - 1)
        {
            q.priority.emplace_back(pt);
            return;
        }
        if (iter == q.priority.begin() && iter->prob < pt.prob)
        {
            q.priority.emplace(iter, pt);
            return;
        }
    }
}

static vector<PT> BuildHashablePTTasks(const model& trained_model)
{
    PriorityQueue q;
    q.m = trained_model;
    q.init();

    vector<PT> tasks;
    vector<PT> pending_tasks;
    unsigned long long buffer_count = 0;
    unsigned long long curr_num = 0;
    unsigned long long history = 0;

    while (!q.priority.empty())
    {
        PT current = q.priority.front();
        const unsigned long long generated_count =
            static_cast<unsigned long long>(q.CountGeneratedGuesses(current));

        pending_tasks.emplace_back(current);
        buffer_count += generated_count;

        vector<PT> new_pts = current.NewPTs();
        for (PT pt : new_pts)
        {
            q.CalProb(pt);
            InsertPTSorted(q, pt);
        }
        q.priority.erase(q.priority.begin());

        if (buffer_count - curr_num >= 100000)
        {
            curr_num = buffer_count;

            if (history + buffer_count > GENERATE_LIMIT)
            {
                break;
            }
        }

        if (curr_num > 1000000)
        {
            tasks.insert(tasks.end(), pending_tasks.begin(), pending_tasks.end());
            history += curr_num;
            curr_num = 0;
            buffer_count = 0;
            pending_tasks.clear();
        }
    }

    return tasks;
}

static WorkerResult ProcessPTBlock(const model& trained_model,
                                   const unordered_set<string>& test_set,
                                   const vector<PT>& task_list,
                                   unsigned long long pt_start,
                                   unsigned long long pt_end)
{
    WorkerResult result = {0, 0, 0.0, 0.0, 0.0};
    const auto worker_start = steady_clock::now();

    PriorityQueue q;
    q.m = trained_model;

    bit32 state[4];
    const unsigned long long safe_end =
        min(pt_end, static_cast<unsigned long long>(task_list.size()));

    for (unsigned long long task_idx = pt_start; task_idx < safe_end; ++task_idx)
    {
        const auto generate_start = steady_clock::now();
        vector<string> guesses;
        q.GenerateToVector(task_list[static_cast<size_t>(task_idx)], guesses);
        const auto generate_end = steady_clock::now();
        result.generate_time += SecondsSince(generate_start, generate_end);

        const auto hash_start = steady_clock::now();

        for (const string& candidate : guesses)
        {
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

    const auto worker_end = steady_clock::now();
    result.worker_time = SecondsSince(worker_start, worker_end);
    return result;
}

static void RunMaster(const model& trained_model,
                      int size,
                      unsigned long long pt_block_size)
{
    const auto wall_start = steady_clock::now();

    vector<PT> task_list = BuildHashablePTTasks(trained_model);
    const int worker_count = size - 1;
    const unsigned long long total_tasks =
        (static_cast<unsigned long long>(task_list.size()) + pt_block_size - 1) /
        pt_block_size;

    unsigned long long next_pt_start = 0;
    int stopped_workers = 0;

    unsigned long long total_guesses = 0;
    unsigned long long total_cracked = 0;
    vector<double> worker_time_by_rank(size, 0.0);
    vector<double> hash_time_by_rank(size, 0.0);
    vector<double> generate_time_by_rank(size, 0.0);

    vector<int> request_tokens(worker_count, 0);
    vector<WorkerResult> result_buffers(worker_count);
    vector<MPI_Request> event_requests(worker_count * 2, MPI_REQUEST_NULL);
    vector<bool> stopped(worker_count, false);
    vector<bool> task_in_flight(worker_count, false);

    auto request_index = [](int worker_idx) { return worker_idx; };
    auto result_index = [worker_count](int worker_idx)
    {
        return worker_count + worker_idx;
    };
    auto worker_rank = [](int worker_idx) { return worker_idx + 1; };

    auto post_request_recv = [&](int worker_idx)
    {
        MPI_Irecv(&request_tokens[worker_idx], 1, MPI_INT, worker_rank(worker_idx),
                  REQUEST_TAG, MPI_COMM_WORLD,
                  &event_requests[request_index(worker_idx)]);
    };

    auto post_result_recv = [&](int worker_idx)
    {
        MPI_Irecv(&result_buffers[worker_idx],
                  static_cast<int>(sizeof(WorkerResult)), MPI_BYTE,
                  worker_rank(worker_idx), RESULT_TAG, MPI_COMM_WORLD,
                  &event_requests[result_index(worker_idx)]);
    };

    auto aggregate_result = [&](int worker_idx)
    {
        const WorkerResult& result = result_buffers[worker_idx];
        const int rank = worker_rank(worker_idx);
        total_guesses += result.guesses;
        total_cracked += result.cracked;
        worker_time_by_rank[rank] += result.worker_time;
        hash_time_by_rank[rank] += result.hash_time;
        generate_time_by_rank[rank] += result.generate_time;
        task_in_flight[worker_idx] = false;
    };

    for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx)
    {
        post_request_recv(worker_idx);
        post_result_recv(worker_idx);
    }

    while (stopped_workers < worker_count)
    {
        int completed_index = MPI_UNDEFINED;
        int flag = 0;
        MPI_Status status;
        MPI_Testany(static_cast<int>(event_requests.size()),
                    event_requests.data(), &completed_index, &flag, &status);

        if (!flag || completed_index == MPI_UNDEFINED)
        {
            continue;
        }

        const bool is_request = completed_index < worker_count;
        const int worker_idx = is_request
            ? completed_index
            : completed_index - worker_count;

        if (is_request)
        {
            if (task_in_flight[worker_idx])
            {
                MPI_Wait(&event_requests[result_index(worker_idx)],
                         MPI_STATUS_IGNORE);
                aggregate_result(worker_idx);
                if (!stopped[worker_idx])
                {
                    post_result_recv(worker_idx);
                }
            }

            if (next_pt_start < static_cast<unsigned long long>(task_list.size()))
            {
                unsigned long long task[2];
                task[0] = next_pt_start;
                task[1] = min(static_cast<unsigned long long>(task_list.size()),
                              next_pt_start + pt_block_size);
                next_pt_start = task[1];

                MPI_Request send_request = MPI_REQUEST_NULL;
                MPI_Isend(task, 2, MPI_UNSIGNED_LONG_LONG,
                          worker_rank(worker_idx), TASK_TAG, MPI_COMM_WORLD,
                          &send_request);
                MPI_Wait(&send_request, MPI_STATUS_IGNORE);

                task_in_flight[worker_idx] = true;
                post_request_recv(worker_idx);
            }
            else
            {
                MPI_Request send_request = MPI_REQUEST_NULL;
                MPI_Isend(nullptr, 0, MPI_BYTE, worker_rank(worker_idx),
                          STOP_TAG, MPI_COMM_WORLD, &send_request);
                MPI_Wait(&send_request, MPI_STATUS_IGNORE);

                stopped[worker_idx] = true;
                stopped_workers += 1;

                if (event_requests[result_index(worker_idx)] != MPI_REQUEST_NULL)
                {
                    MPI_Cancel(&event_requests[result_index(worker_idx)]);
                    MPI_Wait(&event_requests[result_index(worker_idx)],
                             MPI_STATUS_IGNORE);
                }
            }
        }
        else
        {
            aggregate_result(worker_idx);
            if (!stopped[worker_idx])
            {
                post_result_recv(worker_idx);
            }
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
    cout << "MPI mode: master_worker_nonblocking" << endl;
    cout << "MPI size: " << size << endl;
    cout << "Block size: " << pt_block_size << endl;
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
    vector<PT> task_list = BuildHashablePTTasks(trained_model);

    cout.rdbuf(original_cout);

    while (true)
    {
        int request_token = 1;
        MPI_Request request_send = MPI_REQUEST_NULL;
        MPI_Isend(&request_token, 1, MPI_INT, 0, REQUEST_TAG, MPI_COMM_WORLD,
                  &request_send);
        MPI_Wait(&request_send, MPI_STATUS_IGNORE);

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
            ProcessPTBlock(trained_model, test_set, task_list, task[0], task[1]);
        cout.rdbuf(original_cout);

        MPI_Request result_send = MPI_REQUEST_NULL;
        MPI_Isend(&result, static_cast<int>(sizeof(result)), MPI_BYTE, 0,
                  RESULT_TAG, MPI_COMM_WORLD, &result_send);
        MPI_Wait(&result_send, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const unsigned long long pt_block_size = ParsePTBlockSize(argc, argv);

    if (size < 2)
    {
        if (rank == 0)
        {
            cerr << "mpi_master_worker_nonblocking requires at least 2 MPI processes"
                 << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        NullBuffer null_buffer;
        streambuf* original_cout = cout.rdbuf(&null_buffer);

        model trained_model;
        trained_model.train("/guessdata/Rockyou-singleLined-full.txt");
        trained_model.order();

        cout.rdbuf(original_cout);

        RunMaster(trained_model, size, pt_block_size);
    }
    else
    {
        RunWorker();
    }

    MPI_Finalize();
    return 0;
}
