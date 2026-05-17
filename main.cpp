#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace chrono;

static inline size_t GetMD5BlockCount(const string& s)
{
    size_t total = s.size() + 1 + 8;
    if (total % 64 != 0)
    {
        total += 64 - (total % 64);
    }
    return total / 64;
}

static inline void FlushBucketRemainder(vector<string>& bucket)
{
    bit32 state[4];
    for (const string& pw : bucket)
    {
        MD5Hash(pw, state);
    }
    bucket.clear();
}

static bool FileExists(const string& path)
{
    ifstream file(path);
    return file.good();
}

static string ResolveTrainPath(const string& requested_path)
{
    if (!requested_path.empty())
    {
        return requested_path;
    }

    vector<string> candidates = {
        "guessdata/Rockyou-singleLined-full.txt",
        "../guessdata/Rockyou-singleLined-full.txt",
        "../../guessdata/Rockyou-singleLined-full.txt",
        "Rockyou-singleLined-full.txt",
        "/guessdata/Rockyou-singleLined-full.txt",
        "guessdata/benchmark-small.txt"
    };

    for (const string& path : candidates)
    {
        if (FileExists(path))
        {
            return path;
        }
    }

    return candidates.back();
}

static void HashAndClearGuesses(PriorityQueue& q, double& time_hash, size_t& total_one_block_batches, size_t& total_general_batches)
{
    if (q.guesses.empty())
    {
        return;
    }

    auto start_hash = system_clock::now();
    unordered_map<size_t, vector<string>> buckets;
    bit32 states4[4][4];

    for (const string& pw : q.guesses)
    {
        size_t blk = GetMD5BlockCount(pw);
        auto& bucket = buckets[blk];
        bucket.emplace_back(pw);

        if (bucket.size() == 4)
        {
            if (blk == 1)
            {
                ++total_one_block_batches;
                MD5Hash4_1Block(bucket.data(), states4);
            }
            else
            {
                ++total_general_batches;
                MD5Hash4(bucket.data(), states4);
            }

            bucket.clear();
        }
    }

    for (auto& kv : buckets)
    {
        FlushBucketRemainder(kv.second);
    }

    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

    q.guesses.clear();
    q.total_guesses = 0;
}

int main(int argc, char* argv[])
{
    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };

    for (int i = 0; i < 8; i++)
    {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1 += 1)
        {
            ss << std::setw(8) << std::setfill('0') << hex << state[i1];
        }
        if (ss.str() != test_hashes[i])
        {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }

    bit32 batch_states[4][4];
    MD5Hash4(test_pws, batch_states);
    for (int lane = 0; lane < 4; ++lane)
    {
        stringstream ss;
        for (int i1 = 0; i1 < 4; ++i1)
        {
            ss << std::setw(8) << std::setfill('0') << hex << batch_states[lane][i1];
        }
        if (ss.str() != test_hashes[lane])
        {
            cout << "MD5Hash4 test failed for lane " << lane << "!" << endl;
            cout << "Expected: " << test_hashes[lane] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }

    bit32 batch_states_1b[4][4];
    MD5Hash4_1Block(test_pws, batch_states_1b);
    for (int lane = 0; lane < 4; ++lane)
    {
        stringstream ss;
        for (int i1 = 0; i1 < 4; ++i1)
        {
            ss << std::setw(8) << std::setfill('0') << hex << batch_states_1b[lane][i1];
        }
        if (ss.str() != test_hashes[lane])
        {
            cout << "MD5Hash4_1Block test failed for lane " << lane << "!" << endl;
            cout << "Expected: " << test_hashes[lane] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }

    cout << "MD5Hash test passed!" << endl; // Please do not modify this line.

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    const int DEFAULT_BENCHMARK_ROUNDS = 1000;
    const size_t HASH_FLUSH_THRESHOLD = 1000000;
    int benchmark_rounds = DEFAULT_BENCHMARK_ROUNDS;
    string requested_train_path;

    for (int i = 1; i < argc; i += 1)
    {
        string arg = argv[i];
        if (arg == "--guess-rounds" && i + 1 < argc)
        {
            benchmark_rounds = atoi(argv[++i]);
            if (benchmark_rounds < 0)
            {
                benchmark_rounds = 0;
            }
        }
        else if (arg == "--train" && i + 1 < argc)
        {
            requested_train_path = argv[++i];
        }
    }

    string train_path = ResolveTrainPath(requested_train_path);
    if (!FileExists(train_path))
    {
        cout << "Training dataset not found: " << train_path << endl;
        cout << "Use --train <path> to specify an existing training set." << endl;
        return 1;
    }

    PriorityQueue q;

    auto start_train = system_clock::now();
    cout << "Training dataset: " << train_path << endl;
    cout << "Guess benchmark rounds: " << benchmark_rounds << endl;
    q.m.train(train_path);
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "Initial priority size: " << q.priority.size() << endl;

    size_t total_one_block_batches = 0;
    size_t total_general_batches = 0;
    long long benchmark_total_guesses = 0;

    auto start_guess = system_clock::now();
    for (int round = 0; round < benchmark_rounds && !q.priority.empty(); round += 1)
    {
        size_t before = q.guesses.size();
        q.PopNext();
        q.total_guesses = q.guesses.size();
        benchmark_total_guesses += static_cast<long long>(q.guesses.size() - before);

        if (benchmark_total_guesses > 0 && benchmark_total_guesses % 100000 == 0)
        {
            cout << "Guesses generated: " << benchmark_total_guesses << endl;
        }

        if (q.guesses.size() >= HASH_FLUSH_THRESHOLD)
        {
            HashAndClearGuesses(q, time_hash, total_one_block_batches, total_general_batches);
        }
    }
    auto end_guess = system_clock::now();
    auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
    time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;

    HashAndClearGuesses(q, time_hash, total_one_block_batches, total_general_batches);

    cout << "one_block_batches=" << total_one_block_batches << endl;
    cout << "general_batches=" << total_general_batches << endl;
    cout << "total_guesses = " << benchmark_total_guesses << endl;
    cout << "Guess benchmark time:" << time_guess - time_hash << "seconds" << endl;
    cout << "Hash time:" << time_hash << "seconds" << endl;
    cout << "Train time:" << time_train << "seconds" << endl;
    q.PrintGenerateStats();

    return 0;
}
