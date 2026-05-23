#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <iostream>
#include <string>
#include <utility>

#ifdef ENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH
#if !defined(ENABLE_LAZY_GUESS_BLOCK)
#error "ENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH requires ENABLE_LAZY_GUESS_BLOCK"
#endif
#if !defined(_OPENMP)
#error "ENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH requires OpenMP"
#endif
#include <cstdlib>
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

// 编译指令示例：
// scalar:
// g++ -O2 -std=c++17 correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_scalar
//
// SIMD hash:
// g++ -O2 -std=c++17 -DENABLE_SIMD_HASH_BATCH correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_simd
//
// Lazy Block + SIMD hash:
// g++ -O2 -std=c++17 -DENABLE_OPENMP_GENERATE -DENABLE_RELAXED_HEAP_PRIORITY -DENABLE_LAZY_GUESS_BLOCK -DENABLE_SIMD_HASH_BATCH -fopenmp correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_lazy_block_simd

#ifdef ENABLE_SIMD_HASH_BATCH
template <typename GuessRange>
static void HashBatchSIMDAndCount(
    const GuessRange &guesses,
    const unordered_set<string> &test_set,
    int &cracked
) {
    string inputs[4];
    bit32 states4[4][4];
    bit32 state_tail[4];
    int lane = 0;

    for (string pw : guesses) {
        // Cracked 统计逻辑保持不变：仍然对每一个候选口令查 test_set。
        if (test_set.find(pw) != test_set.end()) {
            cracked += 1;
        }

        inputs[lane] = std::move(pw);
        lane += 1;

        if (lane == 4) {
            MD5Hash4(inputs, states4);
            lane = 0;
        }
    }

    // 处理不足 4 个的尾部候选，不能跳过 MD5。
    for (int i = 0; i < lane; i++) {
        MD5Hash(inputs[i], state_tail);
    }
}
#endif

#ifdef ENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH

struct HashBlockTask
{
    size_t block_idx;
    size_t begin;
    size_t end;
};

static size_t GetLazyBlockHashChunkSize()
{
    const char* env = getenv("LAZY_BLOCK_HASH_CHUNK_SIZE");
    size_t v = 8192;
    if (env)
    {
        long parsed = atol(env);
        if (parsed > 0) v = static_cast<size_t>(parsed);
    }
    if (v < 256) v = 256;
    if (v > 65536) v = 65536;
    return v;
}

static inline string MaterializeBlockGuess(
    const LazyGuessBlockBuffer& guesses,
    const GuessBlock& block,
    size_t local_idx)
{
    const string& val = (*(block.values))[block.begin + local_idx];
    if (!block.has_prefix)
    {
        return val;
    }
    const string& prefix = guesses.prefix_storage[block.prefix_index];
    string out;
    out.reserve(prefix.size() + val.size());
    out.append(prefix);
    out.append(val);
    return out;
}

static void HashLazyBlockParallelSIMDAndCount(
    const LazyGuessBlockBuffer& guesses,
    const unordered_set<string>& test_set,
    int& cracked)
{
    const vector<GuessBlock>& blocks = guesses.blocks;
    size_t chunk_size = GetLazyBlockHashChunkSize();

    vector<HashBlockTask> tasks;
    for (size_t bi = 0; bi < blocks.size(); ++bi)
    {
        const GuessBlock& block = blocks[bi];
        if (block.count == 0) continue;
        for (size_t beg = 0; beg < block.count; beg += chunk_size)
        {
            size_t end = beg + chunk_size;
            if (end > block.count) end = block.count;
            tasks.push_back({bi, beg, end});
        }
    }

    int cracked_sum = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:cracked_sum)
    for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id)
    {
        const HashBlockTask& task = tasks[task_id];
        const GuessBlock& block = blocks[task.block_idx];

        string inputs[4];
        bit32 states4[4][4];
        bit32 state_tail[4];
        int lane = 0;
        int local_cracked = 0;

        for (size_t local_idx = task.begin; local_idx < task.end; ++local_idx)
        {
            string pw = MaterializeBlockGuess(guesses, block, local_idx);

            if (test_set.find(pw) != test_set.end())
            {
                local_cracked += 1;
            }

            inputs[lane] = std::move(pw);
            lane++;

            if (lane == 4)
            {
#ifdef ENABLE_SIMD_HASH_1BLOCK_FASTPATH
                if (inputs[0].size() <= 55 &&
                    inputs[1].size() <= 55 &&
                    inputs[2].size() <= 55 &&
                    inputs[3].size() <= 55)
                {
                    MD5Hash4_1Block(inputs, states4);
                }
                else
                {
                    MD5Hash4(inputs, states4);
                }
#else
                MD5Hash4(inputs, states4);
#endif
                lane = 0;
            }
        }

        for (int i = 0; i < lane; ++i)
        {
            MD5Hash(inputs[i], state_tail);
        }

        cracked_sum += local_cracked;
    }

    cracked += cracked_sum;
}

#endif // ENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH

int main()
{
    double time_hash = 0;   // 用于 MD5 哈希的时间
    double time_guess = 0;  // 哈希和猜测的总时长
    double time_train = 0;  // 模型训练的总时长

    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 加载测试数据：训练集前 1,000,000 个口令作为 cracked 判断集合
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

    int cracked = 0;

    q.init();
    cout << "here" << endl;

    int curr_num = 0;
    auto start = system_clock::now();

    // 由于需要定期清空内存，这里记录已经生成并处理过的口令总数
    int history = 0;

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 实验生成的猜测上限
            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                cout << "Cracked:" << cracked << endl;
                break;
            }
        }

        // 为了避免内存超限，当 q.guesses 中口令达到一定数目时，对其中口令进行 Hash，
        // 然后清空 q.guesses。history 用于记录已经处理过的口令数。
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

#if defined(ENABLE_LAZY_BLOCK_PARALLEL_SIMD_HASH)
            HashLazyBlockParallelSIMDAndCount(q.guesses, test_set, cracked);
#elif defined(ENABLE_SIMD_HASH_BATCH)
            HashBatchSIMDAndCount(q.guesses, test_set, cracked);
#else
            bit32 state[4];
            for (string pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end()) {
                    cracked += 1;
                }

                MD5Hash(pw, state);
            }
#endif

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    return 0;
}