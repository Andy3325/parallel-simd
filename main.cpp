#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

static inline size_t GetMD5BlockCount(const string& s)
{
    // MD5 最终总长度 = 原始长度 + 1字节0x80 + 8字节长度字段 + 若干补零
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

int main()
{
    // 下面代码用于测试MD5哈希的正确性
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

    for (int i = 0; i < 8; i++) {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1 += 1) {
            ss << std::setw(8) << std::setfill('0') << hex << state[i1];
        }
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }

    // 再测一次4路并行版本，避免你只验证了串行版本。
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
    cout << "MD5Hash test passed!" << endl; // 请不要修改这一行

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;

    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl; // 请不要修改这一行
                cout << "Hash time:" << time_hash << "seconds" << endl;               // 请不要修改这一行
                cout << "Train time:" << time_train << "seconds" << endl;             // 请不要修改这一行
                break;
            }
        }

        // 为了避免内存超限，我们在 q.guesses 中口令达到一定数目时，将其中的所有口令取出并进行哈希
        // 这里改成：先按 padding 后的 block 数分桶，同一个桶凑满 4 个再调用一次 MD5Hash4
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

            // key = block 数；value = 该 block 桶里的待哈希口令
            unordered_map<size_t, vector<string>> buckets;
            bit32 states4[4][4];

            // 先按 block 数分桶
            for (const string& pw : q.guesses)
            {
                size_t blk = GetMD5BlockCount(pw);
                auto& bucket = buckets[blk];
                bucket.emplace_back(pw);

                // 同一个桶凑够 4 个，就立刻并行哈希
                if (bucket.size() == 4)
                {
                    MD5Hash4(bucket.data(), states4);

                    bucket.clear();
                }
            }

            // 每个桶最后不足 4 个的尾巴，用串行补算
            for (auto& kv : buckets)
            {
                FlushBucketRemainder(kv.second);
            }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}