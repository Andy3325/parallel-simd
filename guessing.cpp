#include "PCFG.h"
#if defined(ENABLE_OPENMP_GENERATE) && defined(ENABLE_PTHREAD_GENERATE)
#error "Do not enable both ENABLE_OPENMP_GENERATE and ENABLE_PTHREAD_GENERATE"
#endif
#if defined(ENABLE_PRIORITY_LAZY_OPT) && defined(ENABLE_RELAXED_HEAP_PRIORITY)
#error "Do not enable both ENABLE_PRIORITY_LAZY_OPT and ENABLE_RELAXED_HEAP_PRIORITY"
#endif
#if defined(ENABLE_RELAXED_BATCH_POPNEXT) && (defined(ENABLE_PRIORITY_LAZY_OPT) || defined(ENABLE_RELAXED_HEAP_PRIORITY))
#error "Do not enable ENABLE_RELAXED_BATCH_POPNEXT with ENABLE_PRIORITY_LAZY_OPT or ENABLE_RELAXED_HEAP_PRIORITY"
#endif
#if defined(ENABLE_RELAXED_BATCH_POPNEXT) && !defined(_OPENMP)
#error "ENABLE_RELAXED_BATCH_POPNEXT requires OpenMP"
#endif
#if defined(ENABLE_OPENMP_GENERATE) || defined(ENABLE_RELAXED_BATCH_POPNEXT)
#include <omp.h>
#endif
#ifdef ENABLE_PTHREAD_GENERATE
#include <pthread.h>
#endif
#if defined(ENABLE_PTHREAD_GENERATE) || defined(ENABLE_RELAXED_BATCH_POPNEXT)
#include <cstdlib>
#endif
#ifndef GENERATE_PARALLEL_THRESHOLD
#define GENERATE_PARALLEL_THRESHOLD 4096
#endif
#include <algorithm>
#include <chrono>
using namespace std;

#if defined(ENABLE_RELAXED_HEAP_PRIORITY) || defined(ENABLE_RELAXED_BATCH_POPNEXT)
static bool PTProbLess(const PT& lhs, const PT& rhs)
{
    return lhs.prob < rhs.prob;
}
#endif

#ifdef ENABLE_RELAXED_BATCH_POPNEXT
struct GenerateChunkTask
{
    int pt_index;
    size_t begin;
    size_t end;
    size_t output_base;
};

static int GetClampedEnvInt(const char* name, int default_value, int min_value, int max_value)
{
    const char* env = std::getenv(name);
    int value = env == nullptr ? default_value : std::atoi(env);
    if (value < min_value)
    {
        value = min_value;
    }
    if (value > max_value)
    {
        value = max_value;
    }
    return value;
}

static int GetRelaxedBatchMaxPT()
{
    if (std::getenv("RELAXED_BATCH_MAX_PT") != nullptr)
    {
        return GetClampedEnvInt("RELAXED_BATCH_MAX_PT", 128, 1, 512);
    }
    if (std::getenv("RELAXED_BATCH_SIZE") != nullptr)
    {
        return GetClampedEnvInt("RELAXED_BATCH_SIZE", 64, 1, 128);
    }
    return 128;
}

static size_t GetRelaxedBatchTargetGuesses()
{
    return static_cast<size_t>(GetClampedEnvInt("RELAXED_BATCH_TARGET_GUESSES", 200000, 10000, 1000000));
}

static size_t GetRelaxedChunkSize()
{
    return static_cast<size_t>(GetClampedEnvInt("RELAXED_CHUNK_SIZE", 8192, 1024, 65536));
}

#endif

#ifdef ENABLE_PTHREAD_GENERATE
struct PthreadGenerateTask
{
    vector<string>* guesses;
    const string* prefix;
    segment* values;
    size_t base;
    int begin;
    int end;
};

static void* PthreadGenerateWorker(void* arg)
{
    PthreadGenerateTask* task = static_cast<PthreadGenerateTask*>(arg);
    vector<string>& guesses = *task->guesses;
    const string& prefix = *task->prefix;
    segment* values = task->values;

    for (int i = task->begin; i < task->end; ++i)
    {
        if (prefix.empty())
        {
            guesses[task->base + i] = values->ordered_values[i];
        }
        else
        {
            guesses[task->base + i] = prefix + values->ordered_values[i];
        }
    }

    return nullptr;
}

static int GetPthreadGenerateThreadCount()
{
    const char* env = std::getenv("PTHREAD_NUM_THREADS");
    int thread_count = env == nullptr ? 4 : std::atoi(env);
    if (thread_count < 1)
    {
        thread_count = 1;
    }
    return thread_count;
}
#endif

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    priority_head = 0;
    popnext_calls = 0;
    new_pts_count = 0;
    popnext_time_sec = 0.0;
    generate_in_popnext_time_sec = 0.0;
    newpts_time_sec = 0.0;
    calprob_time_sec = 0.0;
    priority_insert_time_sec = 0.0;
    priority_erase_time_sec = 0.0;
#ifdef ENABLE_RELAXED_BATCH_POPNEXT
    guesses.reserve(12000000);
#endif

    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
#if defined(ENABLE_RELAXED_HEAP_PRIORITY) || defined(ENABLE_RELAXED_BATCH_POPNEXT)
    make_heap(priority.begin(), priority.end(), PTProbLess);
#endif
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    popnext_calls += 1;
    auto popnext_start = std::chrono::steady_clock::now();

#ifdef ENABLE_RELAXED_BATCH_POPNEXT
    if (priority.empty())
    {
        auto popnext_end = std::chrono::steady_clock::now();
        popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
        return;
    }

    vector<PT> batch;
    vector<size_t> counts;
    int max_batch_pt = GetRelaxedBatchMaxPT();
    size_t target_guesses = GetRelaxedBatchTargetGuesses();
    size_t chunk_size = GetRelaxedChunkSize();
    batch.reserve(max_batch_pt);
    counts.reserve(max_batch_pt);
    size_t batch_generated = 0;

    auto priority_erase_start = std::chrono::steady_clock::now();
    while (!priority.empty())
    {
        pop_heap(priority.begin(), priority.end(), PTProbLess);
        batch.emplace_back(priority.back());
        priority.pop_back();

        size_t cnt = CountGeneratedGuesses(batch.back());
        counts.emplace_back(cnt);
        batch_generated += cnt;

        if (static_cast<int>(batch.size()) >= max_batch_pt || batch_generated >= target_guesses)
        {
            break;
        }
    }
    auto priority_erase_end = std::chrono::steady_clock::now();
    priority_erase_time_sec += std::chrono::duration<double>(priority_erase_end - priority_erase_start).count();

    auto generate_in_popnext_start = std::chrono::steady_clock::now();
    vector<size_t> offsets(batch.size());
    vector<GenerateChunkTask> chunk_tasks;
    chunk_tasks.reserve((batch_generated + chunk_size - 1) / chunk_size);
    size_t running_offset = 0;
    for (size_t i = 0; i < batch.size(); ++i)
    {
        offsets[i] = running_offset;
        for (size_t begin = 0; begin < counts[i]; begin += chunk_size)
        {
            size_t end = begin + chunk_size;
            if (end > counts[i])
            {
                end = counts[i];
            }
            chunk_tasks.push_back({static_cast<int>(i), begin, end, 0});
        }
        running_offset += counts[i];
    }

    size_t base = guesses.size();
    for (GenerateChunkTask& task : chunk_tasks)
    {
        task.output_base = base + offsets[task.pt_index] + task.begin;
    }

    auto append_start = std::chrono::steady_clock::now();
    guesses.resize(base + batch_generated);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < static_cast<int>(chunk_tasks.size()); ++i)
    {
        const GenerateChunkTask& task = chunk_tasks[i];
        GenerateToRangeChunk(batch[task.pt_index], guesses, task.output_base, task.begin, task.end);
    }
    auto append_end = std::chrono::steady_clock::now();
    auto generate_in_popnext_end = std::chrono::steady_clock::now();
    double batch_generate_time = std::chrono::duration<double>(generate_in_popnext_end - generate_in_popnext_start).count();
    generate_in_popnext_time_sec += batch_generate_time;
    generate_time_sec += batch_generate_time;
    append_time_sec += std::chrono::duration<double>(append_end - append_start).count();
    total_guesses += static_cast<int>(batch_generated);

    generate_calls += static_cast<long long>(batch.size());
    append_calls += static_cast<long long>(batch.size());
    append_total_items += static_cast<long long>(batch_generated);
    append_parallel_calls += static_cast<long long>(chunk_tasks.size());
    append_parallel_items += static_cast<long long>(batch_generated);

    vector<PT> new_pts_all;
    auto newpts_start = std::chrono::steady_clock::now();
    for (PT& current : batch)
    {
        vector<PT> new_pts = current.NewPTs();
        new_pts_count += new_pts.size();
        new_pts_all.insert(new_pts_all.end(), new_pts.begin(), new_pts.end());
    }
    auto newpts_end = std::chrono::steady_clock::now();
    newpts_time_sec += std::chrono::duration<double>(newpts_end - newpts_start).count();

    auto calprob_start = std::chrono::steady_clock::now();
    for (PT& pt : new_pts_all)
    {
        CalProb(pt);
    }
    auto calprob_end = std::chrono::steady_clock::now();
    calprob_time_sec += std::chrono::duration<double>(calprob_end - calprob_start).count();

    auto priority_insert_start = std::chrono::steady_clock::now();
    for (const PT& pt : new_pts_all)
    {
        priority.push_back(pt);
        push_heap(priority.begin(), priority.end(), PTProbLess);
    }
    auto priority_insert_end = std::chrono::steady_clock::now();
    priority_insert_time_sec += std::chrono::duration<double>(priority_insert_end - priority_insert_start).count();

    auto popnext_end = std::chrono::steady_clock::now();
    popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
#elif defined(ENABLE_RELAXED_HEAP_PRIORITY)
    if (priority.empty())
    {
        auto popnext_end = std::chrono::steady_clock::now();
        popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
        return;
    }

    auto priority_erase_start = std::chrono::steady_clock::now();
    pop_heap(priority.begin(), priority.end(), PTProbLess);
    PT current = priority.back();
    priority.pop_back();
    auto priority_erase_end = std::chrono::steady_clock::now();
    priority_erase_time_sec += std::chrono::duration<double>(priority_erase_end - priority_erase_start).count();

    auto generate_in_popnext_start = std::chrono::steady_clock::now();
    Generate(current);
    auto generate_in_popnext_end = std::chrono::steady_clock::now();
    generate_in_popnext_time_sec += std::chrono::duration<double>(generate_in_popnext_end - generate_in_popnext_start).count();

    auto newpts_start = std::chrono::steady_clock::now();
    vector<PT> new_pts = current.NewPTs();
    auto newpts_end = std::chrono::steady_clock::now();
    newpts_time_sec += std::chrono::duration<double>(newpts_end - newpts_start).count();
    new_pts_count += new_pts.size();

    for (PT pt : new_pts)
    {
        auto calprob_start = std::chrono::steady_clock::now();
        CalProb(pt);
        auto calprob_end = std::chrono::steady_clock::now();
        calprob_time_sec += std::chrono::duration<double>(calprob_end - calprob_start).count();

        auto priority_insert_start = std::chrono::steady_clock::now();
        priority.emplace_back(pt);
        push_heap(priority.begin(), priority.end(), PTProbLess);
        auto priority_insert_end = std::chrono::steady_clock::now();
        priority_insert_time_sec += std::chrono::duration<double>(priority_insert_end - priority_insert_start).count();
    }

    auto popnext_end = std::chrono::steady_clock::now();
    popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
#elif defined(ENABLE_PRIORITY_LAZY_OPT)
    if (priority_head >= priority.size())
    {
        priority.clear();
        priority_head = 0;
        auto popnext_end = std::chrono::steady_clock::now();
        popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
        return;
    }

    PT current = priority[priority_head];

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    auto generate_in_popnext_start = std::chrono::steady_clock::now();
    Generate(current);
    auto generate_in_popnext_end = std::chrono::steady_clock::now();
    generate_in_popnext_time_sec += std::chrono::duration<double>(generate_in_popnext_end - generate_in_popnext_start).count();

    // 然后需要根据即将出队的PT，生成一系列新的PT
    auto newpts_start = std::chrono::steady_clock::now();
    vector<PT> new_pts = current.NewPTs();
    auto newpts_end = std::chrono::steady_clock::now();
    newpts_time_sec += std::chrono::duration<double>(newpts_end - newpts_start).count();
    new_pts_count += new_pts.size();
    for (PT pt : new_pts)
    {
        // 计算概率
        auto calprob_start = std::chrono::steady_clock::now();
        CalProb(pt);
        auto calprob_end = std::chrono::steady_clock::now();
        calprob_time_sec += std::chrono::duration<double>(calprob_end - calprob_start).count();
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        auto priority_insert_start = std::chrono::steady_clock::now();
        auto active_begin = priority.begin() + priority_head;
        for (auto iter = active_begin; iter != priority.end(); iter++)
        {
            // 复用原 sorted-vector 的插入判定，只把 begin 改成逻辑队首。
            if (iter != priority.end() - 1 && iter != active_begin)
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == active_begin && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
        auto priority_insert_end = std::chrono::steady_clock::now();
        priority_insert_time_sec += std::chrono::duration<double>(priority_insert_end - priority_insert_start).count();
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    auto priority_erase_start = std::chrono::steady_clock::now();
    priority_head += 1;
    if (priority_head >= priority.size())
    {
        priority.clear();
        priority_head = 0;
    }
    else if (priority_head > 4096 && priority_head * 2 > priority.size())
    {
        priority.erase(priority.begin(), priority.begin() + priority_head);
        priority_head = 0;
    }
    auto priority_erase_end = std::chrono::steady_clock::now();
    priority_erase_time_sec += std::chrono::duration<double>(priority_erase_end - priority_erase_start).count();

    auto popnext_end = std::chrono::steady_clock::now();
    popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
#else
    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    auto generate_in_popnext_start = std::chrono::steady_clock::now();
    Generate(priority.front());
    auto generate_in_popnext_end = std::chrono::steady_clock::now();
    generate_in_popnext_time_sec += std::chrono::duration<double>(generate_in_popnext_end - generate_in_popnext_start).count();

    // 然后需要根据即将出队的PT，生成一系列新的PT
    auto newpts_start = std::chrono::steady_clock::now();
    vector<PT> new_pts = priority.front().NewPTs();
    auto newpts_end = std::chrono::steady_clock::now();
    newpts_time_sec += std::chrono::duration<double>(newpts_end - newpts_start).count();
    new_pts_count += new_pts.size();
    for (PT pt : new_pts)
    {
        // 计算概率
        auto calprob_start = std::chrono::steady_clock::now();
        CalProb(pt);
        auto calprob_end = std::chrono::steady_clock::now();
        calprob_time_sec += std::chrono::duration<double>(calprob_end - calprob_start).count();
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        auto priority_insert_start = std::chrono::steady_clock::now();
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
        auto priority_insert_end = std::chrono::steady_clock::now();
        priority_insert_time_sec += std::chrono::duration<double>(priority_insert_end - priority_insert_start).count();
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    auto priority_erase_start = std::chrono::steady_clock::now();
    priority.erase(priority.begin());
    auto priority_erase_end = std::chrono::steady_clock::now();
    priority_erase_time_sec += std::chrono::duration<double>(priority_erase_end - priority_erase_start).count();

    auto popnext_end = std::chrono::steady_clock::now();
    popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
#endif
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}
size_t PriorityQueue::CountGeneratedGuesses(const PT& pt)
{
    if (pt.content.empty() || pt.max_indices.size() < pt.content.size())
    {
        return 0;
    }
    return static_cast<size_t>(pt.max_indices[pt.content.size() - 1]);
}

void PriorityQueue::GenerateToRangeChunk(const PT& pt, vector<string>& dst, size_t output_base, size_t begin, size_t end)
{
    auto getSegmentPtr = [this](const segment& seg) -> segment*
    {
        if (seg.type == 1)
        {
            return &m.letters[m.FindLetter(seg)];
        }
        if (seg.type == 2)
        {
            return &m.digits[m.FindDigit(seg)];
        }
        if (seg.type == 3)
        {
            return &m.symbols[m.FindSymbol(seg)];
        }
        return nullptr;
    };

    auto writeSegmentValues = [&dst, output_base, begin, end](const string& prefix, segment* a)
    {
        for (size_t i = begin; i < end; ++i)
        {
            if (prefix.empty())
            {
                dst[output_base + i - begin] = a->ordered_values[i];
            }
            else
            {
                dst[output_base + i - begin] = prefix + a->ordered_values[i];
            }
        }
    };

    size_t total_count = CountGeneratedGuesses(pt);
    if (pt.content.empty() || begin >= end || end > total_count)
    {
        return;
    }

    if (pt.content.size() == 1)
    {
        segment* a = getSegmentPtr(pt.content[0]);
        writeSegmentValues("", a);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        segment* a = getSegmentPtr(pt.content[pt.content.size() - 1]);
        writeSegmentValues(guess, a);
    }
}

void PriorityQueue::GenerateToRange(const PT& pt, vector<string>& dst, size_t base)
{
    GenerateToRangeChunk(pt, dst, base, 0, CountGeneratedGuesses(pt));
}

void PriorityQueue::GenerateToVector(PT pt, vector<string>& out)
{
    CalProb(pt);

    auto getSegmentPtr = [this](const segment& seg) -> segment*
    {
        if (seg.type == 1)
        {
            return &m.letters[m.FindLetter(seg)];
        }
        if (seg.type == 2)
        {
            return &m.digits[m.FindDigit(seg)];
        }
        if (seg.type == 3)
        {
            return &m.symbols[m.FindSymbol(seg)];
        }
        return nullptr;
    };

    auto appendSegmentValues = [&out](const string& prefix, segment* a, int n)
    {
        size_t base = out.size();
        out.resize(base + n);
        for (int i = 0; i < n; ++i)
        {
            if (prefix.empty())
            {
                out[base + i] = a->ordered_values[i];
            }
            else
            {
                out[base + i] = prefix + a->ordered_values[i];
            }
        }
    };

    if (pt.content.size() == 1)
    {
        segment* a = getSegmentPtr(pt.content[0]);
        appendSegmentValues("", a, pt.max_indices[0]);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        segment* a = getSegmentPtr(pt.content[pt.content.size() - 1]);
        appendSegmentValues(guess, a, pt.max_indices[pt.content.size() - 1]);
    }
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    generate_calls += 1;
    auto generate_start = std::chrono::high_resolution_clock::now();

    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    auto getSegmentPtr = [this](const segment& seg) -> segment*
    {
        if (seg.type == 1)
        {
            return &m.letters[m.FindLetter(seg)];
        }
        if (seg.type == 2)
        {
            return &m.digits[m.FindDigit(seg)];
        }
        if (seg.type == 3)
        {
            return &m.symbols[m.FindSymbol(seg)];
        }
        return nullptr;
    };

    const int PARALLEL_THRESHOLD = GENERATE_PARALLEL_THRESHOLD;

    auto appendSegmentValues = [this, PARALLEL_THRESHOLD](const string& prefix, segment* a, int n)
    {
        append_calls += 1;
        append_total_items += n;
        auto append_start = std::chrono::high_resolution_clock::now();

        size_t base = guesses.size();
        guesses.resize(base + n);
#if defined(ENABLE_PTHREAD_GENERATE)
        if (n >= PARALLEL_THRESHOLD)
        {
            append_parallel_calls += 1;
            append_parallel_items += n;

            int thread_count = GetPthreadGenerateThreadCount();
            vector<pthread_t> threads(thread_count);
            vector<PthreadGenerateTask> tasks(thread_count);

            for (int t = 0; t < thread_count; ++t)
            {
                int begin = t * n / thread_count;
                int end = (t + 1) * n / thread_count;
                tasks[t] = {&guesses, &prefix, a, base, begin, end};
                pthread_create(&threads[t], nullptr, PthreadGenerateWorker, &tasks[t]);
            }

            for (int t = 0; t < thread_count; ++t)
            {
                pthread_join(threads[t], nullptr);
            }

            total_guesses += n;
            auto append_end = std::chrono::high_resolution_clock::now();
            append_time_sec += std::chrono::duration<double>(append_end - append_start).count();
            return;
        }
#endif
#if defined(ENABLE_OPENMP_GENERATE) && defined(_OPENMP)
        if (n >= PARALLEL_THRESHOLD)
        {
            append_parallel_calls += 1;
            append_parallel_items += n;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                if (prefix.empty())
                {
                    guesses[base + i] = a->ordered_values[i];
                }
                else
                {
                    guesses[base + i] = prefix + a->ordered_values[i];
                }
            }
            total_guesses += n;
            auto append_end = std::chrono::high_resolution_clock::now();
            append_time_sec += std::chrono::duration<double>(append_end - append_start).count();
            return;
        }
#endif
        append_serial_calls += 1;
        for (int i = 0; i < n; i += 1)
        {
            if (prefix.empty())
            {
                guesses[base + i] = a->ordered_values[i];
            }
            else
            {
                guesses[base + i] = prefix + a->ordered_values[i];
            }
        }
        total_guesses += n;
        auto append_end = std::chrono::high_resolution_clock::now();
        append_time_sec += std::chrono::duration<double>(append_end - append_start).count();
    };

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a = getSegmentPtr(pt.content[0]);
        // 在模型中定位到这个segment
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        appendSegmentValues("", a, pt.max_indices[0]);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a = getSegmentPtr(pt.content[pt.content.size() - 1]);
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        appendSegmentValues(guess, a, pt.max_indices[pt.content.size() - 1]);
    }

    auto generate_end = std::chrono::high_resolution_clock::now();
    generate_time_sec += std::chrono::duration<double>(generate_end - generate_start).count();
}

void PriorityQueue::PrintGenerateStats() const
{
    cout << "[GenerateStats]" << endl;
    cout << "generate_calls = " << generate_calls << endl;
    cout << "append_calls = " << append_calls << endl;
    cout << "append_serial_calls = " << append_serial_calls << endl;
    cout << "append_parallel_calls = " << append_parallel_calls << endl;
    cout << "append_total_items = " << append_total_items << endl;
    cout << "append_parallel_items = " << append_parallel_items << endl;
    cout << "generate_time_sec = " << generate_time_sec << endl;
    cout << "append_time_sec = " << append_time_sec << endl;
    cout << "[PopNextStats]" << endl;
#ifdef ENABLE_RELAXED_BATCH_POPNEXT
    cout << "priority_queue_mode = relaxed_batch_popnext" << endl;
#elif defined(ENABLE_RELAXED_HEAP_PRIORITY)
    cout << "priority_queue_mode = relaxed_heap" << endl;
#elif defined(ENABLE_PRIORITY_LAZY_OPT)
    cout << "priority_queue_mode = sorted_vector_lazy" << endl;
#else
    cout << "priority_queue_mode = sorted_vector" << endl;
#endif
    cout << "popnext_calls = " << popnext_calls << endl;
    cout << "new_pts_count = " << new_pts_count << endl;
    cout << "popnext_time_sec = " << popnext_time_sec << endl;
    cout << "generate_in_popnext_time_sec = " << generate_in_popnext_time_sec << endl;
    cout << "newpts_time_sec = " << newpts_time_sec << endl;
    cout << "calprob_time_sec = " << calprob_time_sec << endl;
    cout << "priority_insert_time_sec = " << priority_insert_time_sec << endl;
    cout << "priority_erase_time_sec = " << priority_erase_time_sec << endl;
}
