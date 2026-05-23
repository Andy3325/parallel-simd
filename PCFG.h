#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <deque>
#include <omp.h>
// #include <chrono>   
// using namespace chrono;
using namespace std;

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;


    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices;
    // void init();
    float preterm_prob;
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
#if defined(ENABLE_LAZY_GUESS_REF) && defined(ENABLE_LAZY_GUESS_BLOCK)
#error "ENABLE_LAZY_GUESS_REF and ENABLE_LAZY_GUESS_BLOCK are mutually exclusive"
#endif

#ifdef ENABLE_LAZY_GUESS_BLOCK
struct GuessBlock
{
    size_t prefix_index;
    const vector<string>* values;
    size_t begin;
    size_t count;
    bool has_prefix;
};

class LazyGuessBlockBuffer
{
public:
    vector<GuessBlock> blocks;
    deque<string> prefix_storage;
    size_t total_count = 0;

    size_t size() const
    {
        return total_count;
    }

    bool empty() const
    {
        return total_count == 0;
    }

    void clear()
    {
        blocks.clear();
        prefix_storage.clear();
        total_count = 0;
    }

    void resize(size_t n)
    {
        (void)n;
    }

    size_t add_prefix(string prefix)
    {
        prefix_storage.push_back(std::move(prefix));
        return prefix_storage.size() - 1;
    }

    void add_block(size_t prefix_index,
                   const vector<string>* values,
                   size_t begin,
                   size_t count,
                   bool has_prefix)
    {
        blocks.push_back({prefix_index, values, begin, count, has_prefix});
        total_count += count;
    }

    string materialize_block_value(const GuessBlock& block,
                                   size_t local_idx) const
    {
        const string& val = (*(block.values))[block.begin + local_idx];
        if (!block.has_prefix)
        {
            return val;
        }
        const string& prefix = prefix_storage[block.prefix_index];
        string out;
        out.reserve(prefix.size() + val.size());
        out.append(prefix);
        out.append(val);
        return out;
    }

    string materialize(size_t global_idx) const
    {
        size_t offset = 0;
        for (const GuessBlock& block : blocks)
        {
            if (global_idx < offset + block.count)
            {
                return materialize_block_value(block, global_idx - offset);
            }
            offset += block.count;
        }
        return "";
    }

    string operator[](size_t global_idx) const
    {
        return materialize(global_idx);
    }

    struct iterator
    {
        const LazyGuessBlockBuffer* owner;
        size_t block_idx;
        size_t local_idx;

        bool operator!=(const iterator& other) const
        {
            return block_idx != other.block_idx || local_idx != other.local_idx;
        }

        iterator& operator++()
        {
            local_idx += 1;
            if (block_idx < owner->blocks.size() &&
                local_idx >= owner->blocks[block_idx].count)
            {
                block_idx += 1;
                local_idx = 0;
            }
            return *this;
        }

        string operator*() const
        {
            return owner->materialize_block_value(
                owner->blocks[block_idx], local_idx);
        }
    };

    iterator begin() const
    {
        if (blocks.empty())
        {
            return end();
        }
        return {this, 0, 0};
    }

    iterator end() const
    {
        return {this, blocks.size(), 0};
    }
};
#endif

#ifdef ENABLE_LAZY_GUESS_REF
struct GuessRef
{
    size_t prefix_index;
    const vector<string>* values;
    int value_idx;
    bool has_prefix;
};

class LazyGuessBuffer
{
public:
    vector<GuessRef> refs;
    deque<string> prefix_storage;

    size_t size() const
    {
        return refs.size();
    }

    bool empty() const
    {
        return refs.empty();
    }

    void clear()
    {
        refs.clear();
        prefix_storage.clear();
    }

    void resize(size_t n)
    {
        (void)n;
    }

    struct SlotProxy
    {
        const LazyGuessBuffer* owner;
        size_t idx;

        SlotProxy& operator=(const string& value)
        {
            (void)value;
            return *this;
        }

        operator string() const
        {
            return owner->materialize(idx);
        }
    };

    SlotProxy operator[](size_t idx) const
    {
        return {this, idx};
    }

    size_t add_prefix(string prefix)
    {
        prefix_storage.push_back(prefix);
        return prefix_storage.size() - 1;
    }

    void add_ref(size_t prefix_index, const vector<string>* values, int value_idx, bool has_prefix)
    {
        refs.push_back({prefix_index, values, value_idx, has_prefix});
    }

    string materialize(size_t idx) const
    {
        const GuessRef& ref = refs[idx];
        const string& value = (*ref.values)[ref.value_idx];
        if (!ref.has_prefix)
        {
            return value;
        }
        return prefix_storage[ref.prefix_index] + value;
    }

    struct iterator
    {
        const LazyGuessBuffer* owner;
        size_t idx;

        bool operator!=(const iterator& other) const
        {
            return idx != other.idx;
        }

        iterator& operator++()
        {
            idx += 1;
            return *this;
        }

        string operator*() const
        {
            return owner->materialize(idx);
        }
    };

    iterator begin() const
    {
        return {this, 0};
    }

    iterator end() const
    {
        return {this, refs.size()};
    }
};
#endif

class PriorityQueue
{
public:
    // 用vector实现的priority queue
    vector<PT> priority;
    size_t priority_head = 0;

    // 模型作为成员，辅助猜测生成
    model m;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);
    void GenerateToVector(PT pt, vector<string>& out);
    size_t CountGeneratedGuesses(const PT& pt);
    void GenerateToRange(const PT& pt, vector<string>& dst, size_t base);
    void GenerateToRangeChunk(const PT& pt, vector<string>& dst, size_t output_base, size_t begin, size_t end);

    void PrintGenerateStats() const;

    // 将优先队列最前面的一个PT
    void PopNext();
    int total_guesses = 0;
#ifdef ENABLE_LAZY_GUESS_BLOCK
    LazyGuessBlockBuffer guesses;
#elif defined(ENABLE_LAZY_GUESS_REF)
    LazyGuessBuffer guesses;
#else
    vector<string> guesses;
#endif

    long long generate_calls = 0;
    long long append_calls = 0;
    long long append_serial_calls = 0;
    long long append_parallel_calls = 0;
    long long append_total_items = 0;
    long long append_parallel_items = 0;
    long long popnext_calls = 0;
    long long new_pts_count = 0;
    double generate_time_sec = 0.0;
    double append_time_sec = 0.0;
    double popnext_time_sec = 0.0;
    double generate_in_popnext_time_sec = 0.0;
    double newpts_time_sec = 0.0;
    double calprob_time_sec = 0.0;
    double priority_insert_time_sec = 0.0;
    double priority_erase_time_sec = 0.0;
};
