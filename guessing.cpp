#include "PCFG.h"
#if defined(ENABLE_OPENMP_GENERATE) && defined(ENABLE_PTHREAD_GENERATE)
#error "Do not enable both ENABLE_OPENMP_GENERATE and ENABLE_PTHREAD_GENERATE"
#endif
#ifdef ENABLE_OPENMP_GENERATE
#include <omp.h>
#endif
#ifdef ENABLE_PTHREAD_GENERATE
#include <pthread.h>
#include <cstdlib>
#endif
#ifdef ENABLE_HIP_GENERATE
#include "gpu_generate.h"
#endif
#ifdef ENABLE_CUDA_GENERATE
#include "gpu_generate_cuda.h"
#endif
#ifndef GENERATE_PARALLEL_THRESHOLD
#define GENERATE_PARALLEL_THRESHOLD 4096
#endif
#include <algorithm>
#include <chrono>
using namespace std;

#ifdef ENABLE_RELAXED_HEAP_PRIORITY
static bool PTProbLess(const PT& lhs, const PT& rhs)
{
    return lhs.prob < rhs.prob;
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
    // 璁＄畻PriorityQueue閲岄潰涓€涓狿T鐨勬祦绋嬪涓嬶細
    // 1. 棣栧厛闇€瑕佽绠椾竴涓狿T鏈韩鐨勬鐜囥€備緥濡傦紝L6S1鐨勬鐜囦负0.15
    // 2. 闇€瑕佹敞鎰忕殑鏄紝Queue閲岄潰鐨凱T涓嶆槸鈥滅函绮圭殑鈥漃T锛岃€屾槸闄や簡鏈€鍚庝竴涓猻egment浠ュ锛屽叏閮ㄨvalue瀹炰緥鍖栫殑PT
    // 3. 鎵€浠ワ紝瀵逛簬L6S1鑰岃█锛屽叾鍦≦ueue閲岄潰鐨勫疄闄匬T鍙兘鏄?23456S1锛屽叾涓€?23456鈥濅负L6鐨勪竴涓叿浣搗alue銆?
    // 4. 杩欎釜鏃跺€欏氨闇€瑕佽绠?23456鍦↙6涓嚭鐜扮殑姒傜巼浜嗐€傚亣璁?23456鍦ㄦ墍鏈塋6 segment涓殑姒傜巼涓?.1锛岄偅涔?23456S1鐨勬鐜囧氨鏄?.1*0.15

    // 璁＄畻涓€涓狿T鏈韩鐨勬鐜囥€傚悗缁墍鏈夊叿浣搒egment value鐨勬鐜囷紝鐩存帴绱箻鍦ㄨ繖涓垵濮嬫鐜囧€间笂
    pt.prob = pt.preterm_prob;

    // index: 鏍囨敞褰撳墠segment鍦≒T涓殑浣嶇疆
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 涓嬮潰杩欒浠ｇ爜鐨勬剰涔夛細
            // pt.content[index]锛氱洰鍓嶉渶瑕佽绠楁鐜囩殑segment
            // m.FindLetter(seg): 鎵惧埌涓€涓猯etter segment鍦ㄦā鍨嬩腑鐨勫搴斾笅鏍?
            // m.letters[m.FindLetter(seg)]锛氫竴涓猯etter segment鍦ㄦā鍨嬩腑瀵瑰簲鐨勬墍鏈夌粺璁℃暟鎹?
            // m.letters[m.FindLetter(seg)].ordered_values锛氫竴涓猯etter segment鍦ㄦā鍨嬩腑锛屾墍鏈塿alue鐨勬€绘暟鐩?
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
    // cout << m.ordered_pts.size() << endl;
    // 鐢ㄦ墍鏈夊彲鑳界殑PT锛屾寜姒傜巼闄嶅簭濉弧鏁翠釜浼樺厛闃熷垪
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 涓嬮潰杩欒浠ｇ爜鐨勬剰涔夛細
                // max_indices鐢ㄦ潵琛ㄧずPT涓悇涓猻egment鐨勫彲鑳芥暟鐩€備緥濡傦紝L6S1涓紝鍋囪妯″瀷缁熻鍒颁簡100涓狶6锛岄偅涔圠6瀵瑰簲鐨勬渶澶т笅鏍囧氨鏄?9
                // 锛堜絾鐢变簬鍚庨潰閲囩敤浜?<"鐨勬瘮杈冨叧绯伙紝鎵€浠ュ叾瀹瀖ax_indices[0]=100锛?
                // m.FindLetter(seg): 鎵惧埌涓€涓猯etter segment鍦ㄦā鍨嬩腑鐨勫搴斾笅鏍?
                // m.letters[m.FindLetter(seg)]锛氫竴涓猯etter segment鍦ㄦā鍨嬩腑瀵瑰簲鐨勬墍鏈夌粺璁℃暟鎹?
                // m.letters[m.FindLetter(seg)].ordered_values锛氫竴涓猯etter segment鍦ㄦā鍨嬩腑锛屾墍鏈塿alue鐨勬€绘暟鐩?
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

        // 璁＄畻褰撳墠pt鐨勬鐜?
        CalProb(pt);
        // 灏哖T鏀惧叆浼樺厛闃熷垪
        priority.emplace_back(pt);
    }
#ifdef ENABLE_RELAXED_HEAP_PRIORITY
    make_heap(priority.begin(), priority.end(), PTProbLess);
#endif
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    popnext_calls += 1;
    auto popnext_start = std::chrono::steady_clock::now();

#ifdef ENABLE_RELAXED_HEAP_PRIORITY
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

#else
    // 瀵逛紭鍏堥槦鍒楁渶鍓嶉潰鐨凱T锛岄鍏堝埄鐢ㄨ繖涓狿T鐢熸垚涓€绯诲垪鐚滄祴
    auto generate_in_popnext_start = std::chrono::steady_clock::now();
    Generate(priority.front());
    auto generate_in_popnext_end = std::chrono::steady_clock::now();
    generate_in_popnext_time_sec += std::chrono::duration<double>(generate_in_popnext_end - generate_in_popnext_start).count();

    // 鐒跺悗闇€瑕佹牴鎹嵆灏嗗嚭闃熺殑PT锛岀敓鎴愪竴绯诲垪鏂扮殑PT
    auto newpts_start = std::chrono::steady_clock::now();
    vector<PT> new_pts = priority.front().NewPTs();
    auto newpts_end = std::chrono::steady_clock::now();
    newpts_time_sec += std::chrono::duration<double>(newpts_end - newpts_start).count();
    new_pts_count += new_pts.size();
    for (PT pt : new_pts)
    {
        // 璁＄畻姒傜巼
        auto calprob_start = std::chrono::steady_clock::now();
        CalProb(pt);
        auto calprob_end = std::chrono::steady_clock::now();
        calprob_time_sec += std::chrono::duration<double>(calprob_end - calprob_start).count();
        // 鎺ヤ笅鏉ョ殑杩欎釜寰幆锛屼綔鐢ㄦ槸鏍规嵁姒傜巼锛屽皢鏂扮殑PT鎻掑叆鍒颁紭鍏堥槦鍒椾腑
        auto priority_insert_start = std::chrono::steady_clock::now();
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 瀵逛簬闈為槦棣栧拰闃熷熬鐨勭壒娈婃儏鍐?
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 鍒ゅ畾姒傜巼
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

    // 鐜板湪闃熼鐨凱T鍠勫悗宸ヤ綔宸茬粡缁撴潫锛屽皢鍏跺嚭闃燂紙鍒犻櫎锛?
    auto priority_erase_start = std::chrono::steady_clock::now();
    priority.erase(priority.begin());
    auto priority_erase_end = std::chrono::steady_clock::now();
    priority_erase_time_sec += std::chrono::duration<double>(priority_erase_end - priority_erase_start).count();

    auto popnext_end = std::chrono::steady_clock::now();
    popnext_time_sec += std::chrono::duration<double>(popnext_end - popnext_start).count();
#endif
}

// 杩欎釜鍑芥暟浣犲氨绠楃湅涓嶆噦锛屽骞惰绠楁硶鐨勫疄鐜板奖鍝嶄篃涓嶅ぇ
// 褰撶劧濡傛灉浣犳兂鍋氫竴涓熀浜庡浼樺厛闃熷垪鐨勫苟琛岀畻娉曪紝鍙兘寰楃◢寰湅涓€鐪嬩簡
vector<PT> PT::NewPTs()
{
    // 瀛樺偍鐢熸垚鐨勬柊PT
    vector<PT> res;

    // 鍋囧杩欎釜PT鍙湁涓€涓猻egment
    // 閭ｄ箞杩欎釜segment鐨勬墍鏈塿alue鍦ㄥ嚭闃熷墠灏卞凡缁忚閬嶅巻瀹屾瘯锛屽苟浣滀负鐚滄祴杈撳嚭
    // 鍥犳锛屾墍鏈夎繖涓狿T鍙兘瀵瑰簲鐨勫彛浠ょ寽娴嬪凡缁忛亶鍘嗗畬鎴愶紝鏃犻渶鐢熸垚鏂扮殑PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 鏈€鍒濈殑pivot鍊笺€傛垜浠皢鏇存敼浣嶇疆涓嬫爣澶т簬绛変簬杩欎釜pivot鍊肩殑segment鐨勫€硷紙鏈€鍚庝竴涓猻egment闄ゅ锛夛紝骞朵笖涓€娆″彧鏇存敼涓€涓猻egment
        // 涓婇潰杩欏彞璇濋噷鏄笉鏄湁娌＄湅鎳傜殑鍦版柟锛熸帴鐫€寰€涓嬬湅浣犲簲璇ヤ細鏇存槑鐧?
        int init_pivot = pivot;

        // 寮€濮嬮亶鍘嗘墍鏈変綅缃€煎ぇ浜庣瓑浜巌nit_pivot鍊肩殑segment
        // 娉ㄦ剰i < curr_indices.size() - 1锛屼篃灏辨槸闄ゅ幓浜嗘渶鍚庝竴涓猻egment锛堣繖涓猻egment鐨勮祴鍊奸鐣欑粰骞惰鐜妭锛?
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 鏍囪鍚剆egment鐩墠鐨剉alue鍦ㄦā鍨嬮噷瀵瑰簲鐨勪笅鏍?
            curr_indices[i] += 1;

            // max_indices锛氭爣璁板悇segment鍦ㄦā鍨嬩腑涓€鍏辨湁澶氬皯涓獀alue
            if (curr_indices[i] < max_indices[i])
            {
                // 鏇存柊pivot鍊?
                pivot = i;
                res.emplace_back(*this);
            }

            // 杩欎釜姝ラ瀵逛簬浣犵悊瑙ivot鐨勪綔鐢ㄣ€佹柊PT鐢熸垚鐨勮繃绋嬭€岃█锛岃嚦鍏抽噸瑕?
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


// 杩欎釜鍑芥暟鏄疨CFG骞惰鍖栫畻娉曠殑涓昏杞戒綋
// 灏介噺鐪嬫噦锛岀劧鍚庤繘琛屽苟琛屽疄鐜?
void PriorityQueue::Generate(PT pt)
{
    generate_calls += 1;
    auto generate_start = std::chrono::high_resolution_clock::now();

    // 璁＄畻PT鐨勬鐜囷紝杩欓噷涓昏鏄粰PT鐨勬鐜囪繘琛屽垵濮嬪寲
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

#if defined(ENABLE_LAZY_GUESS_BLOCK)
        {
        // block-level lazy: one GuessBlock per PT/segment, not per candidate
        bool has_prefix = !prefix.empty();
        size_t prefix_id = 0;
        if (has_prefix)
        {
            prefix_id = guesses.add_prefix(prefix);
        }
        guesses.add_block(prefix_id, &a->ordered_values, 0, static_cast<size_t>(n), has_prefix);
        append_serial_calls += 1;
        total_guesses += n;
        auto append_end = std::chrono::high_resolution_clock::now();
        append_time_sec += std::chrono::duration<double>(append_end - append_start).count();
        return;
        }
#endif

        size_t base = guesses.size();
        guesses.resize(base + n);
#if defined(ENABLE_HIP_GENERATE) && !defined(ENABLE_LAZY_GUESS_BLOCK)
        if (n >= HIP_GENERATE_THRESHOLD)
        {
            double hip_h2d = 0.0;
            double hip_kernel = 0.0;
            double hip_d2h = 0.0;
            auto hip_start = std::chrono::high_resolution_clock::now();
            bool hip_ok = HipGenerateSegmentValues(
                prefix,
                a->ordered_values,
                n,
                guesses,
                base,
                &hip_h2d,
                &hip_kernel,
                &hip_d2h);
            auto hip_end = std::chrono::high_resolution_clock::now();

            if (hip_ok)
            {
                hip_generate_calls += 1;
                hip_generate_items += n;
                hip_generate_total_time_sec += std::chrono::duration<double>(hip_end - hip_start).count();
                hip_h2d_time_sec += hip_h2d;
                hip_kernel_time_sec += hip_kernel;
                hip_d2h_time_sec += hip_d2h;
                total_guesses += n;
                auto append_end = std::chrono::high_resolution_clock::now();
                append_time_sec += std::chrono::duration<double>(append_end - append_start).count();
                return;
            }
        }
#endif
#if defined(ENABLE_CUDA_GENERATE) && !defined(ENABLE_LAZY_GUESS_BLOCK)
        if (n >= CUDA_GENERATE_THRESHOLD)
        {
            double cuda_h2d = 0.0;
            double cuda_kernel = 0.0;
            double cuda_d2h = 0.0;
            auto cuda_start = std::chrono::high_resolution_clock::now();
            bool cuda_ok = CudaGenerateSegmentValues(
                prefix,
                a->ordered_values,
                n,
                guesses,
                base,
                &cuda_h2d,
                &cuda_kernel,
                &cuda_d2h);
            auto cuda_end = std::chrono::high_resolution_clock::now();

            if (cuda_ok)
            {
                cuda_generate_calls += 1;
                cuda_generate_items += n;
                cuda_generate_total_time_sec += std::chrono::duration<double>(cuda_end - cuda_start).count();
                cuda_h2d_time_sec += cuda_h2d;
                cuda_kernel_time_sec += cuda_kernel;
                cuda_d2h_time_sec += cuda_d2h;
                total_guesses += n;
                auto append_end = std::chrono::high_resolution_clock::now();
                append_time_sec += std::chrono::duration<double>(append_end - append_start).count();
                return;
            }
        }
#endif
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

    // 瀵逛簬鍙湁涓€涓猻egment鐨凱T锛岀洿鎺ラ亶鍘嗙敓鎴愬叾涓殑鎵€鏈塿alue鍗冲彲
    if (pt.content.size() == 1)
    {
        // 鎸囧悜鏈€鍚庝竴涓猻egment鐨勬寚閽堬紝杩欎釜鎸囬拡瀹為檯鎸囧悜妯″瀷涓殑缁熻鏁版嵁
        segment *a = getSegmentPtr(pt.content[0]);
        // 鍦ㄦā鍨嬩腑瀹氫綅鍒拌繖涓猻egment
        
        // Multi-thread TODO锛?
        // 杩欎釜for寰幆灏辨槸浣犻渶瑕佽繘琛屽苟琛屽寲鐨勪富瑕侀儴鍒嗕簡锛岀壒鍒槸鍦ㄥ绾跨▼&GPU缂栫▼浠诲姟涓?
        // 鍙互鐪嬪埌锛岃繖涓惊鐜湰璐ㄤ笂灏辨槸鎶婃ā鍨嬩腑涓€涓猻egment鐨勬墍鏈塿alue锛岃祴鍊煎埌PT涓紝褰㈡垚涓€绯诲垪鏂扮殑鐚滄祴
        // 杩欎釜杩囩▼鏄彲浠ラ珮搴﹀苟琛屽寲鐨?
        appendSegmentValues("", a, pt.max_indices[0]);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 杩欎釜for寰幆鐨勪綔鐢細缁欏綋鍓峆T鐨勬墍鏈塻egment璧嬩簣瀹為檯鐨勫€硷紙鏈€鍚庝竴涓猻egment闄ゅ锛?
        // segment鍊兼牴鎹甤urr_indices涓搴旂殑鍊煎姞浠ョ‘瀹?
        // 杩欎釜for寰幆浣犵湅涓嶆噦涔熸病澶ぇ闂锛屽苟琛岀畻娉曚笉娑夊強杩欓噷鐨勫姞閫?
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

        // 鎸囧悜鏈€鍚庝竴涓猻egment鐨勬寚閽堬紝杩欎釜鎸囬拡瀹為檯鎸囧悜妯″瀷涓殑缁熻鏁版嵁
        segment *a = getSegmentPtr(pt.content[pt.content.size() - 1]);
        
        // Multi-thread TODO锛?
        // 杩欎釜for寰幆灏辨槸浣犻渶瑕佽繘琛屽苟琛屽寲鐨勪富瑕侀儴鍒嗕簡锛岀壒鍒槸鍦ㄥ绾跨▼&GPU缂栫▼浠诲姟涓?
        // 鍙互鐪嬪埌锛岃繖涓惊鐜湰璐ㄤ笂灏辨槸鎶婃ā鍨嬩腑涓€涓猻egment鐨勬墍鏈塿alue锛岃祴鍊煎埌PT涓紝褰㈡垚涓€绯诲垪鏂扮殑鐚滄祴
        // 杩欎釜杩囩▼鏄彲浠ラ珮搴﹀苟琛屽寲鐨?
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
    cout << "[HIPGenerateStats]" << endl;
    cout << "hip_generate_calls = " << hip_generate_calls << endl;
    cout << "hip_generate_items = " << hip_generate_items << endl;
    cout << "hip_generate_total_time_sec = " << hip_generate_total_time_sec << endl;
    cout << "hip_h2d_time_sec = " << hip_h2d_time_sec << endl;
    cout << "hip_kernel_time_sec = " << hip_kernel_time_sec << endl;
    cout << "hip_d2h_time_sec = " << hip_d2h_time_sec << endl;
    cout << "[CUDAGenerateStats]" << endl;
    cout << "cuda_generate_calls=" << cuda_generate_calls << endl;
    cout << "cuda_generate_items=" << cuda_generate_items << endl;
    cout << "cuda_generate_total_time_sec=" << cuda_generate_total_time_sec << endl;
    cout << "cuda_h2d_time_sec=" << cuda_h2d_time_sec << endl;
    cout << "cuda_kernel_time_sec=" << cuda_kernel_time_sec << endl;
    cout << "cuda_d2h_time_sec=" << cuda_d2h_time_sec << endl;
    cout << "[PopNextStats]" << endl;
#ifdef ENABLE_RELAXED_HEAP_PRIORITY
    cout << "priority_queue_mode = relaxed_heap" << endl;
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
