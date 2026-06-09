#include "gpu_generate_cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

struct RepeatResult
{
    size_t n = 0;
    int repeat = 0;
    double cpu_time_sec = 0.0;
    double cuda_total_time_sec = 0.0;
    double cuda_h2d_time_sec = 0.0;
    double cuda_kernel_time_sec = 0.0;
    double cuda_d2h_time_sec = 0.0;
    bool correct = false;
};

static string MakeValue(size_t i)
{
    ostringstream oss;
    oss << "pw" << setw(8) << setfill('0') << i;
    return oss.str();
}

static vector<string> MakeValues(size_t n)
{
    vector<string> values;
    values.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        values.emplace_back(MakeValue(i));
    }
    return values;
}

static uint64_t Fnv1a(const vector<string>& values)
{
    uint64_t hash = 1469598103934665603ull;
    for (const string& value : values)
    {
        for (unsigned char ch : value)
        {
            hash ^= ch;
            hash *= 1099511628211ull;
        }
        hash ^= 0xffu;
        hash *= 1099511628211ull;
    }
    return hash;
}

static vector<size_t> SampleIndices(size_t n)
{
    vector<size_t> indices;
    auto add_range = [&](size_t begin, size_t end)
    {
        for (size_t i = begin; i < end && i < n; ++i)
        {
            indices.push_back(i);
        }
    };

    add_range(0, min<size_t>(5, n));
    size_t mid = n / 2;
    size_t mid_begin = (mid >= 2) ? (mid - 2) : 0;
    add_range(mid_begin, min(n, mid_begin + 5));
    size_t tail_begin = (n > 5) ? (n - 5) : 0;
    add_range(tail_begin, n);

    sort(indices.begin(), indices.end());
    indices.erase(unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

static bool ValidateOutputs(
    const vector<string>& cpu_guesses,
    const vector<string>& cuda_guesses,
    size_t n,
    string& error)
{
    for (size_t idx : SampleIndices(n))
    {
        if (cpu_guesses[idx] != cuda_guesses[idx])
        {
            ostringstream oss;
            oss << "mismatch at index " << idx << ": CPU=" << cpu_guesses[idx]
                << ", CUDA=" << cuda_guesses[idx];
            error = oss.str();
            return false;
        }
    }

    uint64_t cpu_hash = Fnv1a(cpu_guesses);
    uint64_t cuda_hash = Fnv1a(cuda_guesses);
    if (cpu_hash != cuda_hash)
    {
        ostringstream oss;
        oss << "checksum mismatch: CPU=0x" << hex << cpu_hash
            << ", CUDA=0x" << cuda_hash;
        error = oss.str();
        return false;
    }

    return true;
}

static double Mean(const vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }
    return accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

static bool RunSweep(const string& label, const filesystem::path& csv_path, bool use_reuse)
{
    const string prefix = "PFX";
    const vector<size_t> sizes = {44, 1000, 10000, 100000, 1000000};
    const int repeats = 6;

    ofstream csv(csv_path);
    if (!csv)
    {
        cerr << "failed to open " << csv_path.string() << " for writing" << endl;
        return false;
    }

    csv << "n,repeat,cpu_time_sec,cuda_total_time_sec,cuda_h2d_time_sec,"
           "cuda_kernel_time_sec,cuda_d2h_time_sec,correct\n";
    csv << fixed << setprecision(9);

    CudaGenerateContext ctx;
    bool ctx_ready = true;
    if (use_reuse)
    {
        ctx_ready = InitCudaGenerateContext(&ctx);
        if (!ctx_ready)
        {
            cerr << label << " context initialization failed" << endl;
            return false;
        }
    }

    bool all_correct = true;
    cout << "[" << label << "]" << endl;

    for (size_t n : sizes)
    {
        vector<string> values;
        try
        {
            values = MakeValues(n);
        }
        catch (const bad_alloc&)
        {
            if (n == 1000000)
            {
                cerr << "n=" << n << " skipped: host allocation failed" << endl;
                continue;
            }
            throw;
        }

        vector<double> cpu_times;
        vector<double> cuda_total_times;
        vector<double> h2d_times;
        vector<double> kernel_times;
        vector<double> d2h_times;
        bool size_skipped = false;

        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            RepeatResult result;
            result.n = n;
            result.repeat = repeat;

            vector<string> cpu_guesses(n);
            auto cpu_start = Clock::now();
            for (size_t i = 0; i < n; ++i)
            {
                cpu_guesses[i] = prefix + values[i];
            }
            auto cpu_end = Clock::now();
            result.cpu_time_sec = chrono::duration<double>(cpu_end - cpu_start).count();

            vector<string> cuda_guesses(n);
            double h2d = 0.0;
            double kernel = 0.0;
            double d2h = 0.0;

            // End-to-end GPU generate path time. Original mode includes
            // per-call allocation/free; reuse mode keeps buffers alive but
            // still includes flattening, copies, kernel, D2H, and string rebuild.
            auto cuda_start = Clock::now();
            bool cuda_ok = use_reuse
                ? CudaGenerateSegmentValuesWithContext(
                    &ctx,
                    prefix,
                    values,
                    static_cast<int>(n),
                    cuda_guesses,
                    0,
                    &h2d,
                    &kernel,
                    &d2h)
                : CudaGenerateSegmentValues(
                    prefix,
                    values,
                    static_cast<int>(n),
                    cuda_guesses,
                    0,
                    &h2d,
                    &kernel,
                    &d2h);
            auto cuda_end = Clock::now();

            if (!cuda_ok)
            {
                if (n == 1000000)
                {
                    cerr << "n=" << n << " skipped: CUDA generate returned false" << endl;
                    size_skipped = true;
                    break;
                }
                cerr << "n=" << n << " repeat=" << repeat
                     << " failed: CUDA generate returned false" << endl;
                if (use_reuse)
                {
                    DestroyCudaGenerateContext(&ctx);
                }
                return false;
            }

            result.cuda_total_time_sec = chrono::duration<double>(cuda_end - cuda_start).count();
            result.cuda_h2d_time_sec = h2d;
            result.cuda_kernel_time_sec = kernel;
            result.cuda_d2h_time_sec = d2h;

            string error;
            result.correct = ValidateOutputs(cpu_guesses, cuda_guesses, n, error);
            if (!result.correct)
            {
                cerr << label << " n=" << n << " repeat=" << repeat
                     << " correctness FAILED: " << error << endl;
                if (use_reuse)
                {
                    DestroyCudaGenerateContext(&ctx);
                }
                return false;
            }

            if (repeat > 0)
            {
                csv << result.n << ','
                    << result.repeat << ','
                    << result.cpu_time_sec << ','
                    << result.cuda_total_time_sec << ','
                    << result.cuda_h2d_time_sec << ','
                    << result.cuda_kernel_time_sec << ','
                    << result.cuda_d2h_time_sec << ','
                    << (result.correct ? 1 : 0) << '\n';

                cpu_times.push_back(result.cpu_time_sec);
                cuda_total_times.push_back(result.cuda_total_time_sec);
                h2d_times.push_back(result.cuda_h2d_time_sec);
                kernel_times.push_back(result.cuda_kernel_time_sec);
                d2h_times.push_back(result.cuda_d2h_time_sec);
            }
        }

        if (size_skipped)
        {
            continue;
        }

        double cpu_mean = Mean(cpu_times);
        double cuda_mean = Mean(cuda_total_times);
        double speedup = (cuda_mean > 0.0) ? (cpu_mean / cuda_mean) : 0.0;
        cout << fixed << setprecision(6)
             << "n=" << n
             << " cpu=" << cpu_mean
             << " cuda_total=" << cuda_mean
             << " h2d=" << Mean(h2d_times)
             << " kernel=" << Mean(kernel_times)
             << " d2h=" << Mean(d2h_times)
             << " speedup=" << speedup
             << " correct=PASSED" << endl;
    }

    if (use_reuse)
    {
        cout << "reuse_alloc_calls=" << ctx.alloc_calls
             << " reuse_realloc_calls=" << ctx.realloc_calls << endl;
        DestroyCudaGenerateContext(&ctx);
    }

    cout << "CSV written to " << filesystem::absolute(csv_path).string() << endl;
    return all_correct;
}

int main()
{
    filesystem::create_directories("results");
    bool original_ok = RunSweep("original", filesystem::path("results") / "cuda_generate_sweep.csv", false);
    bool reuse_ok = RunSweep("reuse", filesystem::path("results") / "cuda_generate_reuse_summary.csv", true);

    cout << "44 items are correctness-only; 10000+ items are the meaningful performance trend region." << endl;
    cout << "CUDA kernel time is separated from end-to-end CUDA total time, which also includes fixed transfer and string rebuild overhead." << endl;

    return (original_ok && reuse_ok) ? 0 : 1;
}
