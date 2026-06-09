#include "gpu_generate_cuda.h"

#include <cuda_runtime.h>

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

struct BatchTask
{
    string prefix;
    vector<string> values;
};

struct DeviceBatch
{
    vector<char> flat_prefixes;
    vector<int> prefix_offsets;
    vector<int> prefix_lengths;
    vector<char> flat_values;
    vector<int> value_offsets;
    vector<int> value_lengths;
    vector<int> task_value_start;
    vector<int> task_value_count;
    vector<int> task_item_start;
};

__global__ void CudaGenerateBatchKernel(
    const char* flat_prefixes,
    const int* prefix_offsets,
    const int* prefix_lengths,
    const char* flat_values,
    const int* value_offsets,
    const int* value_lengths,
    const int* task_value_start,
    const int* task_value_count,
    const int* task_item_start,
    int num_tasks,
    int total_items,
    char* out_chars,
    int* out_lengths)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total_items)
    {
        return;
    }

    int task_idx = 0;
    for (int t = 0; t < num_tasks; ++t)
    {
        int begin = task_item_start[t];
        int end = begin + task_value_count[t];
        if (global_idx >= begin && global_idx < end)
        {
            task_idx = t;
            break;
        }
    }

    int local_idx = global_idx - task_item_start[task_idx];
    int value_idx = task_value_start[task_idx] + local_idx;
    int prefix_len = prefix_lengths[task_idx];
    int prefix_offset = prefix_offsets[task_idx];
    int value_len = value_lengths[value_idx];
    int value_offset = value_offsets[value_idx];
    char* out = out_chars + static_cast<size_t>(global_idx) * CUDA_GENERATE_MAX_GUESS_LEN;

    for (int i = 0; i < prefix_len; ++i)
    {
        out[i] = flat_prefixes[prefix_offset + i];
    }
    for (int i = 0; i < value_len; ++i)
    {
        out[prefix_len + i] = flat_values[value_offset + i];
    }
    out_lengths[global_idx] = prefix_len + value_len;
}

static bool CheckCuda(cudaError_t status)
{
    return status == cudaSuccess;
}

static string MakeValue(int task_idx, int value_idx)
{
    ostringstream oss;
    oss << "pw" << setw(2) << setfill('0') << task_idx
        << "_" << setw(8) << setfill('0') << value_idx;
    return oss.str();
}

static vector<BatchTask> MakeTasks(int num_tasks, int values_per_task)
{
    vector<BatchTask> tasks;
    tasks.reserve(static_cast<size_t>(num_tasks));
    for (int t = 0; t < num_tasks; ++t)
    {
        BatchTask task;
        ostringstream prefix;
        prefix << "PT" << setw(2) << setfill('0') << t << "_";
        task.prefix = prefix.str();
        task.values.reserve(static_cast<size_t>(values_per_task));
        for (int i = 0; i < values_per_task; ++i)
        {
            task.values.emplace_back(MakeValue(t, i));
        }
        tasks.push_back(std::move(task));
    }
    return tasks;
}

static DeviceBatch FlattenTasks(const vector<BatchTask>& tasks)
{
    DeviceBatch batch;
    int value_start = 0;
    int item_start = 0;

    for (const BatchTask& task : tasks)
    {
        batch.prefix_offsets.push_back(static_cast<int>(batch.flat_prefixes.size()));
        batch.prefix_lengths.push_back(static_cast<int>(task.prefix.size()));
        batch.flat_prefixes.insert(batch.flat_prefixes.end(), task.prefix.begin(), task.prefix.end());

        batch.task_value_start.push_back(value_start);
        batch.task_value_count.push_back(static_cast<int>(task.values.size()));
        batch.task_item_start.push_back(item_start);

        for (const string& value : task.values)
        {
            batch.value_offsets.push_back(static_cast<int>(batch.flat_values.size()));
            batch.value_lengths.push_back(static_cast<int>(value.size()));
            batch.flat_values.insert(batch.flat_values.end(), value.begin(), value.end());
        }

        value_start += static_cast<int>(task.values.size());
        item_start += static_cast<int>(task.values.size());
    }

    return batch;
}

static vector<string> GenerateCpu(const vector<BatchTask>& tasks)
{
    vector<string> output;
    for (const BatchTask& task : tasks)
    {
        for (const string& value : task.values)
        {
            output.push_back(task.prefix + value);
        }
    }
    return output;
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

static bool ValidateOutputs(const vector<string>& cpu, const vector<string>& gpu, string& error)
{
    for (size_t idx : SampleIndices(cpu.size()))
    {
        if (cpu[idx] != gpu[idx])
        {
            ostringstream oss;
            oss << "mismatch at index " << idx << ": CPU=" << cpu[idx] << ", CUDA=" << gpu[idx];
            error = oss.str();
            return false;
        }
    }
    uint64_t cpu_hash = Fnv1a(cpu);
    uint64_t gpu_hash = Fnv1a(gpu);
    if (cpu_hash != gpu_hash)
    {
        ostringstream oss;
        oss << "checksum mismatch: CPU=0x" << hex << cpu_hash << ", CUDA=0x" << gpu_hash;
        error = oss.str();
        return false;
    }
    return true;
}

template <typename T>
static bool CopyVectorToDevice(const vector<T>& src, T** dst)
{
    if (src.empty())
    {
        *dst = nullptr;
        return true;
    }
    size_t bytes = src.size() * sizeof(src[0]);
    return CheckCuda(cudaMalloc(reinterpret_cast<void**>(dst), bytes)) &&
           CheckCuda(cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice));
}

static bool RunCudaBatch(
    const vector<BatchTask>& tasks,
    vector<string>& output,
    double* h2d_time_sec,
    double* kernel_time_sec,
    double* d2h_time_sec)
{
    DeviceBatch batch = FlattenTasks(tasks);
    int num_tasks = static_cast<int>(tasks.size());
    int total_items = 0;
    for (const BatchTask& task : tasks)
    {
        total_items += static_cast<int>(task.values.size());
    }

    char* d_flat_prefixes = nullptr;
    int* d_prefix_offsets = nullptr;
    int* d_prefix_lengths = nullptr;
    char* d_flat_values = nullptr;
    int* d_value_offsets = nullptr;
    int* d_value_lengths = nullptr;
    int* d_task_value_start = nullptr;
    int* d_task_value_count = nullptr;
    int* d_task_item_start = nullptr;
    char* d_out_chars = nullptr;
    int* d_out_lengths = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_stop = nullptr;

    auto cleanup = [&]()
    {
        cudaFree(d_flat_prefixes);
        cudaFree(d_prefix_offsets);
        cudaFree(d_prefix_lengths);
        cudaFree(d_flat_values);
        cudaFree(d_value_offsets);
        cudaFree(d_value_lengths);
        cudaFree(d_task_value_start);
        cudaFree(d_task_value_count);
        cudaFree(d_task_item_start);
        cudaFree(d_out_chars);
        cudaFree(d_out_lengths);
        if (h2d_start != nullptr) cudaEventDestroy(h2d_start);
        if (h2d_stop != nullptr) cudaEventDestroy(h2d_stop);
        if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
        if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
        if (d2h_start != nullptr) cudaEventDestroy(d2h_start);
        if (d2h_stop != nullptr) cudaEventDestroy(d2h_stop);
    };

    auto fail = [&]() -> bool
    {
        cleanup();
        return false;
    };

    if (!CheckCuda(cudaEventCreate(&h2d_start)) ||
        !CheckCuda(cudaEventCreate(&h2d_stop)) ||
        !CheckCuda(cudaEventCreate(&kernel_start)) ||
        !CheckCuda(cudaEventCreate(&kernel_stop)) ||
        !CheckCuda(cudaEventCreate(&d2h_start)) ||
        !CheckCuda(cudaEventCreate(&d2h_stop)))
    {
        return fail();
    }

    const size_t out_chars_bytes = static_cast<size_t>(total_items) * CUDA_GENERATE_MAX_GUESS_LEN;
    const size_t out_lengths_bytes = static_cast<size_t>(total_items) * sizeof(int);

    if (!CheckCuda(cudaEventRecord(h2d_start, 0)) ||
        !CopyVectorToDevice(batch.flat_prefixes, &d_flat_prefixes) ||
        !CopyVectorToDevice(batch.prefix_offsets, &d_prefix_offsets) ||
        !CopyVectorToDevice(batch.prefix_lengths, &d_prefix_lengths) ||
        !CopyVectorToDevice(batch.flat_values, &d_flat_values) ||
        !CopyVectorToDevice(batch.value_offsets, &d_value_offsets) ||
        !CopyVectorToDevice(batch.value_lengths, &d_value_lengths) ||
        !CopyVectorToDevice(batch.task_value_start, &d_task_value_start) ||
        !CopyVectorToDevice(batch.task_value_count, &d_task_value_count) ||
        !CopyVectorToDevice(batch.task_item_start, &d_task_item_start) ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_out_chars), out_chars_bytes)) ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_out_lengths), out_lengths_bytes)) ||
        !CheckCuda(cudaEventRecord(h2d_stop, 0)) ||
        !CheckCuda(cudaEventSynchronize(h2d_stop)))
    {
        return fail();
    }

    float elapsed_ms = 0.0f;
    if (!CheckCuda(cudaEventElapsedTime(&elapsed_ms, h2d_start, h2d_stop)))
    {
        return fail();
    }
    *h2d_time_sec = elapsed_ms / 1000.0;

    int threads_per_block = 256;
    int blocks = (total_items + threads_per_block - 1) / threads_per_block;
    if (!CheckCuda(cudaEventRecord(kernel_start, 0)))
    {
        return fail();
    }
    CudaGenerateBatchKernel<<<blocks, threads_per_block>>>(
        d_flat_prefixes,
        d_prefix_offsets,
        d_prefix_lengths,
        d_flat_values,
        d_value_offsets,
        d_value_lengths,
        d_task_value_start,
        d_task_value_count,
        d_task_item_start,
        num_tasks,
        total_items,
        d_out_chars,
        d_out_lengths);
    if (!CheckCuda(cudaGetLastError()) ||
        !CheckCuda(cudaEventRecord(kernel_stop, 0)) ||
        !CheckCuda(cudaDeviceSynchronize()) ||
        !CheckCuda(cudaEventSynchronize(kernel_stop)) ||
        !CheckCuda(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop)))
    {
        return fail();
    }
    *kernel_time_sec = elapsed_ms / 1000.0;

    vector<char> out_chars(out_chars_bytes);
    vector<int> out_lengths(static_cast<size_t>(total_items));
    if (!CheckCuda(cudaEventRecord(d2h_start, 0)) ||
        !CheckCuda(cudaMemcpy(out_chars.data(), d_out_chars, out_chars_bytes, cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(out_lengths.data(), d_out_lengths, out_lengths_bytes, cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaEventRecord(d2h_stop, 0)) ||
        !CheckCuda(cudaEventSynchronize(d2h_stop)) ||
        !CheckCuda(cudaEventElapsedTime(&elapsed_ms, d2h_start, d2h_stop)))
    {
        return fail();
    }
    *d2h_time_sec = elapsed_ms / 1000.0;

    output.resize(static_cast<size_t>(total_items));
    for (int i = 0; i < total_items; ++i)
    {
        int len = out_lengths[static_cast<size_t>(i)];
        if (len < 0 || len > CUDA_GENERATE_MAX_GUESS_LEN)
        {
            return fail();
        }
        const char* start = out_chars.data() + static_cast<size_t>(i) * CUDA_GENERATE_MAX_GUESS_LEN;
        output[static_cast<size_t>(i)].assign(start, static_cast<size_t>(len));
    }

    cleanup();
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

int main()
{
    const vector<int> task_counts = {1, 4, 16, 64};
    const vector<int> values_per_task_options = {1000, 10000};
    const int repeats = 5;
    filesystem::create_directories("results");
    filesystem::path csv_path = filesystem::path("results") / "cuda_generate_batch_sweep.csv";
    ofstream csv(csv_path);
    if (!csv)
    {
        cerr << "failed to open " << csv_path.string() << endl;
        return 1;
    }

    csv << "num_tasks,values_per_task,total_items,repeat,cpu_time_sec,cuda_total_time_sec,"
           "cuda_h2d_time_sec,cuda_kernel_time_sec,cuda_d2h_time_sec,correct\n";
    csv << fixed << setprecision(9);
    cout << fixed << setprecision(6);

    for (int num_tasks : task_counts)
    {
        for (int values_per_task : values_per_task_options)
        {
            vector<double> cpu_times;
            vector<double> cuda_times;
            vector<double> h2d_times;
            vector<double> kernel_times;
            vector<double> d2h_times;
            int total_items = num_tasks * values_per_task;

            for (int repeat = 1; repeat <= repeats; ++repeat)
            {
                vector<BatchTask> tasks = MakeTasks(num_tasks, values_per_task);
                auto cpu_start = Clock::now();
                vector<string> cpu_output = GenerateCpu(tasks);
                auto cpu_end = Clock::now();
                double cpu_time = chrono::duration<double>(cpu_end - cpu_start).count();

                vector<string> cuda_output;
                double h2d = 0.0;
                double kernel = 0.0;
                double d2h = 0.0;
                auto cuda_start = Clock::now();
                bool cuda_ok = RunCudaBatch(tasks, cuda_output, &h2d, &kernel, &d2h);
                auto cuda_end = Clock::now();
                if (!cuda_ok)
                {
                    cerr << "batch CUDA failed for num_tasks=" << num_tasks
                         << " values_per_task=" << values_per_task << endl;
                    return 1;
                }

                string error;
                bool correct = ValidateOutputs(cpu_output, cuda_output, error);
                if (!correct)
                {
                    cerr << "batch correctness FAILED: " << error << endl;
                    return 1;
                }

                double cuda_total = chrono::duration<double>(cuda_end - cuda_start).count();
                csv << num_tasks << ','
                    << values_per_task << ','
                    << total_items << ','
                    << repeat << ','
                    << cpu_time << ','
                    << cuda_total << ','
                    << h2d << ','
                    << kernel << ','
                    << d2h << ','
                    << (correct ? 1 : 0) << '\n';

                cpu_times.push_back(cpu_time);
                cuda_times.push_back(cuda_total);
                h2d_times.push_back(h2d);
                kernel_times.push_back(kernel);
                d2h_times.push_back(d2h);
            }

            double cuda_mean = Mean(cuda_times);
            double cpu_mean = Mean(cpu_times);
            double speedup = cuda_mean > 0.0 ? cpu_mean / cuda_mean : 0.0;
            cout << "num_tasks=" << num_tasks
                 << " values_per_task=" << values_per_task
                 << " total_items=" << total_items
                 << " cpu=" << cpu_mean
                 << " cuda_total=" << cuda_mean
                 << " h2d=" << Mean(h2d_times)
                 << " kernel=" << Mean(kernel_times)
                 << " d2h=" << Mean(d2h_times)
                 << " speedup=" << speedup
                 << " correct=PASSED" << endl;
        }
    }

    cout << "CSV written to " << filesystem::absolute(csv_path).string() << endl;
    return 0;
}
