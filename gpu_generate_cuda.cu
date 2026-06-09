#include "gpu_generate_cuda.h"

#include <cuda_runtime.h>

#include <limits>
#include <vector>

using namespace std;

__global__ void CudaGenerateKernel(
    const char* prefix,
    int prefix_len,
    const char* flat_values,
    const int* offsets,
    const int* lengths,
    int n,
    char* out_chars,
    int* out_lengths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    char* out = out_chars + static_cast<size_t>(idx) * CUDA_GENERATE_MAX_GUESS_LEN;
    for (int j = 0; j < prefix_len; ++j)
    {
        out[j] = prefix[j];
    }

    int value_len = lengths[idx];
    int value_offset = offsets[idx];
    for (int j = 0; j < value_len; ++j)
    {
        out[prefix_len + j] = flat_values[value_offset + j];
    }

    out_lengths[idx] = prefix_len + value_len;
}

bool CudaGenerateSegmentValues(
    const string& prefix,
    const vector<string>& values,
    int n,
    vector<string>& guesses,
    size_t base,
    double* h2d_time_sec,
    double* kernel_time_sec,
    double* d2h_time_sec)
{
    if (h2d_time_sec != nullptr)
    {
        *h2d_time_sec = 0.0;
    }
    if (kernel_time_sec != nullptr)
    {
        *kernel_time_sec = 0.0;
    }
    if (d2h_time_sec != nullptr)
    {
        *d2h_time_sec = 0.0;
    }

    if (n <= 0)
    {
        return true;
    }
    if (static_cast<size_t>(n) > values.size())
    {
        return false;
    }
    if (prefix.size() > CUDA_GENERATE_MAX_GUESS_LEN)
    {
        return false;
    }
    if (base > guesses.size() || guesses.size() - base < static_cast<size_t>(n))
    {
        return false;
    }
    if (prefix.size() > static_cast<size_t>(numeric_limits<int>::max()))
    {
        return false;
    }

    vector<int> offsets(static_cast<size_t>(n));
    vector<int> lengths(static_cast<size_t>(n));
    vector<char> flat_values;

    size_t flat_total = 0;
    for (int i = 0; i < n; ++i)
    {
        const string& value = values[static_cast<size_t>(i)];
        if (prefix.size() + value.size() > CUDA_GENERATE_MAX_GUESS_LEN)
        {
            return false;
        }
        if (value.size() > static_cast<size_t>(numeric_limits<int>::max()) ||
            flat_total > static_cast<size_t>(numeric_limits<int>::max()) ||
            flat_total + value.size() > static_cast<size_t>(numeric_limits<int>::max()))
        {
            return false;
        }
        offsets[static_cast<size_t>(i)] = static_cast<int>(flat_total);
        lengths[static_cast<size_t>(i)] = static_cast<int>(value.size());
        flat_total += value.size();
    }

    flat_values.reserve(flat_total);
    for (int i = 0; i < n; ++i)
    {
        const string& value = values[static_cast<size_t>(i)];
        flat_values.insert(flat_values.end(), value.begin(), value.end());
    }

    char* d_prefix = nullptr;
    char* d_flat_values = nullptr;
    int* d_offsets = nullptr;
    int* d_lengths = nullptr;
    char* d_out_chars = nullptr;
    int* d_out_lengths = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_stop = nullptr;

    auto cleanup = [&]() -> bool
    {
        bool ok = true;
        if (d_prefix != nullptr && cudaFree(d_prefix) != cudaSuccess)
        {
            ok = false;
        }
        if (d_flat_values != nullptr && cudaFree(d_flat_values) != cudaSuccess)
        {
            ok = false;
        }
        if (d_offsets != nullptr && cudaFree(d_offsets) != cudaSuccess)
        {
            ok = false;
        }
        if (d_lengths != nullptr && cudaFree(d_lengths) != cudaSuccess)
        {
            ok = false;
        }
        if (d_out_chars != nullptr && cudaFree(d_out_chars) != cudaSuccess)
        {
            ok = false;
        }
        if (d_out_lengths != nullptr && cudaFree(d_out_lengths) != cudaSuccess)
        {
            ok = false;
        }
        if (h2d_start != nullptr && cudaEventDestroy(h2d_start) != cudaSuccess)
        {
            ok = false;
        }
        if (h2d_stop != nullptr && cudaEventDestroy(h2d_stop) != cudaSuccess)
        {
            ok = false;
        }
        if (kernel_start != nullptr && cudaEventDestroy(kernel_start) != cudaSuccess)
        {
            ok = false;
        }
        if (kernel_stop != nullptr && cudaEventDestroy(kernel_stop) != cudaSuccess)
        {
            ok = false;
        }
        if (d2h_start != nullptr && cudaEventDestroy(d2h_start) != cudaSuccess)
        {
            ok = false;
        }
        if (d2h_stop != nullptr && cudaEventDestroy(d2h_stop) != cudaSuccess)
        {
            ok = false;
        }
        return ok;
    };

    auto fail = [&]() -> bool
    {
        cleanup();
        return false;
    };

    auto check = [](cudaError_t status) -> bool
    {
        if (status != cudaSuccess)
        {
            cudaGetErrorString(status);
            return false;
        }
        return true;
    };

    const size_t prefix_bytes = prefix.size();
    const size_t flat_bytes = flat_values.size();
    const size_t int_bytes = static_cast<size_t>(n) * sizeof(int);
    const size_t out_chars_bytes = static_cast<size_t>(n) * CUDA_GENERATE_MAX_GUESS_LEN;

    if (prefix_bytes > 0 && !check(cudaMalloc(reinterpret_cast<void**>(&d_prefix), prefix_bytes)))
    {
        return fail();
    }
    if (flat_bytes > 0 && !check(cudaMalloc(reinterpret_cast<void**>(&d_flat_values), flat_bytes)))
    {
        return fail();
    }
    if (!check(cudaMalloc(reinterpret_cast<void**>(&d_offsets), int_bytes)) ||
        !check(cudaMalloc(reinterpret_cast<void**>(&d_lengths), int_bytes)) ||
        !check(cudaMalloc(reinterpret_cast<void**>(&d_out_chars), out_chars_bytes)) ||
        !check(cudaMalloc(reinterpret_cast<void**>(&d_out_lengths), int_bytes)))
    {
        return fail();
    }

    if (!check(cudaEventCreate(&h2d_start)) ||
        !check(cudaEventCreate(&h2d_stop)) ||
        !check(cudaEventCreate(&kernel_start)) ||
        !check(cudaEventCreate(&kernel_stop)) ||
        !check(cudaEventCreate(&d2h_start)) ||
        !check(cudaEventCreate(&d2h_stop)))
    {
        return fail();
    }

    if (!check(cudaEventRecord(h2d_start, 0)))
    {
        return fail();
    }
    if (prefix_bytes > 0 &&
        !check(cudaMemcpy(d_prefix, prefix.data(), prefix_bytes, cudaMemcpyHostToDevice)))
    {
        return fail();
    }
    if (flat_bytes > 0 &&
        !check(cudaMemcpy(d_flat_values, flat_values.data(), flat_bytes, cudaMemcpyHostToDevice)))
    {
        return fail();
    }
    if (!check(cudaMemcpy(d_offsets, offsets.data(), int_bytes, cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(d_lengths, lengths.data(), int_bytes, cudaMemcpyHostToDevice)))
    {
        return fail();
    }
    if (!check(cudaEventRecord(h2d_stop, 0)) ||
        !check(cudaEventSynchronize(h2d_stop)))
    {
        return fail();
    }

    float elapsed_ms = 0.0f;
    if (!check(cudaEventElapsedTime(&elapsed_ms, h2d_start, h2d_stop)))
    {
        return fail();
    }
    if (h2d_time_sec != nullptr)
    {
        *h2d_time_sec = elapsed_ms / 1000.0;
    }

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    if (!check(cudaEventRecord(kernel_start, 0)))
    {
        return fail();
    }
    CudaGenerateKernel<<<blocks, threads_per_block>>>(
        d_prefix,
        static_cast<int>(prefix.size()),
        d_flat_values,
        d_offsets,
        d_lengths,
        n,
        d_out_chars,
        d_out_lengths);
    if (!check(cudaGetLastError()) ||
        !check(cudaEventRecord(kernel_stop, 0)) ||
        !check(cudaDeviceSynchronize()) ||
        !check(cudaEventSynchronize(kernel_stop)))
    {
        return fail();
    }
    if (!check(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop)))
    {
        return fail();
    }
    if (kernel_time_sec != nullptr)
    {
        *kernel_time_sec = elapsed_ms / 1000.0;
    }

    vector<char> out_chars(out_chars_bytes);
    vector<int> out_lengths(static_cast<size_t>(n));
    if (!check(cudaEventRecord(d2h_start, 0)))
    {
        return fail();
    }
    if (!check(cudaMemcpy(out_chars.data(), d_out_chars, out_chars_bytes, cudaMemcpyDeviceToHost)) ||
        !check(cudaMemcpy(out_lengths.data(), d_out_lengths, int_bytes, cudaMemcpyDeviceToHost)))
    {
        return fail();
    }
    if (!check(cudaEventRecord(d2h_stop, 0)) ||
        !check(cudaEventSynchronize(d2h_stop)))
    {
        return fail();
    }
    if (!check(cudaEventElapsedTime(&elapsed_ms, d2h_start, d2h_stop)))
    {
        return fail();
    }
    if (d2h_time_sec != nullptr)
    {
        *d2h_time_sec = elapsed_ms / 1000.0;
    }

    for (int i = 0; i < n; ++i)
    {
        int len = out_lengths[static_cast<size_t>(i)];
        if (len < 0 || len > CUDA_GENERATE_MAX_GUESS_LEN)
        {
            return fail();
        }
        const char* start = out_chars.data() + static_cast<size_t>(i) * CUDA_GENERATE_MAX_GUESS_LEN;
        guesses[base + static_cast<size_t>(i)].assign(start, static_cast<size_t>(len));
    }

    return cleanup();
}
