#include "gpu_generate.h"

#include <hip/hip_runtime.h>

#include <limits>
#include <vector>

using namespace std;

__global__ void HipGenerateKernel(
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

    char* out = out_chars + static_cast<size_t>(idx) * HIP_GENERATE_MAX_GUESS_LEN;
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

bool HipGenerateSegmentValues(
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
    if (prefix.size() > HIP_GENERATE_MAX_GUESS_LEN)
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
        if (prefix.size() + value.size() > HIP_GENERATE_MAX_GUESS_LEN)
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
    hipEvent_t h2d_start = nullptr;
    hipEvent_t h2d_stop = nullptr;
    hipEvent_t kernel_start = nullptr;
    hipEvent_t kernel_stop = nullptr;
    hipEvent_t d2h_start = nullptr;
    hipEvent_t d2h_stop = nullptr;

    auto cleanup = [&]() -> bool
    {
        bool ok = true;
        if (d_prefix != nullptr && hipFree(d_prefix) != hipSuccess)
        {
            ok = false;
        }
        if (d_flat_values != nullptr && hipFree(d_flat_values) != hipSuccess)
        {
            ok = false;
        }
        if (d_offsets != nullptr && hipFree(d_offsets) != hipSuccess)
        {
            ok = false;
        }
        if (d_lengths != nullptr && hipFree(d_lengths) != hipSuccess)
        {
            ok = false;
        }
        if (d_out_chars != nullptr && hipFree(d_out_chars) != hipSuccess)
        {
            ok = false;
        }
        if (d_out_lengths != nullptr && hipFree(d_out_lengths) != hipSuccess)
        {
            ok = false;
        }
        if (h2d_start != nullptr && hipEventDestroy(h2d_start) != hipSuccess)
        {
            ok = false;
        }
        if (h2d_stop != nullptr && hipEventDestroy(h2d_stop) != hipSuccess)
        {
            ok = false;
        }
        if (kernel_start != nullptr && hipEventDestroy(kernel_start) != hipSuccess)
        {
            ok = false;
        }
        if (kernel_stop != nullptr && hipEventDestroy(kernel_stop) != hipSuccess)
        {
            ok = false;
        }
        if (d2h_start != nullptr && hipEventDestroy(d2h_start) != hipSuccess)
        {
            ok = false;
        }
        if (d2h_stop != nullptr && hipEventDestroy(d2h_stop) != hipSuccess)
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

    auto check = [](hipError_t status) -> bool
    {
        return status == hipSuccess;
    };

    const size_t prefix_bytes = prefix.size();
    const size_t flat_bytes = flat_values.size();
    const size_t int_bytes = static_cast<size_t>(n) * sizeof(int);
    const size_t out_chars_bytes = static_cast<size_t>(n) * HIP_GENERATE_MAX_GUESS_LEN;

    if (prefix_bytes > 0 && !check(hipMalloc(reinterpret_cast<void**>(&d_prefix), prefix_bytes)))
    {
        return fail();
    }
    if (flat_bytes > 0 && !check(hipMalloc(reinterpret_cast<void**>(&d_flat_values), flat_bytes)))
    {
        return fail();
    }
    if (!check(hipMalloc(reinterpret_cast<void**>(&d_offsets), int_bytes)) ||
        !check(hipMalloc(reinterpret_cast<void**>(&d_lengths), int_bytes)) ||
        !check(hipMalloc(reinterpret_cast<void**>(&d_out_chars), out_chars_bytes)) ||
        !check(hipMalloc(reinterpret_cast<void**>(&d_out_lengths), int_bytes)))
    {
        return fail();
    }

    if (!check(hipEventCreate(&h2d_start)) ||
        !check(hipEventCreate(&h2d_stop)) ||
        !check(hipEventCreate(&kernel_start)) ||
        !check(hipEventCreate(&kernel_stop)) ||
        !check(hipEventCreate(&d2h_start)) ||
        !check(hipEventCreate(&d2h_stop)))
    {
        return fail();
    }

    if (!check(hipEventRecord(h2d_start, 0)))
    {
        return fail();
    }
    if (prefix_bytes > 0 &&
        !check(hipMemcpy(d_prefix, prefix.data(), prefix_bytes, hipMemcpyHostToDevice)))
    {
        return fail();
    }
    if (flat_bytes > 0 &&
        !check(hipMemcpy(d_flat_values, flat_values.data(), flat_bytes, hipMemcpyHostToDevice)))
    {
        return fail();
    }
    if (!check(hipMemcpy(d_offsets, offsets.data(), int_bytes, hipMemcpyHostToDevice)) ||
        !check(hipMemcpy(d_lengths, lengths.data(), int_bytes, hipMemcpyHostToDevice)))
    {
        return fail();
    }
    if (!check(hipEventRecord(h2d_stop, 0)) ||
        !check(hipEventSynchronize(h2d_stop)))
    {
        return fail();
    }

    float elapsed_ms = 0.0f;
    if (!check(hipEventElapsedTime(&elapsed_ms, h2d_start, h2d_stop)))
    {
        return fail();
    }
    if (h2d_time_sec != nullptr)
    {
        *h2d_time_sec = elapsed_ms / 1000.0;
    }

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    if (!check(hipEventRecord(kernel_start, 0)))
    {
        return fail();
    }
    hipLaunchKernelGGL(
        HipGenerateKernel,
        dim3(blocks),
        dim3(threads_per_block),
        0,
        0,
        d_prefix,
        static_cast<int>(prefix.size()),
        d_flat_values,
        d_offsets,
        d_lengths,
        n,
        d_out_chars,
        d_out_lengths);
    if (!check(hipGetLastError()) ||
        !check(hipEventRecord(kernel_stop, 0)) ||
        !check(hipDeviceSynchronize()) ||
        !check(hipEventSynchronize(kernel_stop)))
    {
        return fail();
    }
    if (!check(hipEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop)))
    {
        return fail();
    }
    if (kernel_time_sec != nullptr)
    {
        *kernel_time_sec = elapsed_ms / 1000.0;
    }

    vector<char> out_chars(out_chars_bytes);
    vector<int> out_lengths(static_cast<size_t>(n));
    if (!check(hipEventRecord(d2h_start, 0)))
    {
        return fail();
    }
    if (!check(hipMemcpy(out_chars.data(), d_out_chars, out_chars_bytes, hipMemcpyDeviceToHost)) ||
        !check(hipMemcpy(out_lengths.data(), d_out_lengths, int_bytes, hipMemcpyDeviceToHost)))
    {
        return fail();
    }
    if (!check(hipEventRecord(d2h_stop, 0)) ||
        !check(hipEventSynchronize(d2h_stop)))
    {
        return fail();
    }
    if (!check(hipEventElapsedTime(&elapsed_ms, d2h_start, d2h_stop)))
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
        if (len < 0 || len > HIP_GENERATE_MAX_GUESS_LEN)
        {
            return fail();
        }
        const char* start = out_chars.data() + static_cast<size_t>(i) * HIP_GENERATE_MAX_GUESS_LEN;
        guesses[base + static_cast<size_t>(i)].assign(start, static_cast<size_t>(len));
    }

    return cleanup();
}
