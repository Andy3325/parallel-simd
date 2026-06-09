#include "gpu_generate_cuda.h"

#include <cuda_runtime.h>

#include <cstring>
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

static bool CheckCuda(cudaError_t status)
{
    return status == cudaSuccess;
}

static bool ReleaseBuffer(void** ptr, size_t* capacity)
{
    bool ok = true;
    if (*ptr != nullptr)
    {
        ok = CheckCuda(cudaFree(*ptr));
    }
    *ptr = nullptr;
    *capacity = 0;
    return ok;
}

static bool EnsureBytes(void** ptr, size_t* capacity, size_t required, CudaGenerateContext* ctx)
{
    if (required == 0 || *capacity >= required)
    {
        return true;
    }

    if (*ptr != nullptr && !CheckCuda(cudaFree(*ptr)))
    {
        *ptr = nullptr;
        *capacity = 0;
        return false;
    }

    *ptr = nullptr;
    *capacity = 0;
    if (!CheckCuda(cudaMalloc(ptr, required)))
    {
        return false;
    }

    *capacity = required;
    ctx->alloc_calls += 1;
    ctx->realloc_calls += 1;
    return true;
}

bool InitCudaGenerateContext(CudaGenerateContext* ctx)
{
    if (ctx == nullptr)
    {
        return false;
    }
    memset(ctx, 0, sizeof(CudaGenerateContext));
    ctx->initialized = true;
    return true;
}

void DestroyCudaGenerateContext(CudaGenerateContext* ctx)
{
    if (ctx == nullptr)
    {
        return;
    }

    ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_prefix), &ctx->prefix_capacity);
    ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_flat_values), &ctx->flat_values_capacity);
    size_t n_bytes_capacity = ctx->n_capacity * sizeof(int);
    ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_offsets), &n_bytes_capacity);
    n_bytes_capacity = ctx->n_capacity * sizeof(int);
    ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_lengths), &n_bytes_capacity);
    ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_out_chars), &ctx->out_chars_capacity);
    n_bytes_capacity = ctx->n_capacity * sizeof(int);
    ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_out_lengths), &n_bytes_capacity);

    ctx->n_capacity = 0;
    ctx->initialized = false;
}

static bool EnsureContextCapacity(
    CudaGenerateContext* ctx,
    size_t prefix_bytes,
    size_t flat_bytes,
    size_t n,
    size_t out_chars_bytes)
{
    if (!EnsureBytes(reinterpret_cast<void**>(&ctx->d_prefix), &ctx->prefix_capacity, prefix_bytes, ctx))
    {
        return false;
    }
    if (!EnsureBytes(reinterpret_cast<void**>(&ctx->d_flat_values), &ctx->flat_values_capacity, flat_bytes, ctx))
    {
        return false;
    }
    if (ctx->n_capacity < n)
    {
        size_t offsets_capacity = ctx->n_capacity * sizeof(int);
        size_t lengths_capacity = ctx->n_capacity * sizeof(int);
        size_t out_lengths_capacity = ctx->n_capacity * sizeof(int);
        const size_t int_bytes = n * sizeof(int);

        if (!ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_offsets), &offsets_capacity) ||
            !ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_lengths), &lengths_capacity) ||
            !ReleaseBuffer(reinterpret_cast<void**>(&ctx->d_out_lengths), &out_lengths_capacity))
        {
            ctx->n_capacity = 0;
            return false;
        }
        ctx->n_capacity = 0;

        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&ctx->d_offsets), int_bytes)) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&ctx->d_lengths), int_bytes)) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&ctx->d_out_lengths), int_bytes)))
        {
            return false;
        }

        ctx->n_capacity = n;
        ctx->alloc_calls += 3;
        ctx->realloc_calls += 3;
    }
    if (!EnsureBytes(reinterpret_cast<void**>(&ctx->d_out_chars), &ctx->out_chars_capacity, out_chars_bytes, ctx))
    {
        return false;
    }
    return true;
}

bool CudaGenerateSegmentValuesWithContext(
    CudaGenerateContext* ctx,
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

    if (ctx == nullptr || !ctx->initialized)
    {
        return false;
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

    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_stop = nullptr;

    auto cleanup = [&]() -> bool
    {
        bool ok = true;
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

    const size_t prefix_bytes = prefix.size();
    const size_t flat_bytes = flat_values.size();
    const size_t int_bytes = static_cast<size_t>(n) * sizeof(int);
    const size_t out_chars_bytes = static_cast<size_t>(n) * CUDA_GENERATE_MAX_GUESS_LEN;

    if (!EnsureContextCapacity(ctx, prefix_bytes, flat_bytes, static_cast<size_t>(n), out_chars_bytes))
    {
        return fail();
    }

    if (!CheckCuda(cudaEventCreate(&h2d_start)) ||
        !CheckCuda(cudaEventCreate(&h2d_stop)) ||
        !CheckCuda(cudaEventCreate(&kernel_start)) ||
        !CheckCuda(cudaEventCreate(&kernel_stop)) ||
        !CheckCuda(cudaEventCreate(&d2h_start)) ||
        !CheckCuda(cudaEventCreate(&d2h_stop)))
    {
        return fail();
    }

    if (!CheckCuda(cudaEventRecord(h2d_start, 0)))
    {
        return fail();
    }
    if (prefix_bytes > 0 &&
        !CheckCuda(cudaMemcpy(ctx->d_prefix, prefix.data(), prefix_bytes, cudaMemcpyHostToDevice)))
    {
        return fail();
    }
    if (flat_bytes > 0 &&
        !CheckCuda(cudaMemcpy(ctx->d_flat_values, flat_values.data(), flat_bytes, cudaMemcpyHostToDevice)))
    {
        return fail();
    }
    if (!CheckCuda(cudaMemcpy(ctx->d_offsets, offsets.data(), int_bytes, cudaMemcpyHostToDevice)) ||
        !CheckCuda(cudaMemcpy(ctx->d_lengths, lengths.data(), int_bytes, cudaMemcpyHostToDevice)))
    {
        return fail();
    }
    if (!CheckCuda(cudaEventRecord(h2d_stop, 0)) ||
        !CheckCuda(cudaEventSynchronize(h2d_stop)))
    {
        return fail();
    }

    float elapsed_ms = 0.0f;
    if (!CheckCuda(cudaEventElapsedTime(&elapsed_ms, h2d_start, h2d_stop)))
    {
        return fail();
    }
    if (h2d_time_sec != nullptr)
    {
        *h2d_time_sec = elapsed_ms / 1000.0;
    }

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    if (!CheckCuda(cudaEventRecord(kernel_start, 0)))
    {
        return fail();
    }
    CudaGenerateKernel<<<blocks, threads_per_block>>>(
        ctx->d_prefix,
        static_cast<int>(prefix.size()),
        ctx->d_flat_values,
        ctx->d_offsets,
        ctx->d_lengths,
        n,
        ctx->d_out_chars,
        ctx->d_out_lengths);
    if (!CheckCuda(cudaGetLastError()) ||
        !CheckCuda(cudaEventRecord(kernel_stop, 0)) ||
        !CheckCuda(cudaDeviceSynchronize()) ||
        !CheckCuda(cudaEventSynchronize(kernel_stop)))
    {
        return fail();
    }
    if (!CheckCuda(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop)))
    {
        return fail();
    }
    if (kernel_time_sec != nullptr)
    {
        *kernel_time_sec = elapsed_ms / 1000.0;
    }

    vector<char> out_chars(out_chars_bytes);
    vector<int> out_lengths(static_cast<size_t>(n));
    if (!CheckCuda(cudaEventRecord(d2h_start, 0)))
    {
        return fail();
    }
    if (!CheckCuda(cudaMemcpy(out_chars.data(), ctx->d_out_chars, out_chars_bytes, cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(out_lengths.data(), ctx->d_out_lengths, int_bytes, cudaMemcpyDeviceToHost)))
    {
        return fail();
    }
    if (!CheckCuda(cudaEventRecord(d2h_stop, 0)) ||
        !CheckCuda(cudaEventSynchronize(d2h_stop)))
    {
        return fail();
    }
    if (!CheckCuda(cudaEventElapsedTime(&elapsed_ms, d2h_start, d2h_stop)))
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
    CudaGenerateContext ctx;
    if (!InitCudaGenerateContext(&ctx))
    {
        return false;
    }
    bool ok = CudaGenerateSegmentValuesWithContext(
        &ctx,
        prefix,
        values,
        n,
        guesses,
        base,
        h2d_time_sec,
        kernel_time_sec,
        d2h_time_sec);
    DestroyCudaGenerateContext(&ctx);
    return ok;
}
