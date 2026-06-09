#ifndef GPU_GENERATE_CUDA_H
#define GPU_GENERATE_CUDA_H

#include <cstddef>
#include <string>
#include <vector>

#ifndef CUDA_GENERATE_THRESHOLD
#define CUDA_GENERATE_THRESHOLD 4096
#endif

#ifndef CUDA_GENERATE_MAX_GUESS_LEN
#define CUDA_GENERATE_MAX_GUESS_LEN 128
#endif

struct CudaGenerateContext
{
    char* d_prefix = nullptr;
    char* d_flat_values = nullptr;
    int* d_offsets = nullptr;
    int* d_lengths = nullptr;
    char* d_out_chars = nullptr;
    int* d_out_lengths = nullptr;
    size_t prefix_capacity = 0;
    size_t flat_values_capacity = 0;
    size_t n_capacity = 0;
    size_t out_chars_capacity = 0;
    long long alloc_calls = 0;
    long long realloc_calls = 0;
    bool initialized = false;
};

bool InitCudaGenerateContext(CudaGenerateContext* ctx);
void DestroyCudaGenerateContext(CudaGenerateContext* ctx);

bool CudaGenerateSegmentValues(
    const std::string& prefix,
    const std::vector<std::string>& values,
    int n,
    std::vector<std::string>& guesses,
    size_t base,
    double* h2d_time_sec,
    double* kernel_time_sec,
    double* d2h_time_sec
);

bool CudaGenerateSegmentValuesWithContext(
    CudaGenerateContext* ctx,
    const std::string& prefix,
    const std::vector<std::string>& values,
    int n,
    std::vector<std::string>& guesses,
    size_t base,
    double* h2d_time_sec,
    double* kernel_time_sec,
    double* d2h_time_sec
);

#endif
