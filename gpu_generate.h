#ifndef GPU_GENERATE_H
#define GPU_GENERATE_H

#include <cstddef>
#include <string>
#include <vector>

#ifndef HIP_GENERATE_THRESHOLD
#define HIP_GENERATE_THRESHOLD 4096
#endif

#ifndef HIP_GENERATE_MAX_GUESS_LEN
#define HIP_GENERATE_MAX_GUESS_LEN 128
#endif

bool HipGenerateSegmentValues(
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
