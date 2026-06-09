#include "gpu_generate_cuda.h"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main()
{
    string prefix = "abc";
    vector<string> values = {"1", "22", "333", "4444"};
    vector<string> expected = {"abc1", "abc22", "abc333", "abc4444"};
    vector<string> guesses(expected.size());

    double h2d_time_sec = 0.0;
    double kernel_time_sec = 0.0;
    double d2h_time_sec = 0.0;

    bool ok = CudaGenerateSegmentValues(
        prefix,
        values,
        static_cast<int>(values.size()),
        guesses,
        0,
        &h2d_time_sec,
        &kernel_time_sec,
        &d2h_time_sec);

    if (!ok)
    {
        cerr << "CUDA generate smoke test FAILED: CudaGenerateSegmentValues returned false" << endl;
        return 1;
    }

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (guesses[i] != expected[i])
        {
            cerr << "CUDA generate smoke test FAILED at index " << i
                 << ": expected " << expected[i]
                 << ", got " << guesses[i] << endl;
            return 1;
        }
    }

    cout << "CUDA generate smoke test PASSED" << endl;
    return 0;
}
