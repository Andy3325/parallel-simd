# CUDA Generate Optimization Notes

## Summary

- Original CUDA generate correctly moves the PT-internal value expansion loop to GPU, but the prior end-to-end total was slower than the CPU loop.
- Buffer reuse reduces repeated `cudaMalloc`/`cudaFree` pressure by keeping device buffers alive across calls.
- Multi-PT batching explores reducing small-PT launch and transfer overhead by packing several synthetic PT generate tasks into one kernel launch.
- These changes are generate-path optimizations only. They do not modify MD5Hash, do not implement SIMD-MD5, and do not claim hash acceleration.

## Results

- All original, reuse, and batch CSV rows have `correct=1`.
- Best reuse delta was at n=1000000: reuse_total - original_total = -0.008255940 sec, ratio=0.8602.
- Buffer reuse reduced CUDA total time for at least one tested size on this run.
- Smallest batch CUDA total row: num_tasks=4, values_per_task=1000, total_items=4000, cuda_total=0.001206840 sec.

## Interpretation for Report

- 44 items should be used only as correctness evidence.
- 10k, 100k, and 1M items are better for performance trends.
- Even after reuse or batching, end-to-end CUDA may remain slower than the CPU loop because the workflow still performs host-side flattening, host/device data format conversion, D2H copies, and CPU string rebuild.
- Kernel time is small relative to total time, which supports that GPU parallel generation is effective internally, while the surrounding CPU/GPU boundary dominates the current application path.
- Because generated strings return to CPU for the existing downstream flow, larger PTs or batching across multiple PTs are needed before end-to-end acceleration is likely.
