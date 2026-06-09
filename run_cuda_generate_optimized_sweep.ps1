$ErrorActionPreference = "Stop"

$cuda124 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if ((!(Test-Path Env:\CUDA_PATH)) -or $env:CUDA_PATH -ne $cuda124) {
    if (Test-Path $cuda124) {
        $env:CUDA_PATH = $cuda124
        $env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
    }
}

New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "[optimized] Build CUDA targets"
.\build_cuda.ps1 -CudaDefines "-DCUDA_GENERATE_THRESHOLD=1"
if ($LASTEXITCODE -ne 0) {
    throw "build_cuda.ps1 failed with exit code $LASTEXITCODE"
}

Write-Host "[optimized] Run original/reuse generate sweep"
.\benchmark_cuda_generate.exe
if ($LASTEXITCODE -ne 0) {
    throw "benchmark_cuda_generate.exe failed with exit code $LASTEXITCODE"
}

Write-Host "[optimized] Run multi-PT batch sweep"
.\benchmark_cuda_generate_batch.exe
if ($LASTEXITCODE -ne 0) {
    throw "benchmark_cuda_generate_batch.exe failed with exit code $LASTEXITCODE"
}

$pythonScript = @'
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

results = Path("results")
orig_path = results / "cuda_generate_sweep.csv"
reuse_path = results / "cuda_generate_reuse_summary.csv"
batch_path = results / "cuda_generate_batch_sweep.csv"
batch_summary_path = results / "cuda_generate_batch_summary.csv"
reuse_plot_path = results / "cuda_generate_reuse_vs_original.png"
batch_plot_path = results / "cuda_generate_batch_trend.png"
notes_path = results / "cuda_generate_optimization_notes.md"

orig = pd.read_csv(orig_path)
reuse = pd.read_csv(reuse_path)
batch = pd.read_csv(batch_path)

if not (orig["correct"].eq(1).all() and reuse["correct"].eq(1).all() and batch["correct"].eq(1).all()):
    raise SystemExit("correctness check failed in one or more CSV files")

orig_summary = orig.groupby("n", as_index=False).mean(numeric_only=True)
reuse_summary = reuse.groupby("n", as_index=False).mean(numeric_only=True)
reuse_compare = orig_summary[["n", "cpu_time_sec", "cuda_total_time_sec", "cuda_kernel_time_sec"]].merge(
    reuse_summary[["n", "cuda_total_time_sec", "cuda_kernel_time_sec"]],
    on="n",
    suffixes=("_original", "_reuse"),
)
reuse_compare["reuse_total_delta_sec"] = reuse_compare["cuda_total_time_sec_reuse"] - reuse_compare["cuda_total_time_sec_original"]
reuse_compare["reuse_total_ratio"] = reuse_compare["cuda_total_time_sec_reuse"] / reuse_compare["cuda_total_time_sec_original"]
reuse_compare.to_csv(results / "cuda_generate_reuse_summary_compare.csv", index=False)

batch_summary = batch.groupby(["num_tasks", "values_per_task", "total_items"], as_index=False).mean(numeric_only=True)
batch_summary["speedup"] = batch_summary["cpu_time_sec"] / batch_summary["cuda_total_time_sec"]
batch_summary.to_csv(batch_summary_path, index=False)

plt.figure(figsize=(8, 5))
plt.plot(orig_summary["n"], orig_summary["cuda_total_time_sec"], marker="o", label="original total")
plt.plot(reuse_summary["n"], reuse_summary["cuda_total_time_sec"], marker="o", label="reuse total")
plt.plot(orig_summary["n"], orig_summary["cpu_time_sec"], marker="o", label="CPU loop")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("items per PT")
plt.ylabel("mean time (sec)")
plt.title("CUDA generate reuse vs original")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(reuse_plot_path, dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
for values_per_task, group in batch_summary.groupby("values_per_task"):
    group = group.sort_values("num_tasks")
    plt.plot(group["num_tasks"], group["cuda_total_time_sec"], marker="o", label=f"CUDA total, values={values_per_task}")
plt.xlabel("num synthetic PT tasks")
plt.ylabel("mean CUDA total time (sec)")
plt.title("Multi-PT batch generate trend")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(batch_plot_path, dpi=160)
plt.close()

best_reuse = reuse_compare.loc[reuse_compare["reuse_total_delta_sec"].idxmin()]
best_batch = batch_summary.sort_values("cuda_total_time_sec").iloc[0]
reuse_helped = bool((reuse_compare["reuse_total_delta_sec"] < 0).any())

notes = f"""# CUDA Generate Optimization Notes

## Summary

- Original CUDA generate correctly moves the PT-internal value expansion loop to GPU, but the prior end-to-end total was slower than the CPU loop.
- Buffer reuse reduces repeated `cudaMalloc`/`cudaFree` pressure by keeping device buffers alive across calls.
- Multi-PT batching explores reducing small-PT launch and transfer overhead by packing several synthetic PT generate tasks into one kernel launch.
- These changes are generate-path optimizations only. They do not modify MD5Hash, do not implement SIMD-MD5, and do not claim hash acceleration.

## Results

- All original, reuse, and batch CSV rows have `correct=1`.
- Best reuse delta was at n={int(best_reuse['n'])}: reuse_total - original_total = {best_reuse['reuse_total_delta_sec']:.9f} sec, ratio={best_reuse['reuse_total_ratio']:.4f}.
- Buffer reuse {'reduced' if reuse_helped else 'did not reduce'} CUDA total time for at least one tested size on this run.
- Smallest batch CUDA total row: num_tasks={int(best_batch['num_tasks'])}, values_per_task={int(best_batch['values_per_task'])}, total_items={int(best_batch['total_items'])}, cuda_total={best_batch['cuda_total_time_sec']:.9f} sec.

## Interpretation for Report

- 44 items should be used only as correctness evidence.
- 10k, 100k, and 1M items are better for performance trends.
- Even after reuse or batching, end-to-end CUDA may remain slower than the CPU loop because the workflow still performs host-side flattening, host/device data format conversion, D2H copies, and CPU string rebuild.
- Kernel time is small relative to total time, which supports that GPU parallel generation is effective internally, while the surrounding CPU/GPU boundary dominates the current application path.
- Because generated strings return to CPU for the existing downstream flow, larger PTs or batching across multiple PTs are needed before end-to-end acceleration is likely.
"""
notes_path.write_text(notes, encoding="utf-8")

print("[optimized] reuse comparison")
print(reuse_compare.to_string(index=False))
print("[optimized] batch summary")
print(batch_summary.to_string(index=False))
print(f"[optimized] wrote {reuse_plot_path}")
print(f"[optimized] wrote {batch_plot_path}")
print(f"[optimized] wrote {notes_path}")
'@

$tmpScript = Join-Path $env:TEMP "cuda_generate_optimized_summary.py"
Set-Content -LiteralPath $tmpScript -Value $pythonScript -Encoding UTF8
python $tmpScript
if ($LASTEXITCODE -ne 0) {
    throw "summary/chart generation failed with exit code $LASTEXITCODE"
}

Write-Host "[optimized] Generated artifacts"
Get-Item -LiteralPath `
    results\cuda_generate_sweep.csv, `
    results\cuda_generate_reuse_summary.csv, `
    results\cuda_generate_batch_sweep.csv, `
    results\cuda_generate_batch_summary.csv, `
    results\cuda_generate_reuse_vs_original.png, `
    results\cuda_generate_batch_trend.png, `
    results\cuda_generate_optimization_notes.md |
    Select-Object Name, Length, LastWriteTime
