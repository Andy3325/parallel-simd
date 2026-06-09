param(
    [int]$Rounds = 1000,
    [string]$TrainPath = "guessdata\benchmark-small.txt"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path (Split-Path $TrainPath))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $TrainPath) | Out-Null
}

if (!(Test-Path "results")) {
    New-Item -ItemType Directory -Force -Path "results" | Out-Null
}

if (!(Test-Path $TrainPath)) {
    @(
        "password123",
        "Password123!",
        "abc123",
        "hello2024",
        "admin!@#",
        "qwerty",
        "letmein42",
        "Test@2026",
        "abcDEF123",
        "zzzz9999!"
    ) | Set-Content -Encoding ascii $TrainPath
}

if (!(Test-Path ".\main_cpu.exe")) {
    throw "main_cpu.exe not found. Run .\build_cuda.ps1 first."
}

if (!(Test-Path ".\main_cuda.exe")) {
    throw "main_cuda.exe not found. Run .\build_cuda.ps1 first."
}

Write-Host "[run] CPU baseline"
.\main_cpu.exe --train $TrainPath --guess-rounds $Rounds > results\cuda_generate_cpu.txt

Write-Host "[run] CUDA generate"
.\main_cuda.exe --train $TrainPath --guess-rounds $Rounds > results\cuda_generate_gpu.txt

if (!(Select-String -Path results\cuda_generate_cpu.txt -Pattern "MD5Hash test passed!" -Quiet)) {
    throw "CPU run did not pass MD5 correctness test."
}

if (!(Select-String -Path results\cuda_generate_gpu.txt -Pattern "MD5Hash test passed!" -Quiet)) {
    throw "CUDA run did not pass MD5 correctness test."
}

$cpuTotalLine = Select-String -Path results\cuda_generate_cpu.txt -Pattern "^total_guesses = " | Select-Object -Last 1
$gpuTotalLine = Select-String -Path results\cuda_generate_gpu.txt -Pattern "^total_guesses = " | Select-Object -Last 1
$cpuTotal = ($cpuTotalLine.Line -split "\s+")[2]
$gpuTotal = ($gpuTotalLine.Line -split "\s+")[2]

if ($cpuTotal -ne $gpuTotal) {
    throw "total_guesses mismatch: cpu=$cpuTotal cuda=$gpuTotal"
}

Write-Host "CPU vs CUDA total_guesses matched: $cpuTotal"

$summaryPattern = "^(total_guesses = |Guess benchmark time:|Guess time:|Hash time:|Train time:|\[GenerateStats\]|\[CUDAGenerateStats\]|cuda_generate_|cuda_h2d_|cuda_kernel_|cuda_d2h_|generate_calls = |append_)"

Write-Host ""
Write-Host "[CPU summary]"
Select-String -Path results\cuda_generate_cpu.txt -Pattern $summaryPattern | ForEach-Object { $_.Line }

Write-Host ""
Write-Host "[CUDA summary]"
Select-String -Path results\cuda_generate_gpu.txt -Pattern $summaryPattern | ForEach-Object { $_.Line }

$cudaCallsLine = Select-String -Path results\cuda_generate_gpu.txt -Pattern "^cuda_generate_calls=" | Select-Object -Last 1
if ($cudaCallsLine -and $cudaCallsLine.Line -match "^cuda_generate_calls=(\d+)") {
    $cudaCalls = [int]$Matches[1]
    if ($cudaCalls -eq 0) {
        Write-Warning "cuda_generate_calls = 0. The PT batch may be below CUDA_GENERATE_THRESHOLD; rebuild with .\build_cuda.ps1 -CudaDefines '-DCUDA_GENERATE_THRESHOLD=1' to force the CUDA path."
    }
}
