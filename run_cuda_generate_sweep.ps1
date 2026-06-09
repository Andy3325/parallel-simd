$ErrorActionPreference = "Stop"

$cuda124 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if ((!(Test-Path Env:\CUDA_PATH)) -or $env:CUDA_PATH -ne $cuda124) {
    if (Test-Path $cuda124) {
        $env:CUDA_PATH = $cuda124
        $env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
    }
}

function Import-VsDevEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vcvars = "D:\VSBuildTools2022\VC\Auxiliary\Build\vcvars64.bat"
    if (!(Test-Path $vcvars)) {
        return
    }

    Write-Host "[sweep] Loading Visual Studio C++ environment from $vcvars"
    $envDump = cmd /c "`"$vcvars`" >nul && set"
    foreach ($line in $envDump) {
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            Set-Item -Path "Env:\$($parts[0])" -Value $parts[1]
        }
    }
}

Import-VsDevEnvironment

if (!(Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc not found in PATH. Install CUDA Toolkit or add nvcc.exe to PATH."
}

New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "[sweep] nvcc version"
nvcc --version
if ($LASTEXITCODE -ne 0) {
    throw "nvcc --version failed with exit code $LASTEXITCODE"
}

Write-Host "[sweep] Build benchmark_cuda_generate.exe"
nvcc -O2 -std=c++17 -DENABLE_CUDA_GENERATE -Xcompiler /utf-8 benchmark_cuda_generate.cpp gpu_generate_cuda.cu -o benchmark_cuda_generate.exe
if ($LASTEXITCODE -ne 0) {
    throw "benchmark build failed with exit code $LASTEXITCODE"
}

Write-Host "[sweep] Run benchmark_cuda_generate.exe"
.\benchmark_cuda_generate.exe
if ($LASTEXITCODE -ne 0) {
    throw "benchmark failed with exit code $LASTEXITCODE"
}

$csvPath = Join-Path (Get-Location) "results\cuda_generate_sweep.csv"
Write-Host "[sweep] CSV: $csvPath"
Write-Host "[sweep] CSV preview"
Get-Content -LiteralPath $csvPath | Select-Object -First 12
