param(
    [string]$CudaDefines = ""
)

$ErrorActionPreference = "Stop"

function Import-VsDevEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (!(Test-Path $vswhere)) {
        return
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$installPath) {
        return
    }

    $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
    if (!(Test-Path $vcvars)) {
        return
    }

    Write-Host "[build] Loading Visual Studio C++ environment from $vcvars"
    $envDump = cmd /c "`"$vcvars`" >nul && set"
    foreach ($line in $envDump) {
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            Set-Item -Path "Env:\$($parts[0])" -Value $parts[1]
        }
    }
}

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    Write-Host "[build] $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Import-VsDevEnvironment

if (!(Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc not found in PATH. Install CUDA Toolkit or add nvcc.exe to PATH."
}

if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "cl.exe not found in PATH. CUDA nvcc on Windows normally requires Visual Studio C++ Build Tools."
}

Invoke-Step "CPU baseline" {
    g++ -O2 -std=c++17 main.cpp train.cpp guessing.cpp md5.cpp -o main_cpu.exe
}

$extraArgs = @()
if ($CudaDefines.Trim().Length -gt 0) {
    $extraArgs = $CudaDefines.Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}
$clExtraArgs = @()
foreach ($arg in $extraArgs) {
    if ($arg.StartsWith("-D")) {
        $clExtraArgs += "/D$($arg.Substring(2))"
    }
    else {
        $clExtraArgs += $arg
    }
}
$nvccHostArgs = @("-Xcompiler", "/utf-8")

try {
    Invoke-Step "CUDA smoke test direct" {
        nvcc -O2 -std=c++17 -DENABLE_CUDA_GENERATE @extraArgs @nvccHostArgs test_cuda_generate.cpp gpu_generate_cuda.cu -o test_cuda_generate.exe
    }

    Invoke-Step "CUDA main direct" {
        nvcc -O2 -std=c++17 -DENABLE_CUDA_GENERATE @extraArgs @nvccHostArgs main.cpp train.cpp guessing.cpp md5.cpp gpu_generate_cuda.cu -o main_cuda.exe
    }
}
catch {
    Write-Host "[build] Direct nvcc build failed; trying object-file fallback"

    Remove-Item -Force -ErrorAction SilentlyContinue `
        main_cuda.exe, `
        test_cuda_generate.exe, `
        main.o, `
        train.o, `
        guessing.o, `
        md5.o, `
        gpu_generate_cuda.o, `
        test_cuda_generate.o, `
        main.obj, `
        train.obj, `
        guessing.obj, `
        md5.obj, `
        gpu_generate_cuda.obj, `
        test_cuda_generate.obj

    Invoke-Step "CPU objects with cl" {
        cl /nologo /O2 /std:c++17 /EHsc /utf-8 /DENABLE_CUDA_GENERATE @clExtraArgs /c main.cpp train.cpp guessing.cpp md5.cpp
    }

    Invoke-Step "CUDA object" {
        nvcc -O2 -std=c++17 -DENABLE_CUDA_GENERATE @extraArgs @nvccHostArgs -c gpu_generate_cuda.cu -o gpu_generate_cuda.obj
    }

    Invoke-Step "CUDA smoke test object with cl" {
        cl /nologo /O2 /std:c++17 /EHsc /utf-8 /DENABLE_CUDA_GENERATE @clExtraArgs /c test_cuda_generate.cpp
    }

    Invoke-Step "Link smoke test with nvcc" {
        nvcc test_cuda_generate.obj gpu_generate_cuda.obj -o test_cuda_generate.exe
    }

    Invoke-Step "Link CUDA main with nvcc" {
        nvcc main.obj train.obj guessing.obj md5.obj gpu_generate_cuda.obj -o main_cuda.exe
    }
}

Write-Host "Build complete."
