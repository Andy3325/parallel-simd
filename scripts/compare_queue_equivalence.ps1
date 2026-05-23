param(
    [string]$TrainPath = "guessdata\benchmark-small.txt",
    [int]$Rounds = 1000
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $TrainPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $TrainPath) | Out-Null
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
    ) | Set-Content -Encoding utf8 $TrainPath
}

g++ -O2 -std=c++17 tests\queue_equivalence_test.cpp train.cpp guessing.cpp md5.cpp -o queue_ref_test
g++ -O2 -std=c++17 -DENABLE_PRIORITY_LAZY_OPT tests\queue_equivalence_test.cpp train.cpp guessing.cpp md5.cpp -o queue_lazy_test

.\queue_ref_test.exe --train $TrainPath --rounds $Rounds |
    Where-Object { $_ -match '^(TRACE|SUMMARY)\|' } |
    Set-Content -Encoding ascii queue_ref_trace.log

.\queue_lazy_test.exe --train $TrainPath --rounds $Rounds |
    Where-Object { $_ -match '^(TRACE|SUMMARY)\|' } |
    Set-Content -Encoding ascii queue_lazy_trace.log

$ref = Get-Content queue_ref_trace.log
$lazy = Get-Content queue_lazy_trace.log

if ($ref.Count -ne $lazy.Count) {
    Write-Host "Queue equivalence failed: line count mismatch"
    Write-Host "reference_lines=$($ref.Count)"
    Write-Host "lazy_lines=$($lazy.Count)"
    exit 1
}

for ($i = 0; $i -lt $ref.Count; ++$i) {
    if ($ref[$i] -ne $lazy[$i]) {
        Write-Host "Queue equivalence failed at trace line $i"
        Write-Host "reference: $($ref[$i])"
        Write-Host "lazy:      $($lazy[$i])"
        exit 1
    }
}

Write-Host "Queue equivalence passed"
Write-Host $ref[-1]
