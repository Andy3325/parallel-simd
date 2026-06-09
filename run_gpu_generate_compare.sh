#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="guessdata/benchmark-small.txt"
ROUNDS="${1:-1000}"

mkdir -p guessdata results

if [ ! -f "${TRAIN_PATH}" ]; then
    cat > "${TRAIN_PATH}" <<'DATA'
password123
Password123!
abc123
hello2024
admin!@#
qwerty
letmein42
Test@2026
abcDEF123
zzzz9999!
DATA
fi

echo "[build] CPU baseline"
g++ -O2 -std=c++17 main.cpp train.cpp guessing.cpp md5.cpp -o main_cpu

echo "[build] HIP generate"
hipcc -O2 -std=c++17 -DENABLE_HIP_GENERATE main.cpp train.cpp guessing.cpp md5.cpp gpu_generate.hip -o main_hip

echo "[run] CPU baseline"
./main_cpu --train "${TRAIN_PATH}" --guess-rounds "${ROUNDS}" > results/gpu_generate_cpu.txt

echo "[run] HIP generate"
./main_hip --train "${TRAIN_PATH}" --guess-rounds "${ROUNDS}" > results/gpu_generate_hip.txt

if ! grep -q 'MD5Hash test passed!' results/gpu_generate_cpu.txt; then
    echo "CPU run did not pass MD5 correctness test"
    exit 1
fi

if ! grep -q 'MD5Hash test passed!' results/gpu_generate_hip.txt; then
    echo "HIP run did not pass MD5 correctness test"
    exit 1
fi

cpu_total="$(grep -E '^total_guesses = ' results/gpu_generate_cpu.txt | tail -n 1 | awk '{print $3}')"
hip_total="$(grep -E '^total_guesses = ' results/gpu_generate_hip.txt | tail -n 1 | awk '{print $3}')"

if [ "${cpu_total}" != "${hip_total}" ]; then
    echo "total_guesses mismatch: cpu=${cpu_total}, hip=${hip_total}"
    exit 1
fi

echo "CPU vs HIP total_guesses matched: ${cpu_total}"
echo
echo "[CPU summary]"
grep -E '^(total_guesses = |Guess benchmark time:|Hash time:|Train time:|\[GenerateStats\]|\[HIPGenerateStats\]|hip_generate_|generate_calls = |append_)' \
    results/gpu_generate_cpu.txt || true
echo
echo "[HIP summary]"
grep -E '^(total_guesses = |Guess benchmark time:|Hash time:|Train time:|\[GenerateStats\]|\[HIPGenerateStats\]|hip_generate_|generate_calls = |append_)' \
    results/gpu_generate_hip.txt || true
