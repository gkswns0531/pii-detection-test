#!/usr/bin/env bash
set -uo pipefail

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN 환경변수를 설정하세요}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_pii_evaluation.py"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results/300"
LOG_DIR="$RESULTS_DIR/logs"

echo "============================================================"
echo "GPT-OSS-20B Benchmark (No-Think Template)"
echo "시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Kill any existing server
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

# Start vLLM with modified chat template (no reasoning)
VLLM_CMD="vllm serve openai/gpt-oss-20b --max-model-len 16384 --gpu-memory-utilization 0.90 --chat-template $SCRIPT_DIR/gpt_oss_no_think.jinja"
echo "서버 시작: $VLLM_CMD"
$VLLM_CMD > "$LOG_DIR/gpt_oss_20b_server.log" 2>&1 &
SERVER_PID=$!

# Wait for server
elapsed=0
echo "서버 대기중..."
while [ $elapsed -lt 600 ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "서버 준비 완료! (${elapsed}초)"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    [ $((elapsed % 30)) -eq 0 ] && echo "  ... ${elapsed}초 경과"
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: 서버 시작 실패"
    tail -30 "$LOG_DIR/gpt_oss_20b_server.log"
    exit 1
fi

# 1) Accuracy benchmark
echo "[1/2] 정확도 벤치마크 (300 cases)..."
python3 "$EVAL_SCRIPT" \
    --model "openai/gpt-oss-20b" \
    --output "$RESULTS_DIR/results_gpt_oss_20b.json" \
    2>&1 | tee "$LOG_DIR/gpt_oss_20b_accuracy.log"

if [ ! -f "$RESULTS_DIR/results_gpt_oss_20b.json" ]; then
    echo "FAIL: 결과 파일 미생성"
    pkill -9 -f "vllm serve" 2>/dev/null || true
    exit 1
fi

# 2) Latency test
echo "[2/2] 레이턴시 측정..."
python3 "$EVAL_SCRIPT" \
    --model "openai/gpt-oss-20b" \
    --latency \
    --output "$RESULTS_DIR/latency_gpt_oss_20b.json" \
    2>&1 | tee "$LOG_DIR/gpt_oss_20b_latency.log"

echo "완료!"
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3
echo "종료: $(date '+%Y-%m-%d %H:%M:%S')"
