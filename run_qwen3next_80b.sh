#!/usr/bin/env bash
set -uo pipefail

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN 환경변수를 설정하세요}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_pii_evaluation.py"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results/300"
LOG_DIR="$RESULTS_DIR/logs"
NAME="qwen3next_80b_a3b_int4"
HF_ID="Forturne/Qwen3-Next-80B-A3B-Instruct-INT4-GPTQ"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "Qwen3-Next-80B-A3B INT4-GPTQ Benchmark"
echo "시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Kill any existing server
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 3

# Start vLLM - GPTQ quantization is embedded in model, max-model-len reduced for tight VRAM
VLLM_CMD="vllm serve $HF_ID --max-model-len 16384 --gpu-memory-utilization 0.97 --max-num-seqs 2"
echo "서버 시작: $VLLM_CMD"
$VLLM_CMD > "$LOG_DIR/${NAME}_server.log" 2>&1 &
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
    echo "=== 서버 로그 마지막 40줄 ==="
    tail -40 "$LOG_DIR/${NAME}_server.log"
    pkill -9 -f "vllm serve" 2>/dev/null || true
    exit 1
fi

# 1) Accuracy benchmark
echo "[1/2] 정확도 벤치마크 (300 cases)..."
python3 "$EVAL_SCRIPT" \
    --model "$HF_ID" \
    --output "$RESULTS_DIR/results_${NAME}.json" \
    --no-think \
    --concurrency 1 \
    2>&1 | tee "$LOG_DIR/${NAME}_accuracy.log"

if [ ! -f "$RESULTS_DIR/results_${NAME}.json" ]; then
    echo "FAIL: 결과 파일 미생성"
    pkill -9 -f "vllm serve" 2>/dev/null || true
    exit 1
fi
echo "정확도 벤치마크 성공"

# 2) Latency test
echo "[2/2] 레이턴시 측정..."
python3 "$EVAL_SCRIPT" \
    --model "$HF_ID" \
    --latency \
    --output "$RESULTS_DIR/latency_${NAME}.json" \
    --no-think \
    2>&1 | tee "$LOG_DIR/${NAME}_latency.log"

echo "완료!"
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3
echo "종료: $(date '+%Y-%m-%d %H:%M:%S')"
