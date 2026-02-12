#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# PII Detection - 2 New Models Benchmark Runner
# Nemotron-3-Nano-30B-A3B-FP8 + gpt-oss-20b
# ============================================================================

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN 환경변수를 설정하세요}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_pii_evaluation.py"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results/300"
LOG_DIR="$RESULTS_DIR/logs"
MAX_MODEL_LEN=16384
WAIT_TIMEOUT=600

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── 모델 정의 (name|hf_id|vllm_extra_args|eval_extra_args|skip_quant) ──
declare -a MODELS=(
    "nemotron3_nano_30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|--trust-remote-code||1"
    "gpt_oss_20b|openai/gpt-oss-20b|||1"
)

# ── 유틸 함수 ──

wait_for_server() {
    local elapsed=0
    echo "  서버 준비 대기중..."
    while [ $elapsed -lt $WAIT_TIMEOUT ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  서버 준비 완료! (${elapsed}초)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  ... ${elapsed}초 경과"
        fi
    done
    echo "  ERROR: 서버 시작 타임아웃 (${WAIT_TIMEOUT}초)"
    return 1
}

kill_server() {
    echo "  서버 종료중..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "from vllm" 2>/dev/null || true
    sleep 5
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [ -n "$gpu_pids" ]; then
        echo "  GPU 프로세스 정리: $gpu_pids"
        for pid in $gpu_pids; do
            if ! ps -p "$pid" -o cmd= 2>/dev/null | grep -q "jupyter"; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        sleep 5
    fi
    echo "  서버 종료 완료"
}

# ── 메인 루프 ──

echo "============================================================"
echo "PII Detection Benchmark - 2 New Models × 300 Cases"
echo "시작 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

TOTAL=${#MODELS[@]}
CURRENT=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name hf_id vllm_extra eval_extra skip_quant <<< "$entry"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] $name ($hf_id)"
    echo "============================================================"

    # 이미 완료된 모델 스킵
    if [ -f "$RESULTS_DIR/results_${name}.json" ] && [ -f "$RESULTS_DIR/latency_${name}.json" ]; then
        echo "  SKIP: 이미 완료됨"
        continue
    fi

    kill_server

    # vLLM 서버 시작
    QUANT_ARG=""
    if [ "${skip_quant:-0}" != "1" ]; then
        QUANT_ARG="--quantization fp8"
    fi

    VLLM_CMD="vllm serve $hf_id --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.90 $QUANT_ARG $vllm_extra"
    echo "  서버 시작: $VLLM_CMD"

    $VLLM_CMD > "$LOG_DIR/${name}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "  서버 PID: $SERVER_PID"

    if ! wait_for_server; then
        echo "  SKIP: 서버 시작 실패 - $name"
        tail -30 "$LOG_DIR/${name}_server.log" 2>/dev/null || true
        kill_server
        continue
    fi

    # 1) 정확도 벤치마크
    echo "  [1/2] 정확도 벤치마크 시작 (300 cases)..."
    python3 "$EVAL_SCRIPT" \
        --model "$hf_id" \
        --output "$RESULTS_DIR/results_${name}.json" \
        $eval_extra \
        2>&1 | tee "$LOG_DIR/${name}_accuracy.log"

    if [ ! -f "$RESULTS_DIR/results_${name}.json" ]; then
        echo "  FAIL: 결과 파일 미생성 - $name 건너뜀"
        kill_server
        continue
    fi
    echo "  정확도 벤치마크 성공"

    # 2) 레이턴시 측정
    echo "  [2/2] 레이턴시 측정 시작..."
    python3 "$EVAL_SCRIPT" \
        --model "$hf_id" \
        --latency \
        --output "$RESULTS_DIR/latency_${name}.json" \
        $eval_extra \
        2>&1 | tee "$LOG_DIR/${name}_latency.log"

    if [ ! -f "$RESULTS_DIR/latency_${name}.json" ]; then
        echo "  WARNING: 레이턴시 파일 미생성"
    else
        echo "  레이턴시 측정 성공"
    fi

    echo "  [$name] 완료!"
    kill_server
done

echo ""
echo "============================================================"
echo "벤치마크 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
ls -la "$RESULTS_DIR"/results_*.json 2>/dev/null
