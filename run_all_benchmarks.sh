#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PII Detection - 11 Model Benchmark Runner
# 각 모델: vLLM 서버 시작 → 정확도 벤치마크 → 레이턴시 측정 → 서버 종료
# ============================================================================

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN 환경변수를 설정하세요}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_pii_evaluation.py"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
LOG_DIR="$RESULTS_DIR/logs"
MAX_MODEL_LEN=8192
WAIT_TIMEOUT=600  # 서버 대기 최대 10분

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── 모델 정의 (name|hf_id|vllm_extra_args|eval_extra_args) ──
declare -a MODELS=(
    # Category 1: ≤1B
    "qwen3_0.6b|Qwen/Qwen3-0.6B||--no-think"
    "gemma3_1b|google/gemma-3-1b-it|--block-size 32|"
    "llama32_1b|meta-llama/Llama-3.2-1B-Instruct||"
    # Category 2: 1B-3B
    "qwen3_1.7b|Qwen/Qwen3-1.7B||--no-think"
    "smollm3_3b|HuggingFaceTB/SmolLM3-3B||"
    "llama32_3b|meta-llama/Llama-3.2-3B-Instruct||"
    # Category 3: 3B-10B
    "qwen3_8b|Qwen/Qwen3-8B||--no-think"
    "qwen3_4b_2507|Qwen/Qwen3-4B-Instruct-2507||--no-think"
    "falcon_h1r_7b|tiiuae/Falcon-H1R-7B||"
    "gemma3_4b|google/gemma-3-4b-it|--block-size 32|"
    # Baseline (native FP8)
    "qwen3_30b_a3b_fp8|Qwen/Qwen3-30B-A3B-Instruct-2507-FP8||--no-think"
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
    # GPU 메모리 해제 대기
    sleep 10
    # 혹시 남아있는 프로세스 강제 종료
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 5
    echo "  서버 종료 완료"
}

# ── 메인 루프 ──

echo "============================================================"
echo "PII Detection Benchmark - 11 Models"
echo "시작 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "결과 저장: $RESULTS_DIR"
echo "============================================================"

TOTAL=${#MODELS[@]}
CURRENT=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name hf_id vllm_extra eval_extra <<< "$entry"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] $name ($hf_id)"
    echo "============================================================"

    # 이미 완료된 모델 스킵
    if [ -f "$RESULTS_DIR/results_${name}.json" ] && [ -f "$RESULTS_DIR/latency_${name}.json" ]; then
        echo "  SKIP: 이미 완료됨 (results + latency 존재)"
        continue
    fi

    # 이전 서버 정리
    kill_server

    # vLLM 서버 시작
    # 30B FP8 모델은 이미 FP8이므로 --quantization 불필요, 나머지는 fp8
    QUANT_ARG=""
    if [[ "$name" != "qwen3_30b_a3b_fp8" ]]; then
        QUANT_ARG="--quantization fp8"
    fi

    VLLM_CMD="vllm serve $hf_id --max-model-len $MAX_MODEL_LEN $QUANT_ARG $vllm_extra"
    echo "  서버 시작: $VLLM_CMD"

    $VLLM_CMD > "$LOG_DIR/${name}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "  서버 PID: $SERVER_PID"

    # 서버 준비 대기
    if ! wait_for_server; then
        echo "  SKIP: 서버 시작 실패 - $name"
        echo "  로그 확인: $LOG_DIR/${name}_server.log"
        tail -20 "$LOG_DIR/${name}_server.log" 2>/dev/null || true
        kill_server
        continue
    fi

    # 1) 정확도 벤치마크
    echo "  [1/2] 정확도 벤치마크 시작..."
    python3 "$EVAL_SCRIPT" \
        --model "$hf_id" \
        --output "$RESULTS_DIR/results_${name}.json" \
        $eval_extra \
        2>&1 | tee "$LOG_DIR/${name}_accuracy.log"

    # 2) 레이턴시 측정
    echo "  [2/2] 레이턴시 측정 시작..."
    python3 "$EVAL_SCRIPT" \
        --model "$hf_id" \
        --latency \
        --output "$RESULTS_DIR/latency_${name}.json" \
        $eval_extra \
        2>&1 | tee "$LOG_DIR/${name}_latency.log"

    echo "  [$name] 완료!"

    # 서버 종료
    kill_server
done

echo ""
echo "============================================================"
echo "전체 벤치마크 완료!"
echo "종료 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "결과 디렉토리: $RESULTS_DIR"
echo "============================================================"
echo ""
echo "결과 파일:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (결과 파일 없음)"
