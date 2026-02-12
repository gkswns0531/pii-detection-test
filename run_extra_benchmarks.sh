#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN 환경변수를 설정하세요}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_pii_evaluation.py"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
LOG_DIR="$RESULTS_DIR/logs"
MAX_MODEL_LEN=8192
WAIT_TIMEOUT=600

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── 추가 모델 4개 (name|hf_id|vllm_extra_args|eval_extra_args) ──
declare -a MODELS=(
    # nemotron3_nano_30b 스킵: MoE FP8 미지원 + BF16 OOM (46GB L40S)
    "qwen3_14b|Qwen/Qwen3-14B||--no-think"
    "gemma3_27b|google/gemma-3-27b-it|--block-size 32|"
    "gpt_oss_20b|openai/gpt-oss-20b||"
)

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
    sleep 10
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 5
    echo "  서버 종료 완료"
}

echo "============================================================"
echo "PII Detection Benchmark - 추가 4 Models"
echo "시작 시각: $(date '+%Y-%m-%d %H:%M:%S')"
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
        echo "  SKIP: 이미 완료됨"
        continue
    fi

    kill_server

    # FP8 추론 (gpt-oss는 MXFP4, nemotron은 MoE FP8 미지원이므로 제외)
    QUANT_ARG="--quantization fp8"
    if [[ "$name" == "gpt_oss_20b" ]] || [[ "$name" == "nemotron3_nano_30b" ]]; then
        QUANT_ARG=""
    fi

    VLLM_CMD="vllm serve $hf_id --max-model-len $MAX_MODEL_LEN $QUANT_ARG $vllm_extra"
    echo "  서버 시작: $VLLM_CMD"

    $VLLM_CMD > "$LOG_DIR/${name}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "  서버 PID: $SERVER_PID"

    if ! wait_for_server; then
        echo "  SKIP: 서버 시작 실패 - $name"
        echo "  로그 마지막 20줄:"
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
    kill_server
done

echo ""
echo "============================================================"
echo "추가 벤치마크 완료!"
echo "종료 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "결과 파일:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (결과 파일 없음)"
