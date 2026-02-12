#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# PII Detection - Accuracy Re-evaluation (11 models + 30B vanilla)
# 업데이트된 프롬프트/데이터로 accuracy만 재평가
# 결과: combined + base + advanced 자동 분할 저장
# ============================================================================

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN 환경변수를 설정하세요}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_pii_evaluation.py"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results/300"
LOG_DIR="$RESULTS_DIR/logs"
MAX_MODEL_LEN=16384
WAIT_TIMEOUT=600

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── 모델 정의 (name|hf_id|vllm_extra_args|eval_extra_args|skip_fp8) ──
declare -a MODELS=(
    # ≤1B
    "qwen3_0.6b|Qwen/Qwen3-0.6B||--no-think|"
    "gemma3_1b|google/gemma-3-1b-it|--block-size 32||"
    "llama32_1b|meta-llama/Llama-3.2-1B-Instruct|||"
    # 1B-3B
    "qwen3_1.7b|Qwen/Qwen3-1.7B||--no-think|"
    "smollm3_3b|HuggingFaceTB/SmolLM3-3B|||"
    "llama32_3b|meta-llama/Llama-3.2-3B-Instruct|||"
    # 3B-10B
    "qwen3_4b_2507|Qwen/Qwen3-4B-Instruct-2507||--no-think|"
    "gemma3_4b|google/gemma-3-4b-it|--block-size 32||"
    "qwen3_8b|Qwen/Qwen3-8B||--no-think|"
    "falcon_h1r_7b|tiiuae/Falcon-H1R-7B|||"
    # 20B
    "gpt_oss_20b|openai/gpt-oss-20b|||skip_fp8"
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
    # GPU 메모리 정리
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

run_model() {
    local name="$1" hf_id="$2" vllm_extra="$3" eval_extra="$4" skip_fp8="$5"
    local prompt_ver="${6:-}"  # optional: prompt version override
    local output_suffix="${7:-}"  # optional: output filename suffix

    local out_name="${name}${output_suffix}"

    # 이전 서버 정리
    kill_server

    # vLLM 서버 시작
    local QUANT_ARG=""
    if [ -z "${skip_fp8:-}" ]; then
        QUANT_ARG="--quantization fp8"
    fi

    local VLLM_CMD="vllm serve $hf_id --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.90 $QUANT_ARG $vllm_extra"
    echo "  서버 시작: $VLLM_CMD"

    $VLLM_CMD > "$LOG_DIR/${out_name}_server.log" 2>&1 &
    local SERVER_PID=$!
    echo "  서버 PID: $SERVER_PID"

    if ! wait_for_server; then
        echo "  FAIL: 서버 시작 실패 - $name"
        tail -20 "$LOG_DIR/${out_name}_server.log" 2>/dev/null || true
        kill_server
        return 1
    fi

    # Accuracy 벤치마크
    local eval_cmd="python3 $EVAL_SCRIPT --model $hf_id --output $RESULTS_DIR/results_${out_name}.json"
    if [ -n "$eval_extra" ]; then
        eval_cmd="$eval_cmd $eval_extra"
    fi
    if [ -n "$prompt_ver" ]; then
        eval_cmd="$eval_cmd --prompt-version $prompt_ver"
    fi

    echo "  평가 시작: $eval_cmd"
    $eval_cmd 2>&1 | tee "$LOG_DIR/${out_name}_accuracy.log"

    if [ ! -f "$RESULTS_DIR/results_${out_name}.json" ]; then
        echo "  FAIL: 결과 파일 미생성 - $out_name"
        kill_server
        return 1
    fi

    echo "  [$out_name] 완료!"
    kill_server
    return 0
}

# ── 메인 ──

echo "============================================================"
echo "PII Detection - Accuracy Re-evaluation"
echo "${#MODELS[@]} models"
echo "시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "결과: $RESULTS_DIR"
echo "============================================================"

TOTAL=${#MODELS[@]}
CURRENT=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name hf_id vllm_extra eval_extra skip_fp8 <<< "$entry"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] $name ($hf_id)"
    echo "============================================================"

    run_model "$name" "$hf_id" "$vllm_extra" "$eval_extra" "$skip_fp8"
done

echo ""
echo "============================================================"
echo "전체 완료!"
echo "종료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "결과 파일:"
ls -la "$RESULTS_DIR"/results_*.json 2>/dev/null | head -50
