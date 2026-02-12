# PII 검출 소형 모델 벤치마크 — Top 10 리스트 (2026년 2월 기준)

## Category 1: ≤1B Parameters

| Rank | Model | Params | Release | Developer | Key Strengths |
|------|-------|--------|---------|-----------|---------------|
| 1 | **Qwen3-0.6B** | 0.6B | 2025.05 | Alibaba | 최강 sub-1B, 119개 언어, thinking/no-think 듀얼모드 |
| 2 | Gemma-3-1B-IT | 1.0B | 2025.03 | Google | 멀티모달, 140+개 언어, 128K context |
| 3 | Llama-3.2-1B-Instruct | 1.0B | 2024.09 | Meta | 범용 NLP, 파인튜닝 효율 최고 |
| 4 | SmolLM2-1.7B-Instruct | 1.7B | 2024.11 | HuggingFace | 완전 오픈소스, 경량 추론 |
| 5 | Qwen2.5-0.5B-Instruct | 0.5B | 2024.09 | Alibaba | 128K context, 29개 언어 |
| 6 | Gemma-3n-E2B-IT | ~2B eff | 2025.06 | Google | 온디바이스 최적화, 텍스트+이미지+오디오 |
| 7 | SmolLM2-360M-Instruct | 0.36B | 2024.11 | HuggingFace | 초경량 엣지 |
| 8 | TinyLlama-1.1B-Chat | 1.1B | 2024.01 | Community | 3T토큰 학습, 모바일 |
| 9 | Danube3-500M | 0.5B | 2024.11 | H2O.ai | 경량 대화형 |
| 10 | SmolLM2-135M-Instruct | 0.13B | 2024.11 | HuggingFace | 가장 작은 실용 모델 |

## Category 2: 1B < x ≤ 3B Parameters

| Rank | Model | Params | Release | Developer | Key Strengths |
|------|-------|--------|---------|-----------|---------------|
| 1 | **SmolLM3-3B** | 3.0B | 2025.H2 | HuggingFace | Llama-3.2-3B/Qwen2.5-3B 능가, think/no_think 듀얼모드 |
| 2 | Qwen3-1.7B | 1.7B | 2025.05 | Alibaba | Qwen2.5-3B 능가, 119개 언어 |
| 3 | EXAONE-3.5-2.4B-Instruct | 2.4B | 2024.12 | LG AI | 한국어-영어 이중언어 특화 |
| 4 | Ministral-3B-Instruct-2512 | 3.4B | 2025.12 | Mistral AI | 최신 엣지 멀티모달 SLM |
| 5 | Llama-3.2-3B-Instruct | 3.0B | 2024.09 | Meta | 범용 텍스트, 파인튜닝 생태계 |
| 6 | Qwen2.5-3B-Instruct | 3.0B | 2024.09 | Alibaba | 안정적 다국어 기반 |
| 7 | Phi-3-mini-instruct | 3.8B | 2024.06 | Microsoft | 추론·수학 강점 |
| 8 | StableLM-2-1.6B | 1.6B | 2024.01 | Stability AI | 유럽어 + 다국어 |
| 9 | MobileLLaMA-2.7B | 2.7B | 2024.05 | Community | 모바일 최적화, 40% 빠른 추론 |
| 10 | Gemma-2-2B-IT | 2.6B | 2024.06 | Google | 안정적 다국어 기반 |

## Category 3: 3B < x ≤ 10B Parameters

| Rank | Model | Params | Release | Developer | Key Strengths |
|------|-------|--------|---------|-----------|---------------|
| 1 | **Falcon-H1R-7B** | 7.0B | 2026.01 | TII | 🆕 최신! Transformer-Mamba2 하이브리드, 32B급 추론, 256K context |
| 2 | Qwen3-4B-Instruct-2507 | 4.0B | 2025.07 | Alibaba | 최신 distill, 일부 벤치에서 8B 능가 |
| 3 | EXAONE-3.5-7.8B-Instruct | 7.8B | 2024.12 | LG AI | 한국어-영어 이중언어 최강 |
| 4 | Qwen3-8B | 8.0B | 2025.05 | Alibaba | Qwen2.5-14B 능가, 듀얼모드 |
| 5 | Gemma-3-4B-IT | 4.0B | 2025.03 | Google | 멀티모달, 140+개 언어 |
| 6 | Phi-4-mini-instruct | 3.8B | 2024.12 | Microsoft | 추론·다국어 7-9B급 성능 |
| 7 | DeepSeek-R1-Distill-Qwen-7B | 7.0B | 2025.01 | DeepSeek | 수학/코딩 최강 (MATH-500 92.8%) |
| 8 | Llama-3.1-8B-Instruct | 8.0B | 2024.07 | Meta | 범용 8B 기준선 |
| 9 | Mistral-7B-Instruct-v0.3 | 7.2B | 2024.05 | Mistral AI | 유럽어 강점 |
| 10 | Gemma-3-12B-IT | 12.0B | 2025.03 | Google | (10B 초과이나 경쟁력) |

---

## 실제 테스트 완료 모델 (총 11개 + 베이스라인 1개)

평가 환경: NVIDIA B200, vLLM v0.15.1, 299개 한국어 PII 검출 테스트 케이스

### Baseline

| Model | Params | HuggingFace ID | Acc | F1 | Median Latency | 비고 |
|-------|--------|----------------|-----|-----|----------------|------|
| Qwen3-30B-A3B-Instruct-2507-FP8 | 30B (3B active) | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | 90.30% | 94.90% | - | MoE 베이스라인 |

### Category 1: ≤1B (3개 테스트)

| Model | Params | HuggingFace ID | Acc | F1 | 비고 |
|-------|--------|----------------|-----|-----|------|
| Qwen3-0.6B | 0.6B | `Qwen/Qwen3-0.6B` | 13.71% | 39.86% | `--no-think` 필요 |
| Gemma-3-1B-IT | 1.0B | `google/gemma-3-1b-it` | 13.38% | 39.01% | `--block-size 32` 필요 |
| Llama-3.2-1B-Instruct | 1.0B | `meta-llama/Llama-3.2-1B-Instruct` | 6.02% | 16.83% | FP 폭발 |

### Category 2: 1B-3B (3개 테스트 + 1개 스킵)

| Model | Params | HuggingFace ID | Acc | F1 | Median Latency | 비고 |
|-------|--------|----------------|-----|-----|----------------|------|
| Qwen3-1.7B | 1.7B | `Qwen/Qwen3-1.7B` | 62.54% | 77.22% | 0.409s | `--no-think` 필요 |
| SmolLM3-3B | 3.0B | `HuggingFaceTB/SmolLM3-3B` | 60.20% | 70.75% | 0.414s | P 높고 R 낮음 |
| Llama-3.2-3B-Instruct | 3.0B | `meta-llama/Llama-3.2-3B-Instruct` | 16.72% | 26.44% | - | FP 폭발 |
| ~~EXAONE-3.5-2.4B~~ | 2.4B | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` | - | - | - | transformers 호환 이슈 (RopeParameters) |

### Category 3: 3-10B (4개 테스트 + 1개 스킵)

| Model | Params | HuggingFace ID | Acc | F1 | Median Latency | 비고 |
|-------|--------|----------------|-----|-----|----------------|------|
| Qwen3-8B | 8.0B | `Qwen/Qwen3-8B` | 79.60% | 90.04% | 0.947s | `--no-think` 필요, 30B에 가장 근접 |
| Qwen3-4B-Instruct-2507 | 4.0B | `Qwen/Qwen3-4B-Instruct-2507` | 79.60% | 87.79% | 0.622s | `--no-think` 필요, 가성비 최강 |
| Falcon-H1R-7B | 7.0B | `tiiuae/Falcon-H1R-7B` | 47.49% | 69.66% | 1.061s | Mamba2 하이브리드, 한국어 PII 부진 |
| Gemma-3-4B-IT | 4.0B | `google/gemma-3-4b-it` | 9.70% | 42.85% | - | `--block-size 32` 필요, FP 폭발 |
| ~~EXAONE-3.5-7.8B~~ | 7.8B | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | - | - | - | transformers 호환 이슈 (RopeParameters) |

### vLLM 실행 참고사항

| 모델 | vLLM 실행 명령 |
|------|---------------|
| Qwen 계열 | `vllm serve <model> --max-model-len 8192` + 평가 시 `--no-think` |
| Gemma 계열 | `vllm serve <model> --max-model-len 8192 --block-size 32` |
| Falcon-H1R | `vllm serve <model> --max-model-len 8192` |
| Llama 계열 | `vllm serve <model> --max-model-len 8192` |
| SmolLM3 | `vllm serve <model> --max-model-len 8192` |
| EXAONE 계열 | `--trust-remote-code` 필요하나 transformers 4.57.3 호환 불가 |
