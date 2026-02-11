# PII Detection Test Suite

LLM 기반 개인정보(PII) 검출 성능을 평가하기 위한 테스트 케이스 100개 + vLLM structured output 평가 스크립트

## Quick Start

### 1. 환경 설치

```bash
git clone https://github.com/gkswns0531/pii-detection-test.git
cd pii-detection-test

pip install vllm openai
```

### 2. vLLM 서버 기동

```bash
# 기본 실행 (GPU 자동 감지)
vllm serve Qwen/Qwen2.5-7B-Instruct

# GPU 지정 (멀티 GPU 중 특정 GPU 사용)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct

# 포트 변경
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8080

# Tensor Parallel (멀티 GPU)
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2

# 큰 모델 (양자화)
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq

# GPU 메모리 제한
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.8

# guided decoding backend 지정 (structured output 성능 향상)
vllm serve Qwen/Qwen2.5-7B-Instruct --guided-decoding-backend outlines
```

서버가 뜨면 `http://localhost:8000/v1` 에서 OpenAI 호환 API를 제공합니다.

헬스체크:
```bash
curl http://localhost:8000/health
# 또는
curl http://localhost:8000/v1/models
```

### 3. 평가 실행

```bash
# 전체 100개 테스트
python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct

# 포트를 변경한 경우
python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8080/v1

# 결과 JSON 저장
python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct --output results.json
```

### 4. 필터링 실행

```bash
# 카테고리별
python run_pii_evaluation.py --model ... --category 이름
python run_pii_evaluation.py --model ... --category 주소
python run_pii_evaluation.py --model ... --category "복합 PII"
python run_pii_evaluation.py --model ... --category "False Positive"

# 난이도별
python run_pii_evaluation.py --model ... --difficulty EASY
python run_pii_evaluation.py --model ... --difficulty HARD

# 특정 케이스만
python run_pii_evaluation.py --model ... --ids TC001 TC074 TC098
```

---

## 테스트 케이스 구성 (100개)

| # | 카테고리 | 케이스 수 | 설명 |
|---|---------|----------|------|
| 1 | 이름 | TC001~TC010 (10) | 한글/영문/한자, 복합성씨, 부분마스킹 |
| 2 | 주소 | TC011~TC022 (12) | 도로명/지번, 영문, 해외주소, 비정형 |
| 3 | 주민등록번호 | TC023~TC032 (10) | 표준/마스킹/OCR오류, 외국인등록번호 |
| 4 | 여권/면허번호 | TC033~TC040 (8) | 한국/외국 여권, 운전면허 |
| 5 | 이메일 | TC041~TC048 (8) | 표준/서브도메인/난독화/마스킹 |
| 6 | IP주소 | TC049~TC056 (8) | IPv4/IPv6/CIDR, 로그 속 IP |
| 7 | 전화번호 | TC057~TC064 (8) | 휴대폰/유선/국제번호/마스킹 |
| 8 | 금융정보 | TC065~TC072 (8) | 계좌/카드/IBAN/암호화폐 |
| 9 | 복합 PII | TC073~TC087 (15) | 이력서, 계약서, 의료기록, 법률문서 등 |
| 10 | False Positive | TC088~TC097 (10) | PII 없는 문서 (오탐지 검증) |
| 11 | 난독화/변형 | TC098~TC100 (3) | 공백삽입, OCR오류, 유니코드변형 |

### 난이도 분포
- **EASY** (21개): 표준 형식, 명확한 레이블
- **MEDIUM** (37개): 문맥 속 포함, 형식 변형
- **HARD** (42개): 난독화, 비정형, 오탐 유사 패턴

---

## JSON Schema (LLM 출력 형태)

vLLM structured output으로 아래 12개 카테고리를 `List[str] | null`로 출력합니다:

```json
{
  "이름": ["김철수", "이영희"],
  "주소": ["서울특별시 강남구 테헤란로 152"],
  "주민등록번호": null,
  "여권번호": null,
  "운전면허번호": null,
  "이메일": ["test@example.com"],
  "IP주소": null,
  "전화번호": ["010-1234-5678"],
  "계좌번호": null,
  "카드번호": null,
  "생년월일": null,
  "기타_고유식별정보": null
}
```

---

## 평가 리포트 예시

```
================================================================================
카테고리별 성능
================================================================================
카테고리              Precision     Recall         F1     TP     FP     FN
이름                    92.3%      87.5%      89.8%   154     13     22
주소                    85.0%      80.0%      82.4%    40      7     10
주민등록번호              98.0%      96.0%      97.0%    24      0      1
...

================================================================================
난이도별 성능
================================================================================
난이도      케이스수   Precision     Recall         F1
EASY          21     98.50%     95.20%     96.82%
MEDIUM        37     88.30%     82.10%     85.09%
HARD          42     72.40%     65.80%     68.95%

================================================================================
전체 Micro-Average: P=85.20%  R=79.30%  F1=82.14%
  테스트 케이스: 100개 중 62개 완벽 통과
================================================================================
```

---

## 파일 구조

```
.
├── README.md
├── all_test_cases.json          # 통합 테스트 케이스 100개 (이것만 있으면 됨)
├── run_pii_evaluation.py        # 평가 스크립트
├── __init__.py                  # Python 패키지 (로컬 개발용)
├── pii_test_cases.py            # Part1 소스
├── pii_test_cases_part2.py      # Part2 소스
├── pii_test_cases_part3.py      # Part3 소스
├── pii_test_cases_part4.py      # Part4 소스
└── pii_test_cases_part5.py      # Part5 소스
```

GPU 장비에서는 `all_test_cases.json` + `run_pii_evaluation.py` 두 파일만 있으면 평가 가능합니다.
