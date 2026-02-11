# PII Detection Test Suite

LLM 기반 개인정보(PII) 검출 성능 평가 - 테스트 케이스 100개 + vLLM structured output 평가 스크립트

서버 기동 없이 **터미널 하나에서 바로 실행** 가능합니다.

## Quick Start

```bash
# 1. 클론 + 의존성
git clone https://github.com/gkswns0531/pii-detection-test.git
cd pii-detection-test
pip install vllm

# 2. 바로 실행 (서버 불필요, 오프라인 배치 추론)
python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct
```

끝입니다. 모델 다운로드 → 로드 → 100개 배치 추론 → 평가 리포트까지 한 번에 나옵니다.

---

## 실행 옵션

### GPU 설정

```bash
# 특정 GPU 지정
CUDA_VISIBLE_DEVICES=0 python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct

# 멀티 GPU (tensor parallel)
python run_pii_evaluation.py --model Qwen/Qwen2.5-72B-Instruct-AWQ --tp 2

# 큰 모델 (양자화)
python run_pii_evaluation.py --model Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq

# GPU 메모리 제한
python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.8

# max model length 지정 (OOM 방지)
python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct --max-model-len 8192
```

### 필터링

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

### 결과 저장

```bash
python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct --output results.json
```

`results.json`에 전체 메트릭 + 케이스별 expected/predicted 비교가 저장됩니다.

---

## 전체 인자 목록

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` | (필수) | HuggingFace 모델 이름 |
| `--tp` | 1 | Tensor parallel size |
| `--gpu-memory-utilization` | 0.9 | GPU 메모리 사용률 |
| `--max-model-len` | 모델 기본값 | 최대 시퀀스 길이 |
| `--quantization` | None | 양자화 (awq, gptq 등) |
| `--category` | None | 특정 카테고리 필터 |
| `--difficulty` | None | EASY / MEDIUM / HARD |
| `--ids` | None | 특정 TC ID 필터 |
| `--output` | None | 결과 JSON 저장 경로 |
| `--max-tokens` | 4096 | 생성 최대 토큰 |
| `--temperature` | 0.0 | 샘플링 온도 |

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

vLLM `GuidedDecodingParams`로 아래 12개 카테고리를 `List[str] | null`로 강제합니다:

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
모델 로딩 중...
모델 로드 완료!

추론 시작 (100개 배치)...
추론 완료! (45.3초, 평균 0.45초/케이스)

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
├── all_test_cases.json          # 통합 테스트 케이스 100개
├── run_pii_evaluation.py        # 평가 스크립트 (이 두 파일만 있으면 됨)
├── __init__.py                  # Python 패키지 (로컬 개발용)
├── pii_test_cases.py            # Part1 소스
├── pii_test_cases_part2.py      # Part2 소스
├── pii_test_cases_part3.py      # Part3 소스
├── pii_test_cases_part4.py      # Part4 소스
└── pii_test_cases_part5.py      # Part5 소스
```
