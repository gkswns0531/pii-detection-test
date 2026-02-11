"""
PII 검출 성능 평가 스크립트 (API 클라이언트)

별도로 띄워놓은 vLLM 서버에 요청을 보내 PII 검출 성능을 평가합니다.

사용법:
    # 1. 서버 띄우기 (터미널 1)
    vllm serve Qwen/Qwen2.5-7B-Instruct --guided-decoding-backend outlines

    # 2. 벤치마크 실행 (터미널 2)
    python run_pii_evaluation.py --model Qwen/Qwen2.5-7B-Instruct

    # 다른 서버 주소
    python run_pii_evaluation.py --model ... --api-url http://gpu-server:8000/v1

    # 필터링
    python run_pii_evaluation.py --model ... --category 이름
    python run_pii_evaluation.py --model ... --difficulty HARD

    # 결과 저장
    python run_pii_evaluation.py --model ... --output results.json
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

# ============================================================================
# 1. PII 카테고리 정규화 매핑
# ============================================================================

TYPE_NORMALIZATION: dict[str, str] = {
    "이름": "이름", "이름(부분마스킹)": "이름",
    "주소": "주소", "주소(부분)": "주소",
    "주민등록번호": "주민등록번호", "주민등록번호(마스킹)": "주민등록번호",
    "주민등록번호(앞자리)": "주민등록번호", "주민등록번호(OCR오류)": "주민등록번호",
    "외국인등록번호": "주민등록번호",
    "여권번호": "여권번호",
    "운전면허번호": "운전면허번호",
    "이메일": "이메일", "이메일(난독화)": "이메일", "이메일(마스킹)": "이메일",
    "IP주소": "IP주소", "IP주소(IPv6)": "IP주소", "IP주소(사설)": "IP주소",
    "IP주소(공인)": "IP주소", "IP주소:포트": "IP주소", "IP주소(CIDR)": "IP주소",
    "전화번호": "전화번호", "전화번호(부분마스킹)": "전화번호",
    "계좌번호": "계좌번호", "계좌번호(부분마스킹)": "계좌번호",
    "가상계좌번호": "계좌번호", "IBAN": "계좌번호",
    "카드번호": "카드번호", "카드번호(부분마스킹)": "카드번호", "카드번호(부분)": "카드번호",
    "암호화폐지갑주소(BTC)": "카드번호", "암호화폐지갑주소(ETH)": "카드번호",
    "생년월일": "생년월일",
    "학번": "기타_고유식별정보", "차량번호": "기타_고유식별정보",
}

PII_CATEGORIES = [
    "이름", "주소", "주민등록번호", "여권번호", "운전면허번호", "이메일",
    "IP주소", "전화번호", "계좌번호", "카드번호", "생년월일", "기타_고유식별정보",
]


# ============================================================================
# 2. JSON Schema (structured output용)
# ============================================================================

def build_json_schema() -> dict:
    """vLLM guided_json에 전달할 JSON Schema"""
    properties = {}
    for cat in PII_CATEGORIES:
        properties[cat] = {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                {"type": "null"},
            ],
        }

    return {
        "type": "object",
        "properties": properties,
        "required": PII_CATEGORIES,
        "additionalProperties": False,
    }


# ============================================================================
# 3. 프롬프트
# ============================================================================

SYSTEM_PROMPT = """당신은 문서에서 개인정보(PII)를 검출하는 전문가입니다.

주어진 문서를 분석하여 아래 카테고리별로 개인정보를 검출해 주세요.

## 검출 카테고리
1. **이름**: 실제 특정 개인의 한국인/외국인 이름 (한글, 영문, 한자). 부분 마스킹(김○수)도 포함.
2. **주소**: 특정 개인이나 사업장을 식별할 수 있는 도로명/지번 주소, 해외 주소. 우편번호 포함 가능.
3. **주민등록번호**: 주민등록번호(YYMMDD-NNNNNNN), 외국인등록번호 포함. 마스킹(*)/변형 포함.
4. **여권번호**: 한국(M+8자리) 및 외국 여권번호.
5. **운전면허번호**: 한국 운전면허번호(NN-NN-NNNNNN-NN).
6. **이메일**: 이메일 주소. 난독화([at],[dot])나 마스킹 형태도 포함.
7. **IP주소**: 개인이나 특정 장비를 식별할 수 있는 IPv4, IPv6, CIDR 주소. IP:포트 형태도 포함 (포트 포함하여 원문 그대로 추출).
8. **전화번호**: 특정 개인의 휴대폰, 유선전화, 국제번호, 내선번호.
9. **계좌번호**: 은행 계좌번호, 가상계좌, IBAN 포함.
10. **카드번호**: 신용/체크카드 번호, 암호화폐 지갑 주소(BTC, ETH 등) 포함.
11. **생년월일**: "생년월일", "출생", "생일", "DOB" 등으로 **명시적으로 표기된** 개인의 출생 날짜만 해당. 주민번호 앞6자리는 여기에 중복 추출하지 마세요.
12. **기타_고유식별정보**: 학번, 차량번호 등 위 11개 카테고리에 해당하지 않으면서 **특정 개인을 직접 식별**할 수 있는 고유 번호.

## 추출 규칙
- 검출된 PII는 문서에 나타난 **원문 그대로** 추출하세요.
- 각 PII 항목은 **완전한 문자열 하나**로 추출해야 합니다. 절대 글자 단위로 쪼개지 마세요.
  - 올바른 예: ["김철수"] / ["서울특별시 강남구 테헤란로 152"] / ["850315-1234567"]
  - 잘못된 예: ["김", "철", "수"] / ["서", "울", "특", ...] / ["8", "5", "0", ...]
- 해당 카테고리에 PII가 없으면 반드시 null로 표시하세요.
- 동일 인물의 이름이 여러 번 등장해도 한 번만 추출합니다.

## PII가 아닌 것 (반드시 제외)

### 이름 제외
- 회사명, 기관명, 부서명, 팀명 (삼성전자, 한국전력, 마케팅팀 등)
- 공인/공적 인물: 뉴스 기사 속 대통령, 장관, 국회의원, 대기업 CEO 등 공인의 이름
- 학술 논문 저자: 참고문헌/인용에 등장하는 저자명 (Kim et al., Bolukbasi 등)
- 소설/영화 등 창작물의 허구적 인물 이름
- 프로그래밍 코드 내 테스트용 더미 데이터로 사용된 이름

### 주소 제외
- 지명, 관광지, 건물 이름, 랜드마크 (강남역 사거리, 남산타워 등)
- 법원명, 기관 소재지 (서울중앙지방법원 등)
- 통계/분석 목적의 지역명 (서울 강남구: 45명)

### 전화번호 제외
- 기업 대표번호: 1588-xxxx, 1577-xxxx, 1566-xxxx, 1544-xxxx 등
- 수신자 부담번호: 080-xxxx-xxxx
- 고객센터/ARS 번호

### IP주소 제외
- 프로그래밍 코드 내 테스트용 IP (127.0.0.1, localhost, 0.0.0.0)
- 공용 DNS 서버 (8.8.8.8, 1.1.1.1 등)
- 기술 문서의 네트워크 대역 설명용 예시 (10.0.0.0/8 등)

### 이메일 제외
- 테스트/예시용 이메일 (test@example.com, user@test.com 등)
- 프로그래밍 코드 내 더미 이메일

### 생년월일 제외 (매우 중요)
- 문서 작성일, 발령일, 계약일, 접수일, 발행일 등 **업무/문서 날짜**는 생년월일이 아닙니다.
- "2024년 3월 1일" 같은 날짜가 문서 내에 있어도 "생년월일", "출생", "생일", "DOB" 등의 레이블과 연결되지 않으면 절대 생년월일로 추출하지 마세요.
- 주민등록번호의 앞 6자리(생년월일 부분)는 주민등록번호로만 분류하고, 생년월일로 중복 추출하지 마세요.

### 기타_고유식별정보 제외 (매우 중요)
- 이 카테고리는 **학번, 차량번호** 등 극히 제한된 유형만 해당합니다.
- 다음은 기타_고유식별정보가 **아닙니다**:
  - 주문번호, 접수번호, 사건번호, 관리번호, 증권번호, 보험증권번호
  - 사업자등록번호 (기업 식별번호이지 개인 식별번호가 아님)
  - 사원번호, 직원번호 (EMP-xxxxx 등)
  - 제품코드, 모델번호, 소프트웨어 버전번호
  - 매출액, 비용, 예산 등 재무 수치
  - 법률 조항 번호 (제3조, 민법 제750조 등)
  - 통계 수치, 비율, 퍼센트
- 확신이 없으면 null로 두세요. 과잉 검출보다 미검출이 낫습니다.

## 출력 예시

문서: "담당자 김철수(010-1234-5678, chulsoo@company.com)에게 서울특별시 강남구 테헤란로 152로 서류를 보내주세요."
→ {"이름": ["김철수"], "주소": ["서울특별시 강남구 테헤란로 152"], "주민등록번호": null, "여권번호": null, "운전면허번호": null, "이메일": ["chulsoo@company.com"], "IP주소": null, "전화번호": ["010-1234-5678"], "계좌번호": null, "카드번호": null, "생년월일": null, "기타_고유식별정보": null}

문서: "계약자: 이영희, 주민등록번호 900101-2345678, 국민은행 계좌 123-456-789012로 입금 바랍니다."
→ {"이름": ["이영희"], "주소": null, "주민등록번호": ["900101-2345678"], "여권번호": null, "운전면허번호": null, "이메일": null, "IP주소": null, "전화번호": null, "계좌번호": ["123-456-789012"], "카드번호": null, "생년월일": null, "기타_고유식별정보": null}

문서: "2024년 인사발령 통보서. 발령일자: 2024년 3월 1일. 성명: 박민호"
→ {"이름": ["박민호"], "주소": null, "주민등록번호": null, "여권번호": null, "운전면허번호": null, "이메일": null, "IP주소": null, "전화번호": null, "계좌번호": null, "카드번호": null, "생년월일": null, "기타_고유식별정보": null}
"""

USER_PROMPT_TEMPLATE = """아래 문서에서 개인정보(PII)를 검출하여 JSON으로 응답하세요.

주의사항:
- 각 PII 항목은 반드시 완전한 문자열 하나로 추출 (글자 단위 쪼개기 금지)
- 문서 날짜(작성일, 발령일 등)는 생년월일이 아님
- 기타_고유식별정보는 학번/차량번호만 해당, 주문번호·사업자등록번호·사원번호 등은 제외
- 공인, 논문저자, 허구인물, 코드 내 더미 데이터의 이름은 PII 아님

---
{document_text}
---"""


# ============================================================================
# 4. expected_pii → 카테고리별 dict 변환
# ============================================================================

def normalize_expected(expected_pii: list[dict]) -> dict[str, list[str] | None]:
    result: dict[str, list[str]] = defaultdict(list)
    for item in expected_pii:
        normalized = TYPE_NORMALIZATION.get(item["type"], "기타_고유식별정보")
        result[normalized].append(item["value"])

    output: dict[str, list[str] | None] = {}
    for cat in PII_CATEGORIES:
        output[cat] = sorted(set(result[cat])) if cat in result else None
    return output


# ============================================================================
# 5. 평가 메트릭
# ============================================================================

def compute_metrics(
    expected: dict[str, list[str] | None],
    predicted: dict[str, list[str] | None],
) -> dict[str, Any]:
    per_category: dict[str, dict] = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for cat in PII_CATEGORIES:
        exp_vals = expected[cat]
        pred_vals = predicted.get(cat)
        exp_set = set(exp_vals) if exp_vals else set()
        pred_set = set(pred_vals) if pred_vals else set()

        tp = len(exp_set & pred_set)
        fp = len(pred_set - exp_set)
        fn = len(exp_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(exp_set) == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(pred_set) == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        exp_exists = exp_vals is not None and len(exp_vals) > 0
        pred_exists = pred_vals is not None and len(pred_vals) > 0

        per_category[cat] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "expected_count": len(exp_set),
            "predicted_count": len(pred_set),
            "category_detection_correct": exp_exists == pred_exists,
            "missing": sorted(exp_set - pred_set) if exp_set - pred_set else [],
            "extra": sorted(pred_set - exp_set) if pred_set - exp_set else [],
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # TP=FP=FN=0 means both expected and predicted are empty → perfect match
    if total_tp == 0 and total_fp == 0 and total_fn == 0:
        micro_p, micro_r, micro_f1 = 1.0, 1.0, 1.0
    else:
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "per_category": per_category,
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


# ============================================================================
# 6. 리포트 출력
# ============================================================================

def print_report(all_results: list[dict]) -> dict:
    cat_agg: dict[str, dict] = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in PII_CATEGORIES}
    diff_agg: dict[str, dict] = {d: {"tp": 0, "fp": 0, "fn": 0, "count": 0} for d in ["EASY", "MEDIUM", "HARD"]}
    failed_cases: list[dict] = []

    for r in all_results:
        metrics = r["metrics"]
        diff = r["difficulty"]
        diff_agg[diff]["count"] += 1
        for cat in PII_CATEGORIES:
            cm = metrics["per_category"][cat]
            cat_agg[cat]["tp"] += cm["tp"]
            cat_agg[cat]["fp"] += cm["fp"]
            cat_agg[cat]["fn"] += cm["fn"]
            diff_agg[diff]["tp"] += cm["tp"]
            diff_agg[diff]["fp"] += cm["fp"]
            diff_agg[diff]["fn"] += cm["fn"]
        if metrics["micro_f1"] < 1.0:
            failed_cases.append(r)

    print("\n" + "=" * 80)
    print("카테고리별 성능")
    print("=" * 80)
    print(f"{'카테고리':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    print("-" * 80)
    for cat in PII_CATEGORIES:
        a = cat_agg[cat]
        p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) > 0 else 0.0
        rc = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) > 0 else 0.0
        f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
        print(f"{cat:<20s} {p:>10.2%} {rc:>10.2%} {f1:>10.2%} {a['tp']:>6d} {a['fp']:>6d} {a['fn']:>6d}")

    print("\n" + "=" * 80)
    print("난이도별 성능")
    print("=" * 80)
    print(f"{'난이도':<10s} {'케이스수':>8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 80)
    for diff in ["EASY", "MEDIUM", "HARD"]:
        a = diff_agg[diff]
        p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) > 0 else 0.0
        rc = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) > 0 else 0.0
        f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
        print(f"{diff:<10s} {a['count']:>8d} {p:>10.2%} {rc:>10.2%} {f1:>10.2%}")

    total_tp = sum(a["tp"] for a in cat_agg.values())
    total_fp = sum(a["fp"] for a in cat_agg.values())
    total_fn = sum(a["fn"] for a in cat_agg.values())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"전체 Micro-Average: P={overall_p:.2%}  R={overall_r:.2%}  F1={overall_f1:.2%}")
    print(f"  총 TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  테스트 케이스: {len(all_results)}개 중 {len(all_results) - len(failed_cases)}개 완벽 통과")
    print("=" * 80)

    if failed_cases:
        print("\n주요 실패 케이스 (F1 낮은 순):")
        failed_cases.sort(key=lambda x: x["metrics"]["micro_f1"])
        for r in failed_cases[:10]:
            m = r["metrics"]
            print(f"  [{r['id']}] {r['category']} ({r['difficulty']}) "
                  f"F1={m['micro_f1']:.2%}  FP={m['total_fp']}  FN={m['total_fn']}")
            for cat in PII_CATEGORIES:
                exp = r["expected"].get(cat)
                pred = r["predicted"].get(cat)
                if exp is None and pred is None:
                    continue
                status = "OK" if exp == pred else "MISS"
                print(f"    [{status}] {cat}:")
                print(f"      정답: {exp}")
                print(f"      예측: {pred}")

    return {
        "total_cases": len(all_results),
        "perfect_cases": len(all_results) - len(failed_cases),
        "overall_precision": round(overall_p, 4),
        "overall_recall": round(overall_r, 4),
        "overall_f1": round(overall_f1, 4),
        "category_metrics": {
            cat: {
                "precision": round(a["tp"] / (a["tp"] + a["fp"]), 4) if (a["tp"] + a["fp"]) > 0 else 0.0,
                "recall": round(a["tp"] / (a["tp"] + a["fn"]), 4) if (a["tp"] + a["fn"]) > 0 else 0.0,
            }
            for cat, a in cat_agg.items()
        },
    }


# ============================================================================
# 7. API 요청
# ============================================================================

def call_api(
    client: OpenAI,
    model: str,
    tc: dict,
    json_schema: dict,
    temperature: float,
    max_tokens: int,
    no_think: bool = False,
) -> dict:
    """단일 테스트 케이스에 대해 API 요청을 보내고 결과를 반환"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(document_text=tc["document_text"])},
    ]

    extra_body: dict[str, Any] = {"guided_json": json_schema}
    if no_think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        predicted = json.loads(raw_text)
    except json.JSONDecodeError:
        predicted: dict[str, list[str] | None] = {cat: None for cat in PII_CATEGORIES}

    expected = normalize_expected(tc["expected_pii"])
    metrics = compute_metrics(expected, predicted)

    return {
        "id": tc["id"],
        "category": tc["category"],
        "difficulty": tc["difficulty"],
        "intent": tc["intent"],
        "expected": expected,
        "predicted": predicted,
        "metrics": metrics,
    }


# ============================================================================
# 8. 메인
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PII 검출 성능 평가 (vLLM API 클라이언트)")
    parser.add_argument("--model", type=str, required=True,
                        help="모델 이름 (서버에 로드된 모델과 일치해야 함)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM 서버 API URL (default: http://localhost:8000/v1)")
    parser.add_argument("--api-key", type=str, default="dummy",
                        help="API key (vLLM 기본값: dummy)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="동시 요청 수 (default: 10)")
    parser.add_argument("--test-cases", type=str, default=None,
                        help="테스트 케이스 JSON 파일 경로")
    parser.add_argument("--category", type=str, default=None,
                        help="특정 카테고리만 테스트 (예: 이름, 주소)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["EASY", "MEDIUM", "HARD"])
    parser.add_argument("--ids", type=str, nargs="+", default=None,
                        help="특정 ID만 실행 (예: TC001 TC074)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="결과 저장 경로 (default: results.json)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true",
                        help="케이스별 expected/predicted 상세 출력")
    parser.add_argument("--no-think", action="store_true",
                        help="Qwen3 thinking 모드 비활성화 (guided_json 충돌 방지)")
    args = parser.parse_args()

    # ── 테스트 케이스 로드 ──
    if args.test_cases:
        tc_path = Path(args.test_cases)
    else:
        tc_path = Path(__file__).parent / "all_test_cases.json"
    with open(tc_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    if args.category:
        test_cases = [tc for tc in test_cases if args.category in tc["category"]]
    if args.difficulty:
        test_cases = [tc for tc in test_cases if tc["difficulty"] == args.difficulty]
    if args.ids:
        id_set = set(args.ids)
        test_cases = [tc for tc in test_cases if tc["id"] in id_set]

    if not test_cases:
        print("필터 조건에 맞는 테스트 케이스가 없습니다.")
        return

    print(f"대상 테스트 케이스: {len(test_cases)}개")
    print(f"모델: {args.model}")
    print(f"API URL: {args.api_url}")
    print(f"동시 요청 수: {args.concurrency}")

    # ── OpenAI 클라이언트 ──
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    json_schema = build_json_schema()

    # ── 병렬 요청 ──
    print(f"\n추론 시작 ({len(test_cases)}개)...")
    start_time = time.time()
    all_results: list[dict] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                call_api, client, args.model, tc, json_schema,
                args.temperature, args.max_tokens, args.no_think,
            ): tc["id"]
            for tc in test_cases
        }

        for future in as_completed(futures):
            tc_id = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"  [{tc_id}] 요청 실패: {e}")
                # 실패한 케이스는 빈 예측으로 처리
                tc = next(t for t in test_cases if t["id"] == tc_id)
                predicted: dict[str, list[str] | None] = {cat: None for cat in PII_CATEGORIES}
                expected = normalize_expected(tc["expected_pii"])
                all_results.append({
                    "id": tc_id,
                    "category": tc["category"],
                    "difficulty": tc["difficulty"],
                    "intent": tc["intent"],
                    "expected": expected,
                    "predicted": predicted,
                    "metrics": compute_metrics(expected, predicted),
                })
            completed += 1
            print(f"\r  진행: {completed}/{len(test_cases)}", end="", flush=True)

    elapsed = time.time() - start_time
    # ID 순 정렬
    all_results.sort(key=lambda x: x["id"])
    print(f"\n추론 완료! ({elapsed:.1f}초, 평균 {elapsed/len(test_cases):.2f}초/케이스)\n")

    # ── verbose: 케이스별 상세 ──
    if args.verbose:
        print("=" * 80)
        print("케이스별 상세 결과")
        print("=" * 80)
        for r in all_results:
            m = r["metrics"]
            f1 = m["micro_f1"]
            status = "PASS" if f1 == 1.0 else "FAIL"
            print(f"\n[{r['id']}] {r['category']} ({r['difficulty']}) - {status} (F1={f1:.2%})")
            for cat in PII_CATEGORIES:
                exp = r["expected"].get(cat)
                pred = r["predicted"].get(cat)
                if exp is None and pred is None:
                    continue
                match = "==" if exp == pred else "!="
                print(f"  {cat}:")
                print(f"    정답: {exp}")
                print(f"    예측: {pred}  {match}")
        print()

    # ── 리포트 ──
    summary = print_report(all_results)

    # ── 결과 저장 (기본: results.json) ──
    output_data = {
        "model": args.model,
        "api_url": args.api_url,
        "concurrency": args.concurrency,
        "inference_time_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "results": [
            {
                "id": r["id"],
                "category": r["category"],
                "difficulty": r["difficulty"],
                "intent": r["intent"],
                "f1": r["metrics"]["micro_f1"],
                "expected": {k: v for k, v in r["expected"].items() if v is not None},
                "predicted": {k: v for k, v in r["predicted"].items() if v is not None},
            }
            for r in all_results
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
