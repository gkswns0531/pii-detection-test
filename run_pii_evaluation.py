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
import random
import statistics
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
    """vLLM response_format에 전달할 JSON Schema"""
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
        "type": "json_schema",
        "json_schema": {
            "name": "pii_detection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": PII_CATEGORIES,
                "additionalProperties": False,
            },
        },
    }


# ============================================================================
# 3. 프롬프트
# ============================================================================

SYSTEM_PROMPT = """당신은 문서에서 개인정보(PII)를 검출하는 전문가입니다.

주어진 문서를 분석하여 아래 카테고리별로 개인정보를 검출해 주세요.

## 검출 카테고리
1. **이름**: 실제 특정 개인의 이름. 한글, 영문, 한자 모두 포함. 같은 사람의 한글명+영문명+한자명이 병기되면 각각 추출. 부분 마스킹(김○수)도 포함. 희귀 성씨(갈, 괴, 피, 라, 봉, 견, 혁, 인, 온, 옥, 란, 탁, 도, 독고, 남궁, 황보, 제갈, 선우, 사공 등)도 반드시 검출.
2. **주소**: 특정 개인이나 사업장을 식별할 수 있는 도로명/지번 주소, 해외 주소(영문 포함). 우편번호 포함 가능. 등록지, 전출지, 전입지, 배송지, 보험 목적물 소재지, 근무지 등 어떤 레이블이든 구체적 주소이면 모두 검출.
3. **주민등록번호**: 주민등록번호(YYMMDD-NNNNNNN). 외국인등록번호(뒷자리 5~8 시작)도 여기에 포함. 마스킹(*)/변형/앞6자리만 노출된 경우도 포함.
4. **여권번호**: 한국(M+8자리) 및 외국 여권번호.
5. **운전면허번호**: 한국 운전면허번호(NN-NN-NNNNNN-NN).
6. **이메일**: 이메일 주소. 난독화([at],[dot])나 마스킹 형태도 포함.
7. **IP주소**: 특정 장비/사용자를 식별할 수 있는 IPv4, IPv6, CIDR 주소. 원문에서 IP와 포트가 콜론(:)으로 직접 연결된 형태(예: 10.0.0.1:8080, 211.45.67.89:443)는 포트까지 포함하여 하나의 값으로 추출. 원문에 IP만 있고 포트가 별도로 기재된 경우(예: "192.168.1.1 ... 포트 3306")는 IP만 추출.
8. **전화번호**: 특정 개인의 휴대폰, 유선전화, 국제번호. 내선번호가 있으면 "02-3456-7001 내선 1001" 형태 전체를 하나로 추출.
9. **계좌번호**: 은행 계좌번호, 가상계좌, IBAN 포함.
10. **카드번호**: 신용/체크카드 번호, 암호화폐 지갑 주소(BTC, ETH 등) 포함.
11. **생년월일**: "생년월일", "출생", "생일", "DOB", "생년월일:" 등의 레이블과 **직접 연결된** 날짜만 해당.
12. **기타_고유식별정보**: **학번**(예: 2023-10315), **차량번호**(예: 123가 4567), **군번**(예: 19-70123456), **사번**(예: 2018-001234), **현관/도어락 비밀번호**(예: #1234*)가 해당. 이 유형들 외에는 절대 이 카테고리에 넣지 마세요.

## 추출 규칙
- 검출된 PII는 문서에 나타난 **원문 그대로** 추출하세요. OCR 오류가 있더라도 절대 교정하지 말고 원문 그대로 추출 (예: 긤철수 → 긤철수로 추출, 김철수로 교정 금지 / O1O-I234-5G78 → 그대로 추출, 010-1234-5678로 교정 금지).
- 각 PII 항목은 **완전한 문자열 하나**로 추출해야 합니다. 절대 글자 단위로 쪼개지 마세요.
  - 올바른 예: ["김철수"] / ["서울특별시 강남구 테헤란로 152"] / ["850315-1234567"]
  - 잘못된 예: ["김", "철", "수"] / ["서", "울", "특", ...] / ["8", "5", "0", ...]
- 해당 카테고리에 PII가 없으면 반드시 null로 표시하세요.
- 동일 인물의 이름이 여러 번 등장해도 한 번만 추출합니다.
- 계좌번호, 카드번호 추출 시 은행명/카드사명은 포함하지 말고 번호만 추출 (예: "국민은행 123-456-789" → "123-456-789")

## PII가 아닌 것 (반드시 제외)

### 이름 제외
- 회사명, 법인명, 기관명, 부서명, 팀명 (삼성전자, 한영회계법인, 한국전력 등)
- 직위, 직책, 직함 단독 사용 (팀장, 부장, 과장, 대리, 센터장, 이사, 정보보안팀장 등) — "승인자: 정보보안팀장"처럼 직책만 있고 이름이 없는 경우 이름이 아님
- 학술 논문/참고문헌의 저자명: 인용 형식(Kim et al., Bolukbasi 등)이나 참고문헌 목록에 등장하는 저자의 성(surname)이나 이니셜은 학술 공개 정보이므로 이름으로 검출하지 마세요
- 프로그래밍 코드 내 테스트용 더미 데이터

### 주소 제외
- 지명, 관광지, 건물 이름, 랜드마크 (강남역, 남산타워 등)
- 시·군·구 등 행정구역명만으로 된 일반 위치 설명 (예: "제주 서귀포시에 위치한", "서울 강남구 일대", "강서구에 거주" → 구체적 도로명/지번이 없으므로 주소 아님)
- 법원명, 기관 소재지 (서울중앙지방법원 등)
- 통계/분석 목적의 지역명 (서울 강남구: 45명)
- "위 원고 1과 같음", "위 학생과 동일", "상동", "동일" 등 다른 곳을 참조하는 표현은 실제 주소가 아님. "위 ~와/과 같음/동일" 패턴은 모두 해당.

### 전화번호 제외
- 기업 대표번호, 전국대표번호: 1588-xxxx, 1577-xxxx, 1566-xxxx, 1544-xxxx, 1661-xxxx, 1600-xxxx, 1522-xxxx 등
- 수신자 부담번호: 080-xxxx-xxxx
- 고객센터/ARS 번호
- 호텔/숙박업소/식당 등 사업장 대표번호, 프론트 데스크 번호

### IP주소 제외
- 프로그래밍 코드 내 테스트용 IP (127.0.0.1, localhost, 0.0.0.0)
- 공용 DNS 서버 (8.8.8.8, 1.1.1.1)
- 네트워크 설계/아키텍처 문서에서 대역 할당 계획, 라우팅 정책, 서브넷 설계 등에 쓰인 IP 주소 및 CIDR 대역 (10.0.0.0/16, 172.16.0.0/24, 192.168.x.x/xx, 게이트웨이 IP 등). 단, 보안 로그, 접근제어(ACL), 방화벽 정책에서 특정 사무실/장비/사용자를 식별하는 IP는 PII.

### 이메일 제외
- 테스트/예시용 이메일 (test@example.com, user@test.com 등)
- 프로그래밍 코드 내 더미 이메일

### 생년월일 제외 (매우 중요)
- 문서 작성일, 발령일, 계약일, 접수일, 발행일, 유효기간, 만료일 등 **업무/문서 날짜**는 생년월일이 아닙니다.
- 여권/면허증의 발급일, 만료일도 생년월일이 아닙니다.
- "생년월일", "출생일", "생일", "DOB" 등의 레이블이 **명시적으로** 앞에 붙은 경우만 추출.
- 주민등록번호의 앞 6자리는 주민등록번호로만 분류.

### 기타_고유식별정보 제외 (매우 중요)
- 이 카테고리는 **학번**, **차량번호**, **군번**, **사번**, **현관/도어락 비밀번호**만 해당합니다.
- 다음은 기타_고유식별정보가 **절대 아닙니다**:
  - 주문번호, 접수번호, 사건번호, 관리번호, 진료번호, 보험증권번호
  - 사업자등록번호, 법인등록번호
  - 사용자ID (USR-xxxxx 등), 회원ID
  - 제품코드, 모델번호, 시리얼번호, 장비코드, 소프트웨어 버전
  - 운전면허번호와 유사한 형식의 장비 번호
  - 매출액, 비용, 예산 등 재무 수치
  - 법률 조항 번호 (제3조, 민법 제750조 등)
- 위 해당 유형이 아니면 null로 두세요.

## 출력 예시

문서: "담당자 김철수(010-1234-5678, chulsoo@company.com)에게 서울특별시 강남구 테헤란로 152로 서류를 보내주세요."
→ {"이름": ["김철수"], "주소": ["서울특별시 강남구 테헤란로 152"], "주민등록번호": null, "여권번호": null, "운전면허번호": null, "이메일": ["chulsoo@company.com"], "IP주소": null, "전화번호": ["010-1234-5678"], "계좌번호": null, "카드번호": null, "생년월일": null, "기타_고유식별정보": null}

문서: "계약자: 이영희(李美英), 주민등록번호 900101-2345678, 국민은행 계좌 123-456-789012."
→ {"이름": ["이영희", "李美英"], "주소": null, "주민등록번호": ["900101-2345678"], "여권번호": null, "운전면허번호": null, "이메일": null, "IP주소": null, "전화번호": null, "계좌번호": ["123-456-789012"], "카드번호": null, "생년월일": null, "기타_고유식별정보": null}

문서: "2024년 인사발령 통보서. 발령일자: 2024년 3월 1일. 성명: 박민호"
→ {"이름": ["박민호"], "주소": null, "주민등록번호": null, "여권번호": null, "운전면허번호": null, "이메일": null, "IP주소": null, "전화번호": null, "계좌번호": null, "카드번호": null, "생년월일": null, "기타_고유식별정보": null}
"""

USER_PROMPT_TEMPLATE = """아래 문서에서 개인정보(PII)를 검출하여 JSON으로 응답하세요.

핵심 규칙:
- 원문 그대로 추출 (OCR 오류도 절대 교정하지 말 것 - 긤철수→긤철수, O1O→O1O)
- 완전한 문자열 하나로 추출 (글자 단위 쪼개기 금지)
- 문서 날짜 ≠ 생년월일 (생년월일/출생일/DOB 레이블이 붙은 것만)
- 기타_고유식별정보 = 학번, 차량번호, 군번, 사번, 현관비밀번호만 (그 외 모든 번호는 null)
- 한자 이름(金賢洙)도 검출, 영문명(Hayoon Jeong, Thomas Mueller)도 검출, Mr./Ms. 뒤의 영문명(Attn: Mr. Park Jinho → Park Jinho)도 검출
- 희귀 성씨(갈, 괴, 피, 라, 봉, 견, 혁, 인, 온, 옥, 란, 탁, 도, 독고, 남궁 등)도 반드시 이름으로 검출
- 등록지, 보험 목적물, 전출지, 배송지 등 문서 내 모든 구체적 주소 검출
- IP:포트가 콜론(:)으로 직접 연결된 경우(10.0.0.1:8080) 포트까지 포함하여 추출, IP만 있으면 IP만
- 학술 논문 인용(Kim et al.)이나 참고문헌 저자는 이름이 아님
- 직위/직책(팀장, 부장 등)만 있고 이름이 없으면 이름 아님
- 호텔/사업장 대표번호, 1661-xxxx 등 서비스번호는 전화번호 아님

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
        exp_set = set(v.strip() for v in exp_vals) if exp_vals else set()
        pred_set = set(v.strip() for v in pred_vals) if pred_vals else set()

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
    diff_agg: dict[str, dict] = {d: {"tp": 0, "fp": 0, "fn": 0, "count": 0, "perfect": 0} for d in ["EASY", "MEDIUM", "HARD"]}
    failed_cases: list[dict] = []

    for r in all_results:
        metrics = r["metrics"]
        diff = r["difficulty"]
        diff_agg[diff]["count"] += 1
        is_perfect = metrics["micro_f1"] == 1.0
        if is_perfect:
            diff_agg[diff]["perfect"] += 1
        for cat in PII_CATEGORIES:
            cm = metrics["per_category"][cat]
            cat_agg[cat]["tp"] += cm["tp"]
            cat_agg[cat]["fp"] += cm["fp"]
            cat_agg[cat]["fn"] += cm["fn"]
            diff_agg[diff]["tp"] += cm["tp"]
            diff_agg[diff]["fp"] += cm["fp"]
            diff_agg[diff]["fn"] += cm["fn"]
        if not is_perfect:
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
    print(f"{'난이도':<10s} {'케이스수':>8s} {'Acc':>8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 80)
    for diff in ["EASY", "MEDIUM", "HARD"]:
        a = diff_agg[diff]
        acc = a["perfect"] / a["count"] if a["count"] > 0 else 0.0
        p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) > 0 else 0.0
        rc = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) > 0 else 0.0
        f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
        print(f"{diff:<10s} {a['count']:>8d} {acc:>8.2%} {p:>10.2%} {rc:>10.2%} {f1:>10.2%}")

    total_tp = sum(a["tp"] for a in cat_agg.values())
    total_fp = sum(a["fp"] for a in cat_agg.values())
    total_fn = sum(a["fn"] for a in cat_agg.values())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    perfect_count = len(all_results) - len(failed_cases)
    overall_acc = perfect_count / len(all_results) if all_results else 0.0

    print("\n" + "=" * 80)
    print(f"전체 Micro-Average: P={overall_p:.2%}  R={overall_r:.2%}  F1={overall_f1:.2%}  Acc={overall_acc:.2%}")
    print(f"  총 TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  테스트 케이스: {len(all_results)}개 중 {perfect_count}개 완벽 통과 (Acc={overall_acc:.2%})")
    print("=" * 80)

    # ── Classification Confusion Matrix (카테고리 존재 여부 이진 분류) ──
    cls_agg: dict[str, dict] = {cat: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for cat in PII_CATEGORIES}
    for r in all_results:
        for cat in PII_CATEGORIES:
            exp = r["expected"].get(cat)
            pred = r["predicted"].get(cat)
            has_exp = exp is not None and len(exp) > 0
            has_pred = pred is not None and len(pred) > 0
            if has_exp and has_pred:
                cls_agg[cat]["tp"] += 1
            elif not has_exp and not has_pred:
                cls_agg[cat]["tn"] += 1
            elif not has_exp and has_pred:
                cls_agg[cat]["fp"] += 1
            else:  # has_exp and not has_pred
                cls_agg[cat]["fn"] += 1

    print("\n" + "=" * 80)
    print("카테고리별 Classification Confusion Matrix (카테고리 존재 여부 이진 분류)")
    print("=" * 80)
    print(f"{'카테고리':<20s} {'TP':>5s} {'TN':>5s} {'FP':>5s} {'FN':>5s} {'P':>8s} {'R':>8s} {'F1':>8s} {'Acc':>8s}")
    print("-" * 80)
    cls_total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    cls_cat_metrics = {}
    for cat in PII_CATEGORIES:
        c = cls_agg[cat]
        cls_total["tp"] += c["tp"]
        cls_total["tn"] += c["tn"]
        cls_total["fp"] += c["fp"]
        cls_total["fn"] += c["fn"]
        cp = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 1.0
        cr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 1.0
        cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        c_total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        ca = (c["tp"] + c["tn"]) / c_total if c_total > 0 else 0.0
        cls_cat_metrics[cat] = {"precision": round(cp, 4), "recall": round(cr, 4), "f1": round(cf1, 4), "accuracy": round(ca, 4)}
        print(f"{cat:<20s} {c['tp']:>5d} {c['tn']:>5d} {c['fp']:>5d} {c['fn']:>5d} {cp:>8.2%} {cr:>8.2%} {cf1:>8.2%} {ca:>8.2%}")

    ct = cls_total
    ct_p = ct["tp"] / (ct["tp"] + ct["fp"]) if (ct["tp"] + ct["fp"]) > 0 else 0.0
    ct_r = ct["tp"] / (ct["tp"] + ct["fn"]) if (ct["tp"] + ct["fn"]) > 0 else 0.0
    ct_f1 = 2 * ct_p * ct_r / (ct_p + ct_r) if (ct_p + ct_r) > 0 else 0.0
    ct_total = ct["tp"] + ct["tn"] + ct["fp"] + ct["fn"]
    ct_acc = (ct["tp"] + ct["tn"]) / ct_total if ct_total > 0 else 0.0
    print("-" * 80)
    print(f"{'전체':<20s} {ct['tp']:>5d} {ct['tn']:>5d} {ct['fp']:>5d} {ct['fn']:>5d} {ct_p:>8.2%} {ct_r:>8.2%} {ct_f1:>8.2%} {ct_acc:>8.2%}")
    print("=" * 80)

    # ── 난이도별 Classification Confusion Matrix ──
    cls_diff_agg: dict[str, dict] = {d: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "count": 0} for d in ["EASY", "MEDIUM", "HARD"]}
    for r in all_results:
        diff = r["difficulty"]
        for cat in PII_CATEGORIES:
            exp = r["expected"].get(cat)
            pred = r["predicted"].get(cat)
            has_exp = exp is not None and len(exp) > 0
            has_pred = pred is not None and len(pred) > 0
            if has_exp and has_pred:
                cls_diff_agg[diff]["tp"] += 1
            elif not has_exp and not has_pred:
                cls_diff_agg[diff]["tn"] += 1
            elif not has_exp and has_pred:
                cls_diff_agg[diff]["fp"] += 1
            else:
                cls_diff_agg[diff]["fn"] += 1
        cls_diff_agg[diff]["count"] += 1

    print("\n" + "=" * 80)
    print("난이도별 Classification Confusion Matrix")
    print("=" * 80)
    print(f"{'난이도':<10s} {'케이스':>6s} {'TP':>5s} {'TN':>5s} {'FP':>5s} {'FN':>5s} {'P':>8s} {'R':>8s} {'F1':>8s} {'Acc':>8s}")
    print("-" * 80)
    cls_diff_metrics = {}
    for diff in ["EASY", "MEDIUM", "HARD"]:
        c = cls_diff_agg[diff]
        dp = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 1.0
        dr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 1.0
        df1 = 2 * dp * dr / (dp + dr) if (dp + dr) > 0 else 0.0
        d_total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        da = (c["tp"] + c["tn"]) / d_total if d_total > 0 else 0.0
        cls_diff_metrics[diff] = {"precision": round(dp, 4), "recall": round(dr, 4), "f1": round(df1, 4), "accuracy": round(da, 4)}
        print(f"{diff:<10s} {c['count']:>6d} {c['tp']:>5d} {c['tn']:>5d} {c['fp']:>5d} {c['fn']:>5d} {dp:>8.2%} {dr:>8.2%} {df1:>8.2%} {da:>8.2%}")
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
                status = "OK" if set(exp or []) == set(pred or []) else "MISS"
                print(f"    [{status}] {cat}:")
                print(f"      정답: {exp}")
                print(f"      예측: {pred}")

    return {
        "total_cases": len(all_results),
        "perfect_cases": perfect_count,
        "overall_accuracy": round(overall_acc, 4),
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
        "classification_confusion_matrix": {
            "per_category": cls_cat_metrics,
            "per_difficulty": cls_diff_metrics,
            "total": {
                "tp": ct["tp"], "tn": ct["tn"], "fp": ct["fp"], "fn": ct["fn"],
                "precision": round(ct_p, 4), "recall": round(ct_r, 4),
                "f1": round(ct_f1, 4), "accuracy": round(ct_acc, 4),
            },
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
    eval_categories: list[str] | None = None,
) -> dict:
    """단일 테스트 케이스에 대해 API 요청을 보내고 결과를 반환"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(document_text=tc["document_text"])},
    ]

    extra_body: dict[str, Any] = {}
    if no_think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=json_schema,
        **({"extra_body": extra_body} if extra_body else {}),
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        predicted = json.loads(raw_text)
    except json.JSONDecodeError:
        predicted: dict[str, list[str] | None] = {cat: None for cat in PII_CATEGORIES}

    # eval_categories가 지정된 경우, 해당 카테고리 외의 예측은 무시 (None 처리)
    if eval_categories:
        for cat in PII_CATEGORIES:
            if cat not in eval_categories:
                predicted[cat] = None

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
        "raw_response": raw_text,
    }


# ============================================================================
# 8. 레이턴시 측정
# ============================================================================

def run_latency_test(args):
    """레이턴시 측정 모드: 3회 워밍업 + 10회 측정, batch_size=1, ~2K token document input"""

    NUM_WARMUP = 3
    NUM_MEASURE = 10
    TOTAL = NUM_WARMUP + NUM_MEASURE
    TARGET_DOC_CHARS = 1000  # ~2K tokens (한국어 평균 ~2 tok/char)

    # ── 테스트 케이스 로드 ──
    if args.test_cases:
        tc_path = Path(args.test_cases)
    else:
        tc_path = Path(__file__).parent / "combined_test_cases.json"
    with open(tc_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    random.seed(42)
    random.shuffle(test_cases)

    # ── 13개 서로 다른 ~2K token 입력 생성 ──
    inputs: list[str] = []
    idx = 0
    for _ in range(TOTAL):
        combined: list[str] = []
        chars = 0
        while idx < len(test_cases):
            doc = test_cases[idx]["document_text"]
            combined.append(doc)
            chars += len(doc)
            idx += 1
            if chars >= TARGET_DOC_CHARS:
                break
        if not combined:
            print(f"Error: 테스트 케이스 부족 ({len(inputs)}개만 생성)")
            return
        inputs.append("\n\n---\n\n".join(combined))

    print(f"모델: {args.model}")
    print(f"레이턴시 측정 모드 (batch_size=1, ~2K token document input)")
    print(f"워밍업: {NUM_WARMUP}회, 측정: {NUM_MEASURE}회")
    print(f"생성된 입력: {TOTAL}개 (각 ~{TARGET_DOC_CHARS}자, 모두 서로 다른 입력)")

    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    json_schema = build_json_schema()

    def send_request(doc_text: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(document_text=doc_text)},
        ]
        extra_body: dict[str, Any] = {}
        if args.no_think:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        return client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            response_format=json_schema,
            **({"extra_body": extra_body} if extra_body else {}),
        )

    # ── 워밍업 ──
    print(f"\n워밍업 ({NUM_WARMUP}회)...")
    for i in range(NUM_WARMUP):
        start = time.time()
        resp = send_request(inputs[i])
        elapsed = time.time() - start
        u = resp.usage
        print(f"  워밍업 {i+1}: {elapsed:.3f}s "
              f"(prompt: {u.prompt_tokens} tok, completion: {u.completion_tokens} tok)")

    # ── 측정 ──
    print(f"\n레이턴시 측정 ({NUM_MEASURE}회)...")
    measurements: list[dict] = []

    for i in range(NUM_MEASURE):
        start = time.time()
        resp = send_request(inputs[NUM_WARMUP + i])
        elapsed = time.time() - start
        u = resp.usage

        m = {
            "run": i + 1,
            "latency_sec": round(elapsed, 4),
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
        measurements.append(m)
        print(f"  측정 {i+1:>2d}: {elapsed:.3f}s "
              f"(prompt: {u.prompt_tokens} tok, completion: {u.completion_tokens} tok)")

    # ── 통계 ──
    lats = [m["latency_sec"] for m in measurements]
    sorted_lats = sorted(lats)

    def percentile(vals: list[float], p: float) -> float:
        k = (len(vals) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        return vals[f] + (k - f) * (vals[c] - vals[f])

    mean_lat = statistics.mean(lats)
    median_lat = statistics.median(lats)
    stdev_lat = statistics.stdev(lats) if len(lats) > 1 else 0.0
    min_lat = min(lats)
    max_lat = max(lats)
    p90 = percentile(sorted_lats, 90)
    p95 = percentile(sorted_lats, 95)
    p99 = percentile(sorted_lats, 99)

    avg_prompt = statistics.mean([m["prompt_tokens"] for m in measurements])
    avg_completion = statistics.mean([m["completion_tokens"] for m in measurements])

    print(f"\n{'='*60}")
    print(f"레이턴시 통계 (batch_size=1, ~2K token document input)")
    print(f"{'='*60}")
    print(f"  모델:       {args.model}")
    print(f"  측정 횟수:  {NUM_MEASURE}")
    print(f"  평균 입력:  {avg_prompt:.0f} tokens (system+user+document)")
    print(f"  평균 출력:  {avg_completion:.0f} tokens")
    print(f"{'─'*60}")
    print(f"  Mean:   {mean_lat:.4f}s")
    print(f"  Median: {median_lat:.4f}s")
    print(f"  StdDev: {stdev_lat:.4f}s")
    print(f"  Min:    {min_lat:.4f}s")
    print(f"  Max:    {max_lat:.4f}s")
    print(f"  P90:    {p90:.4f}s")
    print(f"  P95:    {p95:.4f}s")
    print(f"  P99:    {p99:.4f}s")
    print(f"{'='*60}")

    # ── 결과 저장 ──
    if args.output:
        output_data = {
            "model": args.model,
            "mode": "latency",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "batch_size": 1,
                "target_doc_tokens": 2000,
                "target_doc_chars": TARGET_DOC_CHARS,
                "num_warmup": NUM_WARMUP,
                "num_measure": NUM_MEASURE,
                "temperature": 0.0,
                "max_tokens": 2048,
                "no_think": args.no_think,
            },
            "statistics": {
                "mean_sec": round(mean_lat, 4),
                "median_sec": round(median_lat, 4),
                "stdev_sec": round(stdev_lat, 4),
                "min_sec": round(min_lat, 4),
                "max_sec": round(max_lat, 4),
                "p90_sec": round(p90, 4),
                "p95_sec": round(p95, 4),
                "p99_sec": round(p99, 4),
                "avg_prompt_tokens": round(avg_prompt, 1),
                "avg_completion_tokens": round(avg_completion, 1),
            },
            "measurements": measurements,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.output}")


# ============================================================================
# 9. 메인
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
                        help="Qwen3 thinking 모드 비활성화")
    parser.add_argument("--eval-categories", type=str, nargs="+", default=None,
                        help="평가 대상 카테고리만 지정 (예: --eval-categories 이름 주소). 나머지 카테고리의 예측은 무시")
    parser.add_argument("--latency", action="store_true",
                        help="레이턴시 측정 모드 (3회 워밍업 + 10회 측정, batch_size=1, ~2K token input)")
    args = parser.parse_args()

    # ── 레이턴시 모드 ──
    if args.latency:
        run_latency_test(args)
        return

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
    if args.eval_categories:
        print(f"평가 카테고리: {', '.join(args.eval_categories)} (나머지 무시)")

    # ── OpenAI 클라이언트 ──
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    json_schema = build_json_schema()

    # ── 병렬 요청 ──
    print(f"\n추론 시작 ({len(test_cases)}개)...")
    start_time = time.time()
    all_results: list[dict] = []
    completed = 0

    eval_cats = args.eval_categories

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                call_api, client, args.model, tc, json_schema,
                args.temperature, args.max_tokens, args.no_think, eval_cats,
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
                predicted_err: dict[str, list[str] | None] = {cat: None for cat in PII_CATEGORIES}
                expected = normalize_expected(tc["expected_pii"])
                all_results.append({
                    "id": tc_id,
                    "category": tc["category"],
                    "difficulty": tc["difficulty"],
                    "intent": tc["intent"],
                    "expected": expected,
                    "predicted": predicted_err,
                    "metrics": compute_metrics(expected, predicted_err),
                    "raw_response": f"ERROR: {e}",
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
                match = "==" if set(exp or []) == set(pred or []) else "!="
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
                "raw_response": r.get("raw_response", ""),
            }
            for r in all_results
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
