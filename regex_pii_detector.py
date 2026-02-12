#!/usr/bin/env python3
"""Regex-based PII detector for Korean documents.

Detects PII across 12 categories using pattern matching.
Used as a standalone baseline or as a hybrid complement to LLM-based detection.
"""

import json
import re
import sys
from pathlib import Path


# ============================================================================
# Regex Patterns per Category
# ============================================================================

# 1. 이름 - regex로는 제한적이므로 레이블 기반 추출
NAME_LABELS = (
    r"(?:성명|이름|담당자|신청자|승인자|수신|참조|발신|위임인|수임인|추천인|피추천인|"
    r"환자\s*성명|참석자|퇴직자|수상자|청구인|피보험자|연구책임자|대표자|작성자|"
    r"계약자|수익자|위탁자|수탁자|보증인|연대보증인|진술인|입회인|면접관|보호자|"
    r"담임교사|멘토|피면접자|감사담당자|회원명|입찰담당|대리업무자|피청구인|"
    r"원고|피고|채권자|채무자|임차인|임대인|수하인|송하인|위탁자|수탁자)"
)
NAME_PATTERN = re.compile(
    rf"{NAME_LABELS}\s*[:：]\s*([가-힣]{{2,5}})",
    re.UNICODE,
)

# 2. 주소 - 도로명주소 / 지번주소
ROAD_ADDR = re.compile(
    r"(?:[가-힣]{1,10}(?:특별시|광역시|특별자치시|도|특별자치도)\s+)?"
    r"[가-힣]{1,10}(?:시|군|구)\s+"
    r"(?:[가-힣]{1,10}(?:구|군)\s+)?"
    r"[가-힣\d]{1,20}(?:로|길)\s*\d+"
    r"(?:\s*[-,]\s*\d+)?"
    r"(?:\s*\([가-힣\d\s,]+\))?"
    r"(?:\s*\d{1,4}동?\s*\d{1,4}호)?",
    re.UNICODE,
)
LOT_ADDR = re.compile(
    r"(?:[가-힣]{1,10}(?:특별시|광역시|특별자치시|도|특별자치도)\s+)?"
    r"[가-힣]{1,10}(?:시|군|구)\s+"
    r"(?:[가-힣]{1,10}(?:구|군)\s+)?"
    r"[가-힣]{1,10}(?:읍|면|동|리|가)\s+"
    r"(?:\d+(?:[-의]\d+)?(?:번지)?)",
    re.UNICODE,
)

# 3. 주민등록번호
RRN_PATTERN = re.compile(
    r"\b(\d{6})\s*[-–—]\s*([1-8*]\d{6}|\d[*]{5,6}|\*{6,7})\b"
)
RRN_CONTINUOUS = re.compile(
    r"\b(\d{6}[1-8]\d{6})\b"
)

# 4. 여권번호
PASSPORT_KR = re.compile(r"\b[MmSs]\d{8}\b")
PASSPORT_FOREIGN = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")

# 5. 운전면허번호
DRIVER_LICENSE = re.compile(
    r"\b\d{2}\s*[-–]\s*\d{2}\s*[-–]\s*\d{6}\s*[-–]\s*\d{2}\b"
)

# 6. 이메일
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+\s*[@＠]\s*[a-zA-Z0-9.\-]+\.\s*[a-zA-Z]{2,}",
    re.UNICODE,
)
EMAIL_OBFUSCATED = re.compile(
    r"[a-zA-Z0-9._%+\-]+\s*(?:\[at\]|\(at\)|골뱅이|\{at\})\s*"
    r"[a-zA-Z0-9.\-]+\s*(?:\[dot\]|\(dot\)|\.)\s*[a-zA-Z]{2,}",
    re.IGNORECASE,
)
# Masked emails like h****@gmail.com
EMAIL_MASKED = re.compile(
    r"[a-zA-Z][*]{2,}@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# 7. IP 주소
IPV4_PATTERN = re.compile(
    r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d{1,5}))?\b"
)
IPV6_PATTERN = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
    r"|(?:[0-9a-fA-F]{1,4}:){1,6}::[0-9a-fA-F]{0,4}\b"
)

# IPs to exclude
EXCLUDE_IPS = {
    "127.0.0.1", "0.0.0.0", "255.255.255.255",
    "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1",
    "208.67.222.222", "208.67.220.220", "9.9.9.9",
}

# 8. 전화번호
PHONE_MOBILE = re.compile(
    r"\b01[016789]\s*[-.)]\s*\d{3,4}\s*[-.)]\s*\d{4}\b"
)
PHONE_LANDLINE = re.compile(
    r"\b0(?:2|3[1-3]|4[1-4]|5[1-5]|6[1-4])\s*[-.)]\s*\d{3,4}\s*[-.)]\s*\d{4}\b"
)
PHONE_INTL = re.compile(
    r"\+\d{1,3}\s*[-.)]\s*\d{1,4}\s*[-.)]\s*\d{3,4}\s*[-.)]\s*\d{3,4}"
)
PHONE_CONTINUOUS = re.compile(
    r"\b01[016789]\d{7,8}\b"
)
# Masked phone
PHONE_MASKED = re.compile(
    r"\b01[016789]\s*[-.)]\s*\d{0,2}[*]{2,4}\s*[-.)]\s*\d{0,2}[*]{2,4}\b"
)
# Service numbers to exclude
SERVICE_NUMBER = re.compile(
    r"\b(?:15\d{2}|16\d{2}|18\d{2})\s*[-.]?\s*\d{4}\b"
)
TOLL_FREE = re.compile(r"\b080\s*[-.]?\s*\d{3,4}\s*[-.]?\s*\d{4}\b")

# 9. 계좌번호 - label-based
ACCOUNT_LABELS = re.compile(
    r"(?:계좌|가상계좌|입금계좌|출금계좌|환불계좌)\s*(?:번호)?\s*[:：]?\s*"
    r"(\d{2,6}[-\s]?\d{2,8}[-\s]?\d{2,8}(?:[-\s]?\d{1,4})?)",
    re.UNICODE,
)
# Standalone account patterns (bank name + number)
ACCOUNT_WITH_BANK = re.compile(
    r"(?:국민|신한|우리|하나|농협|기업|SC|씨티|대구|부산|경남|광주|전북|제주|수협|"
    r"새마을|신협|우체국|카카오|토스|케이)\s*(?:은행)?\s*"
    r"(\d{2,6}[-\s]?\d{2,8}[-\s]?\d{2,8}(?:[-\s]?\d{1,4})?)",
    re.UNICODE,
)

# 10. 카드번호
CARD_PATTERN = re.compile(
    r"\b\d{4}\s*[-\s]\s*\d{4}\s*[-\s]\s*\d{4}\s*[-\s]\s*\d{4}\b"
)
CARD_MASKED = re.compile(
    r"\b\d{4}\s*[-\s]\s*\d{2}[*]{2}\s*[-\s]\s*[*]{4}\s*[-\s]\s*\d{4}\b"
)
# BTC address
BTC_ADDR = re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")
ETH_ADDR = re.compile(r"\b0x[0-9a-fA-F]{40}\b")

# 11. 생년월일
DOB_PATTERN = re.compile(
    r"(?:생년월일|출생일?|생일|DOB|Date\s*of\s*Birth|born)\s*[:：]?\s*"
    r"(\d{4}[-./년\s]\s*\d{1,2}[-./월\s]\s*\d{1,2}일?|\d{6})",
    re.IGNORECASE | re.UNICODE,
)

# 12. 기타_고유식별정보
# 학번
STUDENT_ID = re.compile(
    r"(?:학번)\s*[:：]?\s*(\d{4}[-]?\d{4,6})"
)
# 차량번호
VEHICLE_PLATE = re.compile(
    r"\b\d{2,3}\s*[가-힣]\s*\d{4}\b"
)
# 군번
MILITARY_ID = re.compile(
    r"(?:군번)\s*[:：]?\s*(\d{2}[-]?\d{8,})"
)
# 사번
EMPLOYEE_ID = re.compile(
    r"(?:사번)\s*[:：]?\s*(\d{4}[-]?\d{4,6})"
)
# 도어락 비밀번호
DOORLOCK = re.compile(
    r"(?:비밀번호|현관|도어락)\s*[:：]?\s*([#*]?\d{4,8}[#*]?)"
)


def detect_pii_regex(text: str) -> dict[str, list[str] | None]:
    """Detect PII in text using regex patterns. Returns dict matching eval schema."""
    result: dict[str, list[str] | None] = {
        "이름": None, "주소": None, "주민등록번호": None,
        "여권번호": None, "운전면허번호": None, "이메일": None,
        "IP주소": None, "전화번호": None, "계좌번호": None,
        "카드번호": None, "생년월일": None, "기타_고유식별정보": None,
    }

    def add(cat: str, val: str) -> None:
        val = val.strip()
        if not val:
            return
        if result[cat] is None:
            result[cat] = []
        if val not in result[cat]:
            result[cat].append(val)

    # 1. 이름 (label-based only for regex)
    for m in NAME_PATTERN.finditer(text):
        add("이름", m.group(1))

    # 2. 주소
    for m in ROAD_ADDR.finditer(text):
        add("주소", m.group(0))
    for m in LOT_ADDR.finditer(text):
        add("주소", m.group(0))

    # 3. 주민등록번호
    for m in RRN_PATTERN.finditer(text):
        full = f"{m.group(1)}-{m.group(2)}"
        add("주민등록번호", full)
    for m in RRN_CONTINUOUS.finditer(text):
        add("주민등록번호", m.group(1))

    # 4. 여권번호
    for m in PASSPORT_KR.finditer(text):
        add("여권번호", m.group(0))

    # 5. 운전면허번호
    for m in DRIVER_LICENSE.finditer(text):
        add("운전면허번호", m.group(0))

    # 6. 이메일
    for m in EMAIL_PATTERN.finditer(text):
        email = m.group(0).strip()
        # Exclude test/role-based emails
        lower = email.lower()
        if any(x in lower for x in ["example.com", "test.com", "noreply", "no-reply"]):
            continue
        if re.match(r"^(?:info|support|contact|admin|webmaster|help)@", lower):
            continue
        add("이메일", email)
    for m in EMAIL_OBFUSCATED.finditer(text):
        add("이메일", m.group(0).strip())
    for m in EMAIL_MASKED.finditer(text):
        add("이메일", m.group(0).strip())

    # 7. IP주소
    for m in IPV4_PATTERN.finditer(text):
        ip = m.group(1)
        port = m.group(2)
        if ip in EXCLUDE_IPS:
            continue
        # Skip private ranges in design docs (simple heuristic)
        octets = ip.split(".")
        if octets[0] == "10" or (octets[0] == "172" and 16 <= int(octets[1]) <= 31) or (octets[0] == "192" and octets[1] == "168"):
            # Check if it looks like CIDR context
            idx = m.start()
            context = text[max(0, idx - 80):idx + len(m.group(0)) + 20]
            if "/" in context or "대역" in context or "서브넷" in context or "설계" in context or "할당" in context:
                continue
        full = f"{ip}:{port}" if port else ip
        add("IP주소", full)
    for m in IPV6_PATTERN.finditer(text):
        v6 = m.group(0)
        if v6 == "::1":
            continue
        add("IP주소", v6)

    # 8. 전화번호
    all_phones: list[str] = []
    for m in PHONE_MOBILE.finditer(text):
        all_phones.append(m.group(0))
    for m in PHONE_LANDLINE.finditer(text):
        all_phones.append(m.group(0))
    for m in PHONE_INTL.finditer(text):
        all_phones.append(m.group(0))
    for m in PHONE_CONTINUOUS.finditer(text):
        all_phones.append(m.group(0))
    for m in PHONE_MASKED.finditer(text):
        all_phones.append(m.group(0))

    # Filter out service numbers
    service_spans = set()
    for m in SERVICE_NUMBER.finditer(text):
        service_spans.add((m.start(), m.end()))
    for m in TOLL_FREE.finditer(text):
        service_spans.add((m.start(), m.end()))

    for phone in all_phones:
        # Check if this phone overlaps with a service number
        idx = text.find(phone)
        is_service = False
        for s, e in service_spans:
            if s <= idx < e or s < idx + len(phone) <= e:
                is_service = True
                break
        if not is_service:
            add("전화번호", phone)

    # 9. 계좌번호
    for m in ACCOUNT_LABELS.finditer(text):
        add("계좌번호", m.group(1))
    for m in ACCOUNT_WITH_BANK.finditer(text):
        add("계좌번호", m.group(1))

    # 10. 카드번호
    for m in CARD_PATTERN.finditer(text):
        add("카드번호", m.group(0))
    for m in CARD_MASKED.finditer(text):
        add("카드번호", m.group(0))
    for m in BTC_ADDR.finditer(text):
        add("카드번호", m.group(0))
    for m in ETH_ADDR.finditer(text):
        add("카드번호", m.group(0))

    # 11. 생년월일
    for m in DOB_PATTERN.finditer(text):
        add("생년월일", m.group(1))

    # 12. 기타_고유식별정보
    for m in STUDENT_ID.finditer(text):
        add("기타_고유식별정보", m.group(1))
    for m in VEHICLE_PLATE.finditer(text):
        add("기타_고유식별정보", m.group(0))
    for m in MILITARY_ID.finditer(text):
        add("기타_고유식별정보", m.group(1))
    for m in EMPLOYEE_ID.finditer(text):
        add("기타_고유식별정보", m.group(1))
    for m in DOORLOCK.finditer(text):
        add("기타_고유식별정보", m.group(1))

    return result


def merge_predictions(llm_pred: dict, regex_pred: dict) -> dict:
    """Merge LLM and regex predictions (union). Hybrid approach."""
    merged = {}
    for cat in llm_pred:
        llm_vals = set(llm_pred.get(cat) or [])
        regex_vals = set(regex_pred.get(cat) or [])
        combined = llm_vals | regex_vals
        merged[cat] = sorted(combined) if combined else None
    return merged


def run_regex_benchmark(test_cases_path: str, output_path: str | None = None) -> dict:
    """Run regex-only benchmark on test cases and return stats."""
    with open(test_cases_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    total_tp = total_fp = total_fn = 0

    for tc in test_cases:
        pred = detect_pii_regex(tc["document_text"])

        # Convert expected to same format
        expected: dict[str, list[str] | None] = {}
        for item in tc.get("expected_pii", []):
            cat = item["type"]
            if cat not in expected:
                expected[cat] = []
            expected[cat].append(item["value"])

        # Compute per-case metrics
        tp = fp = fn = 0
        cats = ["이름", "주소", "주민등록번호", "여권번호", "운전면허번호",
                "이메일", "IP주소", "전화번호", "계좌번호", "카드번호",
                "생년월일", "기타_고유식별정보"]
        for cat in cats:
            e_set = set(expected.get(cat, []))
            p_set = set(pred.get(cat) or [])
            tp += len(e_set & p_set)
            fp += len(p_set - e_set)
            fn += len(e_set - p_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results.append({
            "id": tc["id"],
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "expected": {k: v for k, v in expected.items() if v},
            "predicted": {k: v for k, v in pred.items() if v is not None},
        })

    # Overall stats
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    perfect = sum(1 for r in results if r["f1"] == 1.0)

    stats = {
        "total": len(results),
        "perfect": perfect,
        "accuracy": round(perfect / len(results) * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    output = {"stats": stats, "results": results}

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")

    return output


if __name__ == "__main__":
    tc_path = sys.argv[1] if len(sys.argv) > 1 else "combined_test_cases.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/regex_results.json"

    print(f"Running regex benchmark on {tc_path}...")
    output = run_regex_benchmark(tc_path, out_path)
    s = output["stats"]
    print(f"\n=== Regex-Only Results ===")
    print(f"Total: {s['total']} | Perfect: {s['perfect']} ({s['accuracy']}%)")
    print(f"Precision: {s['precision']}% | Recall: {s['recall']}% | F1: {s['f1']}%")
    print(f"TP: {s['tp']} | FP: {s['fp']} | FN: {s['fn']}")
