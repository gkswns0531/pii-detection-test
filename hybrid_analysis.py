#!/usr/bin/env python3
"""Comprehensive hybrid evaluation analysis for PII detection.

Analyzes LLM, regex, and hybrid strategies across 12 PII categories.
"""

import json
from collections import defaultdict
from typing import Any

# =============================================================================
# Load Data
# =============================================================================

with open("combined_test_cases.json", encoding="utf-8") as f:
    test_cases: list[dict[str, Any]] = json.load(f)

with open("benchmark_results/results_qwen3_30b_v2_300.json", encoding="utf-8") as f:
    llm_data: dict[str, Any] = json.load(f)

with open("benchmark_results/regex_results.json", encoding="utf-8") as f:
    regex_data: dict[str, Any] = json.load(f)

CATEGORIES = [
    "이름", "주소", "주민등록번호", "여권번호", "운전면허번호",
    "이메일", "IP주소", "전화번호", "계좌번호", "카드번호",
    "생년월일", "기타_고유식별정보",
]

# Build lookup dicts by case ID
tc_by_id: dict[str, dict] = {tc["id"]: tc for tc in test_cases}
llm_by_id: dict[str, dict] = {r["id"]: r for r in llm_data["results"]}
regex_by_id: dict[str, dict] = {r["id"]: r for r in regex_data["results"]}


def get_expected(tc: dict) -> dict[str, set[str]]:
    """Get expected PII as {category: set of values}."""
    result: dict[str, set[str]] = defaultdict(set)
    for item in tc.get("expected_pii", []):
        result[item["type"]].add(item["value"])
    return result


def get_predicted_set(result_dict: dict, cat: str) -> set[str]:
    """Get predicted values for a category as a set."""
    pred = result_dict.get("predicted", {})
    vals = pred.get(cat) or []
    return set(vals)


def get_expected_set(result_dict: dict, cat: str) -> set[str]:
    """Get expected values from a result dict (for regex/llm results that include expected)."""
    exp = result_dict.get("expected", {})
    vals = exp.get(cat) or []
    return set(vals)


def compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, f1)."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# =============================================================================
# Analysis 1: Per-Category Breakdown
# =============================================================================

print("=" * 100)
print("  분석 1: 카테고리별 성능 비교 (LLM vs Regex vs Hybrid Union)")
print("=" * 100)

# We'll compute per-category TP/FP/FN for each method
cat_stats: dict[str, dict[str, dict[str, int]]] = {}
for cat in CATEGORIES:
    cat_stats[cat] = {
        "llm": {"tp": 0, "fp": 0, "fn": 0},
        "regex": {"tp": 0, "fp": 0, "fn": 0},
        "hybrid": {"tp": 0, "fp": 0, "fn": 0},
        "count": 0,  # type: ignore
    }

for tc in test_cases:
    tc_id = tc["id"]
    expected = get_expected(tc)
    llm_r = llm_by_id.get(tc_id, {})
    regex_r = regex_by_id.get(tc_id, {})

    for cat in CATEGORIES:
        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)
        hybrid_set = llm_set | regex_set

        cat_stats[cat]["count"] += len(e_set)  # type: ignore

        # LLM
        cat_stats[cat]["llm"]["tp"] += len(e_set & llm_set)
        cat_stats[cat]["llm"]["fp"] += len(llm_set - e_set)
        cat_stats[cat]["llm"]["fn"] += len(e_set - llm_set)

        # Regex
        cat_stats[cat]["regex"]["tp"] += len(e_set & regex_set)
        cat_stats[cat]["regex"]["fp"] += len(regex_set - e_set)
        cat_stats[cat]["regex"]["fn"] += len(e_set - regex_set)

        # Hybrid (union)
        cat_stats[cat]["hybrid"]["tp"] += len(e_set & hybrid_set)
        cat_stats[cat]["hybrid"]["fp"] += len(hybrid_set - e_set)
        cat_stats[cat]["hybrid"]["fn"] += len(e_set - hybrid_set)

# Print table
header = f"{'카테고리':<16} {'기대값':>5} | {'LLM TP':>6} {'FP':>4} {'FN':>4} {'P%':>6} {'R%':>6} {'F1%':>6} | {'Regex TP':>8} {'FP':>4} {'FN':>4} {'P%':>6} {'R%':>6} {'F1%':>6} | {'Hybrid TP':>9} {'FP':>4} {'FN':>4} {'P%':>6} {'R%':>6} {'F1%':>6}"
print(header)
print("-" * len(header))

totals = {"llm": {"tp": 0, "fp": 0, "fn": 0}, "regex": {"tp": 0, "fp": 0, "fn": 0}, "hybrid": {"tp": 0, "fp": 0, "fn": 0}}
total_expected = 0

for cat in CATEGORIES:
    s = cat_stats[cat]
    cnt = s["count"]
    total_expected += cnt

    for method in ["llm", "regex", "hybrid"]:
        for k in ["tp", "fp", "fn"]:
            totals[method][k] += s[method][k]

    lp, lr, lf = compute_f1(s["llm"]["tp"], s["llm"]["fp"], s["llm"]["fn"])
    rp, rr, rf = compute_f1(s["regex"]["tp"], s["regex"]["fp"], s["regex"]["fn"])
    hp, hr, hf = compute_f1(s["hybrid"]["tp"], s["hybrid"]["fp"], s["hybrid"]["fn"])

    print(
        f"{cat:<16} {cnt:>5} | "
        f"{s['llm']['tp']:>6} {s['llm']['fp']:>4} {s['llm']['fn']:>4} {lp*100:>6.1f} {lr*100:>6.1f} {lf*100:>6.1f} | "
        f"{s['regex']['tp']:>8} {s['regex']['fp']:>4} {s['regex']['fn']:>4} {rp*100:>6.1f} {rr*100:>6.1f} {rf*100:>6.1f} | "
        f"{s['hybrid']['tp']:>9} {s['hybrid']['fp']:>4} {s['hybrid']['fn']:>4} {hp*100:>6.1f} {hr*100:>6.1f} {hf*100:>6.1f}"
    )

print("-" * len(header))
lp, lr, lf = compute_f1(totals["llm"]["tp"], totals["llm"]["fp"], totals["llm"]["fn"])
rp, rr, rf = compute_f1(totals["regex"]["tp"], totals["regex"]["fp"], totals["regex"]["fn"])
hp, hr, hf = compute_f1(totals["hybrid"]["tp"], totals["hybrid"]["fp"], totals["hybrid"]["fn"])
print(
    f"{'합계':<16} {total_expected:>5} | "
    f"{totals['llm']['tp']:>6} {totals['llm']['fp']:>4} {totals['llm']['fn']:>4} {lp*100:>6.1f} {lr*100:>6.1f} {lf*100:>6.1f} | "
    f"{totals['regex']['tp']:>8} {totals['regex']['fp']:>4} {totals['regex']['fn']:>4} {rp*100:>6.1f} {rr*100:>6.1f} {rf*100:>6.1f} | "
    f"{totals['hybrid']['tp']:>9} {totals['hybrid']['fp']:>4} {totals['hybrid']['fn']:>4} {hp*100:>6.1f} {hr*100:>6.1f} {hf*100:>6.1f}"
)

# Also compute per-category regex precision for later strategy decisions
regex_precision: dict[str, float] = {}
for cat in CATEGORIES:
    s = cat_stats[cat]
    tp = s["regex"]["tp"]
    fp = s["regex"]["fp"]
    regex_precision[cat] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

print("\n\n카테고리별 Regex Precision 순위:")
for cat, prec in sorted(regex_precision.items(), key=lambda x: -x[1]):
    s = cat_stats[cat]
    print(f"  {cat:<20} Precision={prec*100:>6.1f}%  (TP={s['regex']['tp']}, FP={s['regex']['fp']})")


# =============================================================================
# Analysis 2: Cases Where Regex Catches LLM Misses (Rescue Potential)
# =============================================================================

print("\n\n" + "=" * 100)
print("  분석 2: Regex가 LLM의 미탐지(FN)를 복구하는 케이스")
print("=" * 100)

rescues: list[dict[str, str]] = []

for tc in test_cases:
    tc_id = tc["id"]
    expected = get_expected(tc)
    llm_r = llm_by_id.get(tc_id, {})
    regex_r = regex_by_id.get(tc_id, {})

    for cat in CATEGORIES:
        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)

        llm_fn = e_set - llm_set  # LLM missed
        regex_tp_on_fn = llm_fn & regex_set  # Regex caught what LLM missed

        for val in sorted(regex_tp_on_fn):
            rescues.append({"id": tc_id, "category": cat, "value": val})

print(f"\n총 복구 건수: {len(rescues)}건 (LLM이 놓쳤지만 Regex가 잡은 값)")
print(f"\n카테고리별 복구 건수:")
rescue_by_cat: dict[str, int] = defaultdict(int)
for r in rescues:
    rescue_by_cat[r["category"]] += 1
for cat, cnt in sorted(rescue_by_cat.items(), key=lambda x: -x[1]):
    print(f"  {cat:<20} {cnt}건")

print(f"\n상세 목록 (전체 {len(rescues)}건):")
print(f"{'케이스ID':<10} {'카테고리':<18} {'복구된 값'}")
print("-" * 80)
for r in rescues:
    print(f"{r['id']:<10} {r['category']:<18} {r['value']}")


# =============================================================================
# Analysis 3: Cases Where Regex Adds False Positives (LLM was correct)
# =============================================================================

print("\n\n" + "=" * 100)
print("  분석 3: Regex가 오탐(FP)하는 케이스 (LLM은 올바르게 미포함)")
print("=" * 100)

regex_only_fps: list[dict[str, str]] = []

for tc in test_cases:
    tc_id = tc["id"]
    expected = get_expected(tc)
    llm_r = llm_by_id.get(tc_id, {})
    regex_r = regex_by_id.get(tc_id, {})

    for cat in CATEGORIES:
        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)

        regex_fp = regex_set - e_set  # Regex false positives
        for val in sorted(regex_fp):
            # Check if LLM also predicted this (shared FP) or not (regex-only FP)
            if val not in llm_set:
                regex_only_fps.append({"id": tc_id, "category": cat, "value": val})

print(f"\n총 Regex-only 오탐 건수: {len(regex_only_fps)}건")
print(f"(Regex가 잘못 예측했지만 LLM은 올바르게 예측하지 않은 값)")

fp_by_cat: dict[str, list[str]] = defaultdict(list)
for item in regex_only_fps:
    fp_by_cat[item["category"]].append(item["value"])

print(f"\n카테고리별 Regex-only 오탐 건수:")
for cat, vals in sorted(fp_by_cat.items(), key=lambda x: -len(x[1])):
    print(f"\n  [{cat}] {len(vals)}건:")
    # Show sample values (up to 10)
    for v in vals[:10]:
        # Find the case ID
        case_id = [x["id"] for x in regex_only_fps if x["category"] == cat and x["value"] == v][0]
        print(f"    {case_id}: \"{v}\"")
    if len(vals) > 10:
        print(f"    ... 외 {len(vals) - 10}건")

# Summarize FP patterns
print("\n오탐 패턴 분석:")
for cat in sorted(fp_by_cat.keys()):
    vals = fp_by_cat[cat]
    print(f"\n  [{cat}] 주요 오탐 패턴:")
    if cat == "이름":
        print(f"    - 팀명/부서명을 이름으로 오인 (예: 마케팅팀, 기획팀 등)")
        print(f"    - 라벨 기반 추출의 한계: '담당자: <직함>' 같은 패턴")
    elif cat == "주소":
        print(f"    - 부분 주소 매칭 (짧은 도로명)")
        print(f"    - 시/군/구 이름만으로 오탐")
    elif cat == "전화번호":
        print(f"    - 서비스번호 오탐")
        print(f"    - 문서 내 참조번호를 전화번호로 오인")
    elif cat == "카드번호":
        print(f"    - 16자리 숫자 패턴 과매칭")
    elif cat == "기타_고유식별정보":
        print(f"    - 차량번호 패턴 과매칭")
        print(f"    - 비밀번호/숫자 코드 과매칭")
    else:
        print(f"    - 패턴 과매칭 ({len(vals)}건)")


# =============================================================================
# Analysis 4: Smart Hybrid Strategies
# =============================================================================

print("\n\n" + "=" * 100)
print("  분석 4: 스마트 하이브리드 전략")
print("=" * 100)


def compute_strategy_metrics(
    strategy_name: str,
    strategy_fn,  # Callable: (expected_set, llm_set, regex_set, cat) -> predicted_set
) -> tuple[int, int, int, float, float, float]:
    """Compute overall TP/FP/FN/P/R/F1 for a given strategy."""
    total_tp = total_fp = total_fn = 0
    per_cat: dict[str, dict[str, int]] = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in CATEGORIES}

    for tc in test_cases:
        tc_id = tc["id"]
        expected = get_expected(tc)
        llm_r = llm_by_id.get(tc_id, {})
        regex_r = regex_by_id.get(tc_id, {})

        for cat in CATEGORIES:
            e_set = expected.get(cat, set())
            llm_set = get_predicted_set(llm_r, cat)
            regex_set = get_predicted_set(regex_r, cat)

            pred_set = strategy_fn(e_set, llm_set, regex_set, cat)

            tp = len(e_set & pred_set)
            fp = len(pred_set - e_set)
            fn = len(e_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            per_cat[cat]["tp"] += tp
            per_cat[cat]["fp"] += fp
            per_cat[cat]["fn"] += fn

    prec, rec, f1 = compute_f1(total_tp, total_fp, total_fn)
    return total_tp, total_fp, total_fn, prec, rec, f1, per_cat  # type: ignore


# Strategy A: Regex only for high-precision categories (precision > 80%)
high_prec_cats = {cat for cat, p in regex_precision.items() if p > 0.80}

print(f"\n{'─' * 80}")
print(f"  전략 A: Regex 고정밀 카테고리만 사용 (Regex Precision > 80%)")
print(f"{'─' * 80}")
print(f"  해당 카테고리: {', '.join(sorted(high_prec_cats))}")


def strategy_a(e_set, llm_set, regex_set, cat):
    if cat in high_prec_cats:
        return llm_set | regex_set
    return llm_set


tp_a, fp_a, fn_a, p_a, r_a, f1_a, per_cat_a = compute_strategy_metrics("A", strategy_a)
print(f"\n  결과: TP={tp_a}, FP={fp_a}, FN={fn_a}")
print(f"  Precision={p_a*100:.1f}%, Recall={r_a*100:.1f}%, F1={f1_a*100:.1f}%")

print(f"\n  카테고리별 상세:")
print(f"  {'카테고리':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7}")
for cat in CATEGORIES:
    s = per_cat_a[cat]
    p, r, f = compute_f1(s["tp"], s["fp"], s["fn"])
    marker = " <-- regex 추가" if cat in high_prec_cats else ""
    print(f"  {cat:<16} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {p*100:>7.1f} {r*100:>7.1f} {f*100:>7.1f}{marker}")


# Strategy B: Regex as LLM-miss recovery (only add regex when LLM returned null/empty for that category)
print(f"\n{'─' * 80}")
print(f"  전략 B: Regex를 LLM 미응답 복구용으로만 사용")
print(f"  (LLM이 해당 카테고리에 null/empty를 반환한 경우에만 regex 결과 추가)")
print(f"{'─' * 80}")


def strategy_b(e_set, llm_set, regex_set, cat):
    if len(llm_set) == 0:
        return regex_set  # LLM returned nothing, fall back to regex
    return llm_set  # LLM had predictions, trust them


tp_b, fp_b, fn_b, p_b, r_b, f1_b, per_cat_b = compute_strategy_metrics("B", strategy_b)
print(f"\n  결과: TP={tp_b}, FP={fp_b}, FN={fn_b}")
print(f"  Precision={p_b*100:.1f}%, Recall={r_b*100:.1f}%, F1={f1_b*100:.1f}%")

print(f"\n  카테고리별 상세:")
print(f"  {'카테고리':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7}")
for cat in CATEGORIES:
    s = per_cat_b[cat]
    p, r, f = compute_f1(s["tp"], s["fp"], s["fn"])
    print(f"  {cat:<16} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {p*100:>7.1f} {r*100:>7.1f} {f*100:>7.1f}")


# Strategy C: Confidence-weighted (always add regex for strong cats, never for weak)
strong_regex_cats = {"주민등록번호", "여권번호", "운전면허번호", "이메일", "계좌번호", "생년월일", "카드번호"}
weak_regex_cats = {"이름", "주소", "기타_고유식별정보"}
# Medium categories: IP주소, 전화번호 - use conditional logic

print(f"\n{'─' * 80}")
print(f"  전략 C: 신뢰도 기반 선택적 적용")
print(f"{'─' * 80}")
print(f"  Regex 항상 적용 (강한 카테고리): {', '.join(sorted(strong_regex_cats))}")
print(f"  Regex 미적용 (약한 카테고리): {', '.join(sorted(weak_regex_cats))}")
print(f"  중간 카테고리 (IP주소, 전화번호): LLM이 비어있을 때만 regex 적용")


def strategy_c(e_set, llm_set, regex_set, cat):
    if cat in strong_regex_cats:
        return llm_set | regex_set  # Always add regex
    elif cat in weak_regex_cats:
        return llm_set  # Never use regex
    else:
        # Medium: only use regex if LLM is empty
        if len(llm_set) == 0:
            return regex_set
        return llm_set


tp_c, fp_c, fn_c, p_c, r_c, f1_c, per_cat_c = compute_strategy_metrics("C", strategy_c)
print(f"\n  결과: TP={tp_c}, FP={fp_c}, FN={fn_c}")
print(f"  Precision={p_c*100:.1f}%, Recall={r_c*100:.1f}%, F1={f1_c*100:.1f}%")

print(f"\n  카테고리별 상세:")
print(f"  {'카테고리':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7} {'방식'}")
for cat in CATEGORIES:
    s = per_cat_c[cat]
    p, r, f = compute_f1(s["tp"], s["fp"], s["fn"])
    if cat in strong_regex_cats:
        mode = "union (항상)"
    elif cat in weak_regex_cats:
        mode = "LLM only"
    else:
        mode = "조건부 (LLM 비어있을때)"
    print(f"  {cat:<16} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {p*100:>7.1f} {r*100:>7.1f} {f*100:>7.1f} {mode}")


# Strategy D: Intersection boost - only trust regex predictions that partially match LLM output
print(f"\n{'─' * 80}")
print(f"  전략 D: 교차 검증 (Regex 결과 중 LLM과 부분 매칭되는 것만 추가)")
print(f"{'─' * 80}")
print(f"  방법: Regex 예측 중 LLM 예측값과 부분 문자열 일치가 있는 경우만 추가")
print(f"        + LLM 전체 결과는 항상 유지")


def partial_match(val_a: str, val_b: str) -> bool:
    """Check if two values partially overlap (substring match)."""
    a = val_a.strip().replace(" ", "")
    b = val_b.strip().replace(" ", "")
    return a in b or b in a


def strategy_d(e_set, llm_set, regex_set, cat):
    result = set(llm_set)
    for rv in regex_set:
        for lv in llm_set:
            if partial_match(rv, lv):
                result.add(rv)
                break
    return result


tp_d, fp_d, fn_d, p_d, r_d, f1_d, per_cat_d = compute_strategy_metrics("D", strategy_d)
print(f"\n  결과: TP={tp_d}, FP={fp_d}, FN={fn_d}")
print(f"  Precision={p_d*100:.1f}%, Recall={r_d*100:.1f}%, F1={f1_d*100:.1f}%")

print(f"\n  카테고리별 상세:")
print(f"  {'카테고리':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7}")
for cat in CATEGORIES:
    s = per_cat_d[cat]
    p, r, f = compute_f1(s["tp"], s["fp"], s["fn"])
    print(f"  {cat:<16} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {p*100:>7.1f} {r*100:>7.1f} {f*100:>7.1f}")


# =============================================================================
# Strategy Comparison Summary
# =============================================================================

print("\n\n" + "=" * 100)
print("  전략 비교 요약 (Summary)")
print("=" * 100)

strategies = [
    ("LLM Only (베이스라인)", totals["llm"]["tp"], totals["llm"]["fp"], totals["llm"]["fn"]),
    ("Regex Only", totals["regex"]["tp"], totals["regex"]["fp"], totals["regex"]["fn"]),
    ("단순 Union (Hybrid)", totals["hybrid"]["tp"], totals["hybrid"]["fp"], totals["hybrid"]["fn"]),
    ("전략 A: 고정밀 카테고리만", tp_a, fp_a, fn_a),
    ("전략 B: LLM 미응답 복구", tp_b, fp_b, fn_b),
    ("전략 C: 신뢰도 기반 선택", tp_c, fp_c, fn_c),
    ("전략 D: 교차 검증", tp_d, fp_d, fn_d),
]

print(f"\n{'전략':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision%':>12} {'Recall%':>10} {'F1%':>8} {'P-R Gap':>8}")
print("-" * 108)
for name, tp, fp, fn in strategies:
    p, r, f = compute_f1(tp, fp, fn)
    gap = abs(p - r)
    print(f"{name:<30} {tp:>5} {fp:>5} {fn:>5} {p*100:>12.1f} {r*100:>10.1f} {f*100:>8.1f} {gap*100:>8.1f}")


# =============================================================================
# Strategy B+C Hybrid: Best of B and C
# =============================================================================

print(f"\n\n{'─' * 80}")
print(f"  보너스 전략 E: 전략 C + LLM 미응답 시 약한 카테고리도 regex 적용")
print(f"{'─' * 80}")
print(f"  강한 카테고리: 항상 union")
print(f"  약한 카테고리: LLM이 비어있을 때만 regex")
print(f"  중간 카테고리: LLM이 비어있을 때만 regex")


def strategy_e(e_set, llm_set, regex_set, cat):
    if cat in strong_regex_cats:
        return llm_set | regex_set
    else:
        if len(llm_set) == 0:
            return regex_set
        return llm_set


tp_e, fp_e, fn_e, p_e, r_e, f1_e, per_cat_e = compute_strategy_metrics("E", strategy_e)
print(f"\n  결과: TP={tp_e}, FP={fp_e}, FN={fn_e}")
print(f"  Precision={p_e*100:.1f}%, Recall={r_e*100:.1f}%, F1={f1_e*100:.1f}%")

# Add to comparison
strategies.append(("전략 E: C+B 결합", tp_e, fp_e, fn_e))


# =============================================================================
# Final Recommendation
# =============================================================================

print("\n\n" + "=" * 100)
print("  최종 추천")
print("=" * 100)

# Find best F1 strategy
best = max(strategies, key=lambda x: compute_f1(x[1], x[2], x[3])[2])
best_name = best[0]
bp, br, bf = compute_f1(best[1], best[2], best[3])

print(f"\n  최고 F1 전략: {best_name}")
print(f"  F1={bf*100:.1f}%, Precision={bp*100:.1f}%, Recall={br*100:.1f}%")

# Compare with LLM baseline
llm_p, llm_r, llm_f = compute_f1(totals["llm"]["tp"], totals["llm"]["fp"], totals["llm"]["fn"])
print(f"\n  LLM 베이스라인 대비:")
print(f"  F1:        {llm_f*100:.1f}% → {bf*100:.1f}% ({(bf-llm_f)*100:+.1f}%p)")
print(f"  Precision: {llm_p*100:.1f}% → {bp*100:.1f}% ({(bp-llm_p)*100:+.1f}%p)")
print(f"  Recall:    {llm_r*100:.1f}% → {br*100:.1f}% ({(br-llm_r)*100:+.1f}%p)")

# Which strategies improve over LLM?
print(f"\n  LLM 베이스라인(F1={llm_f*100:.1f}%) 대비 향상되는 전략:")
for name, tp, fp, fn in strategies:
    p, r, f = compute_f1(tp, fp, fn)
    if f > llm_f:
        print(f"    {name}: F1={f*100:.1f}% (+{(f-llm_f)*100:.1f}%p)")

print(f"\n  LLM 베이스라인 대비 하락하는 전략:")
for name, tp, fp, fn in strategies:
    p, r, f = compute_f1(tp, fp, fn)
    if f < llm_f and name not in ("Regex Only",):
        print(f"    {name}: F1={f*100:.1f}% ({(f-llm_f)*100:.1f}%p)")

print("\n\n분석 완료.")
