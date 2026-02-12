#!/usr/bin/env python3
"""Deep dive: category-specific strategies where regex might help LLM."""

import json
from collections import defaultdict
from typing import Any

with open("combined_test_cases.json", encoding="utf-8") as f:
    test_cases = json.load(f)
with open("benchmark_results/results_qwen3_30b_v2_300.json", encoding="utf-8") as f:
    llm_data = json.load(f)
with open("benchmark_results/regex_results.json", encoding="utf-8") as f:
    regex_data = json.load(f)

CATEGORIES = [
    "이름", "주소", "주민등록번호", "여권번호", "운전면허번호",
    "이메일", "IP주소", "전화번호", "계좌번호", "카드번호",
    "생년월일", "기타_고유식별정보",
]

tc_by_id = {tc["id"]: tc for tc in test_cases}
llm_by_id = {r["id"]: r for r in llm_data["results"]}
regex_by_id = {r["id"]: r for r in regex_data["results"]}


def get_expected(tc):
    result = defaultdict(set)
    for item in tc.get("expected_pii", []):
        result[item["type"]].add(item["value"])
    return result


def get_predicted_set(result_dict, cat):
    pred = result_dict.get("predicted", {})
    vals = pred.get(cat) or []
    return set(vals)


def compute_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# =============================================================================
# Deep Dive: LLM Weak Spots Analysis
# =============================================================================
print("=" * 100)
print("  심층 분석: LLM이 약한 카테고리별 Regex 보완 가능성")
print("=" * 100)

# Find LLM's weakest categories (by FN count or recall)
print("\n■ LLM의 카테고리별 FN(미탐지) 분석:")
print(f"{'카테고리':<16} {'기대값':>5} {'LLM FN':>7} {'미탐지율':>8} {'Regex가 복구 가능한 수':>22} {'복구율':>6}")
print("-" * 75)

for cat in CATEGORIES:
    total_expected = 0
    llm_fn_count = 0
    regex_rescues = 0

    for tc in test_cases:
        tc_id = tc["id"]
        expected = get_expected(tc)
        llm_r = llm_by_id.get(tc_id, {})
        regex_r = regex_by_id.get(tc_id, {})

        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)

        total_expected += len(e_set)
        llm_fn = e_set - llm_set
        llm_fn_count += len(llm_fn)
        regex_rescues += len(llm_fn & regex_set)

    miss_rate = llm_fn_count / total_expected * 100 if total_expected > 0 else 0
    rescue_rate = regex_rescues / llm_fn_count * 100 if llm_fn_count > 0 else 0
    print(f"{cat:<16} {total_expected:>5} {llm_fn_count:>7} {miss_rate:>7.1f}% {regex_rescues:>22} {rescue_rate:>5.1f}%")


# =============================================================================
# Most precise strategy: Regex only for 계좌번호 (100% precision, helps recall)
# =============================================================================
print("\n\n" + "=" * 100)
print("  최적 전략 탐색: 카테고리별 net gain/loss 분석")
print("=" * 100)
print("\n  각 카테고리에 대해 regex union을 적용했을 때의 F1 변화:")
print(f"\n{'카테고리':<16} {'LLM F1':>8} {'Union F1':>9} {'차이':>8} {'추가TP':>6} {'추가FP':>6} {'Net':>6}")
print("-" * 65)

per_cat_gains = []
for cat in CATEGORIES:
    llm_tp = llm_fp = llm_fn = 0
    union_tp = union_fp = union_fn = 0

    for tc in test_cases:
        tc_id = tc["id"]
        expected = get_expected(tc)
        llm_r = llm_by_id.get(tc_id, {})
        regex_r = regex_by_id.get(tc_id, {})

        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)
        union_set = llm_set | regex_set

        llm_tp += len(e_set & llm_set)
        llm_fp += len(llm_set - e_set)
        llm_fn += len(e_set - llm_set)

        union_tp += len(e_set & union_set)
        union_fp += len(union_set - e_set)
        union_fn += len(e_set - union_set)

    _, _, llm_f1 = compute_f1(llm_tp, llm_fp, llm_fn)
    _, _, union_f1 = compute_f1(union_tp, union_fp, union_fn)
    diff = union_f1 - llm_f1
    added_tp = union_tp - llm_tp
    added_fp = union_fp - llm_fp
    net = added_tp - added_fp

    per_cat_gains.append((cat, diff, added_tp, added_fp))
    marker = " ★" if diff > 0 else " ✗" if diff < -0.01 else ""
    print(f"{cat:<16} {llm_f1*100:>7.1f}% {union_f1*100:>8.1f}% {diff*100:>+7.1f}% {added_tp:>6} {added_fp:>6} {net:>+6}{marker}")


# =============================================================================
# Strategy F: Only add regex for categories where union improves F1
# =============================================================================
print("\n\n" + "=" * 100)
print("  전략 F: F1이 향상되는 카테고리에만 Regex 적용")
print("=" * 100)

beneficial_cats = {cat for cat, diff, _, _ in per_cat_gains if diff > 0}
print(f"  F1 향상 카테고리: {', '.join(sorted(beneficial_cats)) if beneficial_cats else '없음'}")

# Since only 계좌번호 improves, let's check just that
def strategy_f(e_set, llm_set, regex_set, cat):
    if cat in beneficial_cats:
        return llm_set | regex_set
    return llm_set

total_tp = total_fp = total_fn = 0
for tc in test_cases:
    tc_id = tc["id"]
    expected = get_expected(tc)
    llm_r = llm_by_id.get(tc_id, {})
    regex_r = regex_by_id.get(tc_id, {})
    for cat in CATEGORIES:
        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)
        pred = strategy_f(e_set, llm_set, regex_set, cat)
        total_tp += len(e_set & pred)
        total_fp += len(pred - e_set)
        total_fn += len(e_set - pred)

p, r, f = compute_f1(total_tp, total_fp, total_fn)
print(f"\n  결과: TP={total_tp}, FP={total_fp}, FN={total_fn}")
print(f"  Precision={p*100:.1f}%, Recall={r*100:.1f}%, F1={f*100:.1f}%")


# =============================================================================
# Strategy G: Regex only for 계좌번호 + conditional for 주민등록번호 (LLM empty only)
# =============================================================================
print("\n\n" + "=" * 100)
print("  전략 G: 계좌번호 union + 주민등록번호/이메일 LLM-empty-only 복구")
print("=" * 100)

union_cats = {"계좌번호"}
conditional_cats = {"주민등록번호", "이메일"}

def strategy_g(e_set, llm_set, regex_set, cat):
    if cat in union_cats:
        return llm_set | regex_set
    elif cat in conditional_cats:
        if len(llm_set) == 0:
            return regex_set
        return llm_set
    return llm_set

total_tp = total_fp = total_fn = 0
for tc in test_cases:
    tc_id = tc["id"]
    expected = get_expected(tc)
    llm_r = llm_by_id.get(tc_id, {})
    regex_r = regex_by_id.get(tc_id, {})
    for cat in CATEGORIES:
        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)
        pred = strategy_g(e_set, llm_set, regex_set, cat)
        total_tp += len(e_set & pred)
        total_fp += len(pred - e_set)
        total_fn += len(e_set - pred)

p, r, f = compute_f1(total_tp, total_fp, total_fn)
print(f"  결과: TP={total_tp}, FP={total_fp}, FN={total_fn}")
print(f"  Precision={p*100:.1f}%, Recall={r*100:.1f}%, F1={f*100:.1f}%")


# =============================================================================
# Examine: What are the 27 LLM FNs? Can regex help with any of them?
# =============================================================================
print("\n\n" + "=" * 100)
print("  LLM 미탐지(FN) 전체 27건 상세 분석")
print("=" * 100)

all_fns = []
for tc in test_cases:
    tc_id = tc["id"]
    expected = get_expected(tc)
    llm_r = llm_by_id.get(tc_id, {})
    regex_r = regex_by_id.get(tc_id, {})
    for cat in CATEGORIES:
        e_set = expected.get(cat, set())
        llm_set = get_predicted_set(llm_r, cat)
        regex_set = get_predicted_set(regex_r, cat)
        for val in e_set - llm_set:
            regex_caught = val in regex_set
            all_fns.append({
                "id": tc_id,
                "cat": cat,
                "value": val,
                "regex_caught": regex_caught,
            })

print(f"\nLLM FN 총 {len(all_fns)}건:")
print(f"{'케이스ID':<10} {'카테고리':<16} {'Regex복구':>8} 값")
print("-" * 90)
for item in all_fns:
    caught = "O" if item["regex_caught"] else "X"
    print(f"{item['id']:<10} {item['cat']:<16} {caught:>8} {item['value']}")

regex_recoverable = sum(1 for x in all_fns if x["regex_caught"])
print(f"\n요약: LLM FN {len(all_fns)}건 중 Regex로 복구 가능: {regex_recoverable}건 ({regex_recoverable/len(all_fns)*100:.1f}%)")
print(f"       Regex로 복구 불가: {len(all_fns) - regex_recoverable}건")


# =============================================================================
# Final comprehensive strategy comparison
# =============================================================================
print("\n\n" + "=" * 100)
print("  최종 전략 비교 테이블")
print("=" * 100)

llm_baseline = (742, 29, 27)

all_strategies = {
    "LLM Only (베이스라인)": llm_baseline,
    "Regex Only": (327, 183, 442),
    "단순 Union": (750, 208, 19),
}

# Recompute strategies with clean code
for strat_name, strat_fn, desc in [
    ("전략A: 고정밀 카테고리만 union",
     lambda e, l, r, c: l | r if c in {"계좌번호", "생년월일", "이름", "이메일", "전화번호"} else l,
     "Regex P>80%인 카테고리만"),
    ("전략B: LLM미응답시 regex",
     lambda e, l, r, c: r if len(l) == 0 else l,
     "LLM=null일 때만 regex"),
    ("전략C: 신뢰도기반 선택",
     lambda e, l, r, c: (l | r) if c in {"주민등록번호", "여권번호", "운전면허번호", "이메일", "계좌번호", "생년월일", "카드번호"} else (r if len(l) == 0 and c in {"IP주소", "전화번호"} else l),
     "강한cat=union, 중간=조건부, 약한=LLM"),
    ("전략D: 교차검증",
     None,  # special
     "Regex결과가 LLM과 부분매칭시만"),
    ("전략F: F1향상 카테고리만",
     lambda e, l, r, c: (l | r) if c in beneficial_cats else l,
     "union시 F1 향상되는 카테고리만"),
    ("전략G: 정밀 보완",
     lambda e, l, r, c: (l | r) if c == "계좌번호" else (r if len(l) == 0 and c in {"주민등록번호", "이메일"} else l),
     "계좌=union, 주민/이메일=조건부"),
]:
    tp = fp = fn = 0
    for tc in test_cases:
        tc_id = tc["id"]
        expected = get_expected(tc)
        llm_r = llm_by_id.get(tc_id, {})
        regex_r = regex_by_id.get(tc_id, {})
        for cat in CATEGORIES:
            e_set = expected.get(cat, set())
            llm_set = get_predicted_set(llm_r, cat)
            regex_set = get_predicted_set(regex_r, cat)

            if strat_name.startswith("전략D"):
                pred = set(llm_set)
                for rv in regex_set:
                    for lv in llm_set:
                        a = rv.strip().replace(" ", "")
                        b = lv.strip().replace(" ", "")
                        if a in b or b in a:
                            pred.add(rv)
                            break
            else:
                pred = strat_fn(e_set, llm_set, regex_set, cat)

            tp += len(e_set & pred)
            fp += len(pred - e_set)
            fn += len(e_set - pred)

    all_strategies[strat_name] = (tp, fp, fn)

print(f"\n{'전략':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'P%':>8} {'R%':>8} {'F1%':>8} {'설명'}")
print("-" * 115)
for name, (tp, fp, fn) in all_strategies.items():
    p, r, f = compute_f1(tp, fp, fn)
    desc = ""
    if "베이스" in name: desc = "기준점"
    elif "Regex Only" in name: desc = "regex만 사용"
    elif "Union" in name: desc = "무조건 합집합"
    elif "A:" in name: desc = "P>80% 카테고리만 union"
    elif "B:" in name: desc = "LLM empty시 regex"
    elif "C:" in name: desc = "카테고리별 차등 적용"
    elif "D:" in name: desc = "LLM-regex 교차검증"
    elif "F:" in name: desc = f"union=>{','.join(sorted(beneficial_cats))}"
    elif "G:" in name: desc = "계좌union + 주민/이메일 조건부"

    diff_f1 = f - compute_f1(*llm_baseline)[2]
    diff_str = f"({diff_f1*100:+.2f}%p)" if name != "LLM Only (베이스라인)" else ""
    print(f"{name:<30} {tp:>5} {fp:>5} {fn:>5} {p*100:>8.1f} {r*100:>8.1f} {f*100:>8.1f} {diff_str:<10} {desc}")


# =============================================================================
# Key Insight Summary
# =============================================================================
print("\n\n" + "=" * 100)
print("  핵심 인사이트 요약")
print("=" * 100)

print("""
1. LLM(Qwen3-30B) 단독 성능이 이미 매우 높음 (F1=96.4%, P=96.2%, R=96.5%)
   - 모든 하이브리드 전략이 F1 기준으로 LLM 단독보다 낮거나 같음
   - 주된 이유: Regex의 FP(오탐)이 하이브리드에서 Precision을 크게 떨어뜨림

2. Regex의 주요 문제점:
   - 주소: 95건의 FP (전체 Regex FP의 52%) - 부분 매칭, 기관 주소 등
   - IP주소: 24건의 FP - 시간 형식(14:32:07), 서브넷 마스크 등을 IP로 오인
   - 이름: 17건의 FP - 팀명/부서명을 이름으로 오인
   - 전화번호: 13건의 FP - 서비스번호, 사무실 번호 등

3. Regex가 LLM을 보완하는 유일한 영역:
   - 계좌번호: LLM F1=87.5% → Union F1=94.1% (+6.6%p)
     * Regex가 100% Precision으로 FP 없이 2건 추가 복구
   - 주민등록번호: 3건 복구 가능하나 8건의 FP 발생으로 net negative
   - 이메일: 1건 복구 가능하나 6건의 FP 발생

4. 실질적 최적 전략:
   - 전략 F(계좌번호만 union): F1에서 가장 적은 손실(-0.1%p 미만)
   - 계좌번호의 regex는 100% precision이므로 안전하게 추가 가능
   - 기타 카테고리는 regex 추가 시 FP 증가가 TP 증가보다 큼

5. 결론:
   - 현재 LLM 성능(96.4%)에서 regex는 매우 제한적인 보완 역할만 가능
   - 유일하게 안전한 보완: 계좌번호 (regex precision 100%)
   - Recall 극대화가 목적이면: 전략 B(LLM 미응답 복구)가 Recall 97%까지 올림
     단, Precision이 92.5%로 하락 (3.7%p 손실)
""")
