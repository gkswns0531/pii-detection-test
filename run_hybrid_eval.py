#!/usr/bin/env python3
"""Hybrid evaluation: merge LLM predictions with regex predictions and compute stats.

Usage:
    python run_hybrid_eval.py <llm_results.json> <test_cases.json> <output.json>
"""

import json
import sys
from pathlib import Path
from regex_pii_detector import detect_pii_regex, merge_predictions


CATS = [
    "이름", "주소", "주민등록번호", "여권번호", "운전면허번호",
    "이메일", "IP주소", "전화번호", "계좌번호", "카드번호",
    "생년월일", "기타_고유식별정보",
]


def load_expected(test_cases: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Convert test cases expected_pii to {id: {cat: [values]}} format."""
    expected_map = {}
    for tc in test_cases:
        exp: dict[str, list[str]] = {}
        for item in tc.get("expected_pii", []):
            cat = item["type"]
            if cat not in exp:
                exp[cat] = []
            exp[cat].append(item["value"])
        expected_map[tc["id"]] = exp
    return expected_map


def compute_f1(expected: dict, predicted: dict) -> tuple[float, int, int, int]:
    """Compute F1, TP, FP, FN for a single case."""
    tp = fp = fn = 0
    for cat in CATS:
        e_set = set(expected.get(cat, []))
        p_set = set(predicted.get(cat) or [])
        tp += len(e_set & p_set)
        fp += len(p_set - e_set)
        fn += len(e_set - p_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, tp, fp, fn


def main() -> None:
    llm_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/results_qwen3_30b_v2_300.json"
    tc_path = sys.argv[2] if len(sys.argv) > 2 else "combined_test_cases.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "benchmark_results/hybrid_results.json"

    # Load LLM results
    with open(llm_path, encoding="utf-8") as f:
        llm_data = json.load(f)

    # Load test cases
    with open(tc_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    expected_map = load_expected(test_cases)
    tc_map = {tc["id"]: tc for tc in test_cases}

    # Build LLM prediction map
    llm_results = llm_data.get("results", llm_data) if isinstance(llm_data, dict) else llm_data
    if isinstance(llm_results, dict):
        llm_results = llm_results.get("results", [])

    llm_pred_map: dict[str, dict] = {}
    for r in llm_results:
        tc_id = r["id"]
        pred = {}
        for cat in CATS:
            val = r.get("predicted", r.get("predictions", {})).get(cat)
            if val:
                pred[cat] = val if isinstance(val, list) else [val]
        llm_pred_map[tc_id] = pred

    # Compute stats for LLM-only, Regex-only, and Hybrid
    methods = {"llm": {}, "regex": {}, "hybrid": {}, "smart": {}}
    for method in methods:
        methods[method] = {"tp": 0, "fp": 0, "fn": 0, "perfect": 0, "results": []}

    for tc in test_cases:
        tc_id = tc["id"]
        expected = expected_map[tc_id]
        doc_text = tc["document_text"]

        # LLM prediction
        llm_pred = llm_pred_map.get(tc_id, {})

        # Regex prediction
        regex_raw = detect_pii_regex(doc_text)
        regex_pred = {k: v for k, v in regex_raw.items() if v is not None}

        # Hybrid = union (all categories)
        hybrid_pred = merge_predictions(
            {cat: llm_pred.get(cat) for cat in CATS},
            regex_raw,
        )
        hybrid_pred_clean = {k: v for k, v in hybrid_pred.items() if v is not None}

        # Smart hybrid = LLM + regex for 계좌번호 only
        smart_pred = dict(llm_pred)
        regex_acct = regex_raw.get("계좌번호")
        if regex_acct:
            llm_acct = set(llm_pred.get("계좌번호") or [])
            merged_acct = sorted(llm_acct | set(regex_acct))
            smart_pred["계좌번호"] = merged_acct

        for method, pred in [("llm", llm_pred), ("regex", regex_pred), ("hybrid", hybrid_pred_clean), ("smart", smart_pred)]:
            f1, tp, fp, fn = compute_f1(expected, pred)
            methods[method]["tp"] += tp
            methods[method]["fp"] += fp
            methods[method]["fn"] += fn
            if f1 == 1.0:
                methods[method]["perfect"] += 1
            methods[method]["results"].append({
                "id": tc_id,
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            })

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON: LLM-only vs Regex-only vs Hybrid (LLM + Regex)")
    print("=" * 70)
    header = "{:<12} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}".format(
        "Method", "TP", "FP", "FN", "Prec", "Recall", "F1", "Perfect"
    )
    print(header)
    print("-" * 70)

    summary = {}
    total = len(test_cases)
    for method in ["llm", "regex", "hybrid", "smart"]:
        m = methods[method]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        r = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        perfect = m["perfect"]
        print("{:<12} {:>6} {:>6} {:>6} {:>7.1f}% {:>7.1f}% {:>7.1f}% {:>4}/{:>3}".format(
            method.upper(), tp, fp, fn, p, r, f1, perfect, total
        ))
        summary[method] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 1), "recall": round(r, 1), "f1": round(f1, 1),
            "perfect": perfect, "total": total,
            "accuracy": round(perfect / total * 100, 1),
        }

    print()

    # Save
    output = {
        "summary": summary,
        "per_case": {m: methods[m]["results"] for m in methods},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
