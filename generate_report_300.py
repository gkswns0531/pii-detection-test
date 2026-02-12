#!/usr/bin/env python3
"""HTML report generator for 12-model × 300-case PII benchmark.

Reads result + latency JSONs from benchmark_results/300/,
splits into base/advanced/combined, generates interactive HTML.
"""

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "benchmark_results" / "300"
OUTPUT = Path(__file__).parent / "benchmark_results" / "report.html"

CATS = [
    "이름", "주소", "주민등록번호", "여권번호", "운전면허번호",
    "이메일", "IP주소", "전화번호", "계좌번호", "카드번호",
    "생년월일", "기타_고유식별정보",
]

MODELS = [
    ("qwen3_30b_a3b_fp8", "Qwen3-30B-A3B", "30B", "#2563eb"),
    ("gpt_oss_20b", "GPT-OSS-20B", "20B", "#1e3a5f"),
    ("qwen3_8b", "Qwen3-8B", "8B", "#3b82f6"),
    ("qwen3_4b_2507", "Qwen3-4B", "4B", "#60a5fa"),
    ("qwen3_1.7b", "Qwen3-1.7B", "1.7B", "#93c5fd"),
    ("qwen3_0.6b", "Qwen3-0.6B", "0.6B", "#bfdbfe"),
    ("smollm3_3b", "SmolLM3-3B", "3B", "#10b981"),
    ("falcon_h1r_7b", "Falcon-H1R-7B", "7B", "#f59e0b"),
    ("gemma3_4b", "Gemma3-4B", "4B", "#8b5cf6"),
    ("gemma3_1b", "Gemma3-1B", "1B", "#a78bfa"),
    ("llama32_3b", "Llama3.2-3B", "3B", "#ef4444"),
    ("llama32_1b", "Llama3.2-1B", "1B", "#fca5a5"),
]

# Case study analysis for Qwen3-30B-A3B imperfect cases
CASE_STUDY_ANALYSIS: dict[str, str] = {
    "TC004": "모델이 '강서구에 거주하는 윤서연'에서 '강서구'를 주소 PII로 추가 예측. 이름은 4명 모두 정확히 검출했으나, 문맥 속 지명을 주소로 과잉 해석.",
    "TC006": "지명('김포', '이천', '한강')과 이름('이천호', '한강수')이 공존하는 문서에서, 이름 3개는 정확히 검출했으나 '마포구 한강로'를 주소 PII로 추가 예측.",
    "TC008": "한자 이름 '李美英'을 '李美영'으로 예측. 한자-한글 혼용 이름에서 마지막 글자의 한자 인식 오류. 한자 OCR/인코딩 경계 문제.",
    "TC022": "붙여쓰기된 주소 4개를 인식은 했으나, 공백/괄호 포함 범위가 ground truth와 불일치. 예: 기대 '서울시강남구 테헤란로152번길 (역삼동)' vs 출력 '서울시강남구테헤란로152번길'. Boundary 정의의 모호성 문제.",
    "TC030": "주민등록번호 앞 6자리('870214')만 노출된 경우를 PII로 인식하지 못함. 6자리 숫자만으로는 주민번호로 단정하기 어렵다는 보수적 판단.",
    "TC045": "'macbook.seller 골뱅이 gmail 닷 com'을 검출하지 못함. 한글로 치환된 '@'('골뱅이')와 '.'('닷')를 이메일 구성요소로 인식하는 데 실패.",
    "TC050": "모델이 직함/역할명을 이름으로 과탐. IP주소 검출은 정확하나 이름 카테고리에서 FP 발생.",
    "TC058": "재실행에서 전체 실패(F1=0.0). 이전 실행(F1=0.76)에서는 부분 검출이 있었으나, 동일 프롬프트에서 결과가 비결정적으로 악화. LLM 비결정성(non-determinism)에 의한 것으로 추정.",
    "TC062": "내선번호가 포함된 전화번호 '02-3456-7001 내선 1102'에서 내선 부분을 누락. 프롬프트에 내선번호 포함 규칙이 있으나 모델이 이를 적용하지 못함.",
    "TC063": "호텔 대표번호 '(02)771-2200'과 '02 - 771 - 2201'을 개인 전화번호로 과탐. 비표준 구분자(·, 공백, 점) 전화번호는 정확히 검출.",
    "TC064": "'010-****-3456'(마스킹된 이전 번호)을 누락. 수정 이력 맥락에서의 과거 전화번호 검출 실패.",
    "TC069": "세금계산서에서 모든 PII 검출 실패(이름 2건, 계좌번호 1건 누락). 사업자등록번호/법인등록번호와 계좌번호가 혼재된 문서에서 모델이 혼동한 극단적 실패 사례.",
    "TC070": "가상계좌 3건 중 현재 활성 계좌만 검출하고, 이전 주문의 계좌 2건을 누락. 문맥상 '과거' 계좌에 대한 검출 누락.",
    "TC071": "IBAN 형식 계좌(DE89..., GB29...)를 계좌번호로 인식하지 못함. 국내 은행 계좌는 검출했으나 국제 계좌번호 형식 지원 부족.",
    "TC072": "암호화폐 지갑 주소 7건(BTC 4건, ETH 3건)을 전혀 인식하지 못함. 프롬프트에 '카드번호' 카테고리에 암호화폐 지갑 주소가 포함된다고 명시했으나 모델이 이를 적용하지 못함.",
    "TC078": "회의록 작성자 '송다영'을 이름으로 누락. 문서 말미의 작성자 정보가 참여자 이름과 분리되어 놓친 것으로 추정.",
    "TC080": "입사지원서에서 지도교수 '김영호'를 이름으로 누락. 본 지원자(한소희)의 정보는 완벽히 검출했으나 부수적 인물 누락.",
    "TC083": "소장에서 '위 원고 1과 같음'이라는 주소 참조 표현을 주소로 추가 예측. 법률 문서의 참조 표현 해석 오류.",
    "TC096": "Python 설정 코드에서 127.0.0.1, 8.8.8.8 등 코드 내 IP 리터럴을 실제 PII로 과탐. 프로그래밍 컨텍스트 인식 부족.",
    "TC098": "'비밀번호 앞 두자리 구오'의 '구오'를 기타_고유식별정보로 과탐. 8가지 난독화 PII 중 7가지를 완벽히 검출한 것은 인상적.",
    "TC099": "OCR 오류 이름 '긤철수'→'금철수', '긤영회'→'금영회'로 다르게 인식. OCR 오류 문자의 정확한 재현에 한계.",
}

FAILURE_PATTERNS = [
    ("비전통적 금융정보", "3건", "TC069, TC071, TC072", "IBAN, 암호화폐 지갑, 세금계산서 내 계좌 미검출"),
    ("문맥 지명→주소 과탐", "2건", "TC004, TC006", "지명을 주소 PII로 과탐"),
    ("Boundary 불일치", "1건", "TC022", "주소 공백/괄호 포함 범위가 ground truth와 불일치"),
    ("범위(Scope) 판단 차이", "4건", "TC058, TC062, TC064, TC070", "기업전화, 내선번호, 이전번호, 과거계좌"),
    ("난독화/OCR 한계", "3건", "TC045, TC098, TC099", "한글 치환 이메일, 한글 숫자 과탐, OCR 문자 교정"),
    ("부수적 인물 누락", "2건", "TC078, TC080", "작성자, 지도교수 등 부수적 인물 이름 누락"),
    ("코드/설정 IP 과탐", "1건", "TC096", "코드 내 IP 리터럴을 PII로 과탐"),
    ("대표번호 과탐", "1건", "TC063", "호텔 대표번호를 개인 전화번호로 과탐"),
    ("직함/역할명 과탐", "1건", "TC050", "직함이나 역할명을 이름으로 과탐"),
    ("한자 인식 오류", "1건", "TC008", "한자-한글 혼용 이름 마지막 글자 오인"),
    ("법률문서 참조 과탐", "1건", "TC083", "'위 원고 1과 같음'을 주소로 과탐"),
    ("주민번호 앞자리 누락", "1건", "TC030", "마스킹된 주민번호 앞 6자리만 노출 시 미검출"),
]


def normalize_difficulty(diff: str) -> str:
    return "base" if diff == "EASY" else "advanced"


def compute_stats(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0, "perfect": 0, "acc": 0, "p": 0, "r": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0}
    perfect = sum(1 for r in results if r["f1"] == 1.0)
    tp = fp = fn = 0
    for r in results:
        exp = r.get("expected", {})
        pred = r.get("predicted", {})
        for cat in CATS:
            e_set = set(exp.get(cat, []) or [])
            p_set = set(pred.get(cat, []) or [])
            tp += len(e_set & p_set)
            fp += len(p_set - e_set)
            fn += len(e_set - p_set)
    p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    r_ = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0
    acc = perfect / total * 100 if total > 0 else 0
    return {
        "total": total, "perfect": perfect,
        "acc": round(acc, 1), "p": round(p, 1), "r": round(r_, 1), "f1": round(f1, 1),
        "tp": tp, "fp": fp, "fn": fn,
    }


def compute_confusion_matrix(results: list[dict]) -> dict:
    tp = tn = fp = fn = 0
    for r in results:
        exp = r.get("expected", {})
        pred = r.get("predicted", {})
        for cat in CATS:
            e = exp.get(cat, []) or []
            p = pred.get(cat, []) or []
            has_e = len(e) > 0
            has_p = len(p) > 0
            if has_e and has_p:
                tp += 1
            elif not has_e and not has_p:
                tn += 1
            elif not has_e and has_p:
                fp += 1
            else:
                fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_category_confusion(results: list[dict]) -> dict[str, dict]:
    cm = {cat: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for cat in CATS}
    for r in results:
        exp = r.get("expected", {})
        pred = r.get("predicted", {})
        for cat in CATS:
            e = exp.get(cat, []) or []
            p = pred.get(cat, []) or []
            has_e = len(e) > 0
            has_p = len(p) > 0
            if has_e and has_p:
                cm[cat]["tp"] += 1
            elif not has_e and not has_p:
                cm[cat]["tn"] += 1
            elif not has_e and has_p:
                cm[cat]["fp"] += 1
            else:
                cm[cat]["fn"] += 1
    return cm


def load_test_cases() -> dict[str, dict]:
    """Load test cases for document_text lookup."""
    tc_path = Path(__file__).parent / "combined_test_cases.json"
    with open(tc_path, encoding="utf-8") as f:
        tcs = json.load(f)
    return {tc["id"]: tc for tc in tcs}


def load_all_models() -> dict[str, dict]:
    """Load all model results and compute per-split stats."""
    all_models: dict[str, dict] = {}
    tc_map = load_test_cases()

    for fname, label, size, color in MODELS:
        rpath = RESULTS_DIR / f"results_{fname}.json"
        lpath = RESULTS_DIR / f"latency_{fname}.json"
        if not rpath.exists():
            continue

        with open(rpath, encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        for r in results:
            r["difficulty"] = normalize_difficulty(r["difficulty"])
            tc = tc_map.get(r.get("id", ""), {})
            r["document_text"] = tc.get("document_text", "")

        base = [r for r in results if r["difficulty"] == "base"]
        advanced = [r for r in results if r["difficulty"] == "advanced"]

        latency = None
        if lpath.exists():
            with open(lpath, encoding="utf-8") as f:
                lat_data = json.load(f)
            latency = lat_data.get("statistics", {})

        all_models[fname] = {
            "label": label,
            "size": size,
            "color": color,
            "inference_time": data.get("inference_time_sec", 0),
            "latency": latency,
            "stats": {
                "combined": compute_stats(results),
                "base": compute_stats(base),
                "advanced": compute_stats(advanced),
            },
            "confusion": {
                "combined": compute_confusion_matrix(results),
                "base": compute_confusion_matrix(base),
                "advanced": compute_confusion_matrix(advanced),
            },
            "cat_confusion": {
                "combined": compute_category_confusion(results),
                "base": compute_category_confusion(base),
                "advanced": compute_category_confusion(advanced),
            },
            "results": {
                "combined": results,
                "base": base,
                "advanced": advanced,
            },
        }

    return all_models


def build_case_study_data(all_models: dict[str, dict]) -> str:
    """Build JS data for the Qwen3-30B-A3B case study section."""
    model_id = "qwen3_30b_a3b_fp8"
    if model_id not in all_models:
        return "[]"
    results = all_models[model_id]["results"]["combined"]
    imperfect = [r for r in results if r["f1"] < 1.0]
    imperfect.sort(key=lambda r: r["f1"])

    cases = []
    for r in imperfect:
        tc_id = r["id"]
        # Build comparison rows
        exp = r.get("expected", {})
        pred = r.get("predicted", {})
        all_types = list(dict.fromkeys(list(exp.keys()) + list(pred.keys())))

        rows = []
        for t in all_types:
            e_vals = exp.get(t, []) or []
            p_vals = pred.get(t, []) or []
            for v in e_vals:
                if v in p_vals:
                    rows.append({"type": t, "exp": v, "pred": v, "status": "match"})
                else:
                    rows.append({"type": t, "exp": v, "pred": None, "status": "miss"})
            for v in p_vals:
                if v not in e_vals:
                    rows.append({"type": t, "exp": None, "pred": v, "status": "fp"})

        cases.append({
            "id": tc_id,
            "category": r.get("category", ""),
            "difficulty": r.get("difficulty", ""),
            "f1": r["f1"],
            "intent": r.get("intent", ""),
            "document_text": r.get("document_text", ""),
            "analysis": CASE_STUDY_ANALYSIS.get(tc_id, ""),
            "rows": rows,
        })
    return json.dumps(cases, ensure_ascii=False)


def build_failure_patterns_json() -> str:
    """Build JS data for failure pattern taxonomy."""
    return json.dumps([
        {"type": t, "count": c, "cases": cs, "desc": d}
        for t, c, cs, d in FAILURE_PATTERNS
    ], ensure_ascii=False)


def build_html(all_models: dict[str, dict]) -> str:
    # Prepare JS-friendly data
    model_list = []
    for fname, label, size, color in MODELS:
        if fname not in all_models:
            continue
        m = all_models[fname]
        lat = m["latency"]
        model_list.append({
            "id": fname,
            "label": label,
            "size": size,
            "color": color,
            "inference_time": m["inference_time"],
            "latency_mean": lat["mean_sec"] if lat else None,
            "latency_p95": lat.get("p95_sec") if lat else None,
            "stats": m["stats"],
            "confusion": m["confusion"],
            "cat_confusion": m["cat_confusion"],
        })

    # Case data for browser (only store combined to save space)
    case_data = {}
    for fname in all_models:
        case_data[fname] = all_models[fname]["results"]["combined"]

    model_list_json = json.dumps(model_list, ensure_ascii=False)
    case_data_json = json.dumps(case_data, ensure_ascii=False)
    cats_json = json.dumps(CATS, ensure_ascii=False)
    case_study_json = build_case_study_data(all_models)
    failure_patterns_json = build_failure_patterns_json()

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PII Detection - 12 Model Benchmark Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
.subtitle {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
.section-title {{ font-size: 18px; font-weight: 700; margin: 32px 0 14px; }}
.desc-box {{ background: #fff; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 13px; color: #475569; box-shadow: 0 1px 4px rgba(0,0,0,0.06); line-height: 1.7; }}
.model-select {{ padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 13px; font-weight: 600; background: #fff; color: #1a1a2e; cursor: pointer; min-width: 180px; }}

/* Summary Cards */
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 28px; }}
.card {{ background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }}
.card .label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
.card .value {{ font-size: 28px; font-weight: 800; margin: 4px 0; }}
.card .sub {{ font-size: 12px; color: #aaa; }}

/* Tabs */
.tab-row {{ display: inline-flex; gap: 4px; margin-bottom: 16px; background: #fff; border-radius: 8px; padding: 3px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.tab-btn {{ padding: 8px 18px; border: none; background: transparent; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; color: #666; transition: all 0.15s; }}
.tab-btn.active {{ background: #3b82f6; color: #fff; }}
.tab-btn:hover:not(.active) {{ background: #f0f0f0; }}

/* Main Stats Table */
.stats-wrap {{ overflow-x: auto; margin-bottom: 28px; }}
table.main {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
table.main th {{ background: #f8fafc; padding: 10px 12px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; font-weight: 600; text-align: right; border-bottom: 2px solid #e2e8f0; white-space: nowrap; }}
table.main th:first-child {{ text-align: left; }}
table.main td {{ padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #f1f5f9; text-align: right; white-space: nowrap; }}
table.main td:first-child {{ text-align: left; font-weight: 600; }}
table.main tr:hover {{ background: #f8fafc; }}
table.main tr.best {{ background: #eff6ff; }}
.rank {{ display: inline-block; width: 20px; height: 20px; border-radius: 50%; text-align: center; line-height: 20px; font-size: 11px; font-weight: 700; color: #fff; margin-right: 6px; }}
.rank.r1 {{ background: #f59e0b; }}
.rank.r2 {{ background: #94a3b8; }}
.rank.r3 {{ background: #b45309; }}

/* Bar Chart */
.bar-chart {{ margin-bottom: 28px; }}
.bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
.bar-label {{ width: 140px; font-size: 13px; font-weight: 600; text-align: right; flex-shrink: 0; }}
.bar-track {{ flex: 1; background: #e5e7eb; border-radius: 8px; height: 26px; position: relative; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 8px; display: flex; align-items: center; padding: 0 10px; font-size: 12px; font-weight: 700; color: #fff; transition: width 0.6s ease; min-width: 30px; }}

/* Confusion Matrix */
.cm-container {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 28px; }}
.cm-box {{ background: #fff; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); min-width: 320px; }}
.cm-box h3 {{ font-size: 14px; font-weight: 700; margin-bottom: 12px; text-align: center; }}
.cm-grid {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 0; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }}
.cm-corner {{ background: #f8fafc; padding: 8px 12px; font-size: 10px; color: #94a3b8; font-weight: 600; text-transform: uppercase; display: flex; align-items: center; justify-content: center; }}
.cm-header {{ background: #f8fafc; padding: 8px 12px; text-align: center; font-size: 11px; font-weight: 700; color: #475569; border-bottom: 2px solid #e2e8f0; }}
.cm-row-header {{ background: #f8fafc; padding: 8px 12px; font-size: 11px; font-weight: 700; color: #475569; border-right: 2px solid #e2e8f0; }}
.cm-cell {{ padding: 14px 12px; text-align: center; font-size: 20px; font-weight: 800; }}
.cm-cell .pct {{ font-size: 10px; font-weight: 500; color: #64748b; display: block; margin-top: 2px; }}
.cm-cell.tp {{ background: #dcfce7; color: #166534; }}
.cm-cell.tn {{ background: #f0fdf4; color: #166534; }}
.cm-cell.fp {{ background: #fef2f2; color: #991b1b; }}
.cm-cell.fn {{ background: #fff7ed; color: #9a3412; }}

/* Per-Category CM Table */
.cat-cm-table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 28px; }}
.cat-cm-table th {{ background: #f8fafc; padding: 8px 10px; font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; border-bottom: 2px solid #e2e8f0; text-align: center; }}
.cat-cm-table th:first-child {{ text-align: left; }}
.cat-cm-table td {{ padding: 8px 10px; font-size: 13px; border-bottom: 1px solid #f1f5f9; text-align: center; }}
.cat-cm-table td:first-child {{ text-align: left; font-weight: 600; }}
.cat-cm-table tr:hover {{ background: #f8fafc; }}

/* Filters */
.filters {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
.filter-group {{ display: flex; gap: 4px; background: #fff; border-radius: 8px; padding: 3px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.filter-btn {{ padding: 6px 14px; border: none; background: transparent; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; color: #666; transition: all 0.15s; }}
.filter-btn.active {{ background: #3b82f6; color: #fff; }}
.filter-btn:hover:not(.active) {{ background: #f0f0f0; }}
.filter-label {{ font-size: 12px; color: #888; font-weight: 600; text-transform: uppercase; margin-right: 4px; }}

/* Case List */
.case-card {{ background: #fff; border-radius: 10px; padding: 14px 16px; margin-bottom: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); cursor: pointer; transition: all 0.15s; }}
.case-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
.case-card.expanded {{ box-shadow: 0 2px 12px rgba(0,0,0,0.15); }}
.case-top {{ display: flex; justify-content: space-between; align-items: center; }}
.case-meta {{ display: flex; gap: 6px; align-items: center; }}
.case-id {{ font-weight: 700; font-size: 14px; margin-right: 6px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.badge.cat {{ background: #e0e7ff; color: #3730a3; }}
.case-f1 {{ font-weight: 700; font-size: 14px; }}
.case-f1.perfect {{ color: #10b981; }}
.case-f1.good {{ color: #3b82f6; }}
.case-f1.bad {{ color: #ef4444; }}
.case-intent {{ font-size: 13px; color: #666; margin-top: 4px; }}
.case-detail {{ display: none; margin-top: 12px; border-top: 1px solid #f1f5f9; padding-top: 12px; }}
.case-card.expanded .case-detail {{ display: block; }}
.doc-text {{ background: #f8fafc; padding: 12px; border-radius: 8px; font-size: 12px; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; margin-bottom: 12px; border: 1px solid #e2e8f0; }}
.cat-row {{ display: flex; align-items: flex-start; padding: 5px 0; border-bottom: 1px solid #f8fafc; font-size: 13px; }}
.cat-status {{ width: 36px; text-align: center; font-weight: 700; flex-shrink: 0; }}
.cat-status.ok {{ color: #10b981; }} .cat-status.fp {{ color: #ef4444; }} .cat-status.fn {{ color: #f59e0b; }} .cat-status.ne {{ color: #ef4444; }}
.cat-name {{ width: 100px; font-weight: 600; flex-shrink: 0; color: #374151; }}
.cat-values {{ flex: 1; }} .cat-values .exp {{ color: #059669; font-size: 12px; }} .cat-values .pred {{ color: #2563eb; font-size: 12px; }} .cat-values .missing {{ color: #dc2626; font-size: 12px; }}
.pagination {{ display: flex; justify-content: center; gap: 6px; margin-top: 12px; }}
.page-btn {{ padding: 6px 12px; border: 1px solid #e2e8f0; background: #fff; border-radius: 6px; cursor: pointer; font-size: 13px; }}
.page-btn.active {{ background: #3b82f6; color: #fff; border-color: #3b82f6; }}
.page-btn:hover:not(.active) {{ background: #f0f0f0; }}
.case-count {{ font-size: 13px; color: #888; }}

/* Case Study */
.cs-summary {{ background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }}
.cs-summary h3 {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; }}
.cs-kpi {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
.cs-kpi-item {{ background: #f8fafc; border-radius: 8px; padding: 12px 16px; text-align: center; min-width: 100px; }}
.cs-kpi-item .num {{ font-size: 24px; font-weight: 800; }}
.cs-kpi-item .lbl {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.fp-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.fp-table th {{ background: #f8fafc; padding: 8px 10px; text-align: left; font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; border-bottom: 2px solid #e2e8f0; }}
.fp-table td {{ padding: 8px 10px; border-bottom: 1px solid #f1f5f9; }}
.fp-table tr:hover {{ background: #f8fafc; }}
.cs-card {{ background: #fff; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow: hidden; }}
.cs-header {{ padding: 14px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
.cs-header:hover {{ background: #f8fafc; }}
.cs-header .left {{ display: flex; gap: 8px; align-items: center; }}
.cs-header .tc-id {{ font-weight: 700; font-size: 14px; }}
.cs-header .tc-f1 {{ font-weight: 700; font-size: 14px; }}
.cs-header .tc-f1.zero {{ color: #ef4444; }}
.cs-header .tc-f1.low {{ color: #f59e0b; }}
.cs-header .tc-f1.mid {{ color: #3b82f6; }}
.cs-header .tc-f1.high {{ color: #10b981; }}
.cs-body {{ display: none; padding: 0 16px 16px; }}
.cs-card.open .cs-body {{ display: block; }}
.cs-doc {{ background: #f8fafc; padding: 12px; border-radius: 8px; font-size: 12px; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; margin-bottom: 12px; border: 1px solid #e2e8f0; }}
.cs-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 12px; }}
.cs-table th {{ background: #f8fafc; padding: 6px 10px; text-align: left; font-size: 11px; font-weight: 600; color: #64748b; border-bottom: 2px solid #e2e8f0; }}
.cs-table td {{ padding: 6px 10px; border-bottom: 1px solid #f1f5f9; }}
.cs-table .match {{ color: #10b981; }}
.cs-table .miss {{ color: #f59e0b; }}
.cs-table .fp-mark {{ color: #ef4444; }}
.cs-analysis {{ background: #eff6ff; border-left: 3px solid #3b82f6; padding: 10px 14px; border-radius: 4px; font-size: 13px; color: #1e3a5f; line-height: 1.6; }}

@media (max-width: 768px) {{
  .cm-container {{ flex-direction: column; }}
  .bar-label {{ width: 100px; font-size: 12px; }}
}}
</style>
</head>
<body>
<div class="container">
  <h1>PII Detection - 12 Model Benchmark Report</h1>
  <p class="subtitle">300 Test Cases (Base 200 + Advanced 100) &middot; V1 Full Prompt &middot; FP8 Quantization &middot; NVIDIA L40S 46GB</p>

  <!-- Summary Cards -->
  <div id="summary-cards" class="summary-grid"></div>

  <!-- Score Comparison -->
  <div class="section-title">Score Comparison</div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;">
    <div class="tab-row" id="bar-metric-tabs">
      <button class="tab-btn active" onclick="switchBarMetric('r',this)">Recall</button>
      <button class="tab-btn" onclick="switchBarMetric('p',this)">Precision</button>
      <button class="tab-btn" onclick="switchBarMetric('f1',this)">F1</button>
      <button class="tab-btn" onclick="switchBarMetric('perfect',this)">Perfect %</button>
      <button class="tab-btn" onclick="switchBarMetric('latency',this)">Latency</button>
    </div>
    <div class="tab-row" id="bar-tabs">
      <button class="tab-btn active" onclick="switchBar('combined',this)">Combined (300)</button>
      <button class="tab-btn" onclick="switchBar('base',this)">Base (200)</button>
      <button class="tab-btn" onclick="switchBar('advanced',this)">Advanced (100)</button>
    </div>
  </div>
  <div id="bar-metric-desc" class="desc-box"></div>
  <div class="desc-box" style="margin-top:0;">
    <strong>Base (200)</strong>: 명확한 레이블과 정형화된 문서에서의 기본 PII 검출 &middot;
    <strong>Advanced (100)</strong>: 난독화, OCR 오류, 혼합 문서, 엣지케이스 등 노이즈가 반영된 어려운 상황 &middot;
    <strong>Combined (300)</strong>: Base + Advanced 전체
  </div>
  <div id="bar-chart" class="bar-chart"></div>

  <!-- Stats Table -->
  <div class="section-title">Detailed Statistics</div>
  <div class="tab-row" id="stats-tabs">
    <button class="tab-btn active" onclick="switchStats('combined',this)">Combined (300)</button>
    <button class="tab-btn" onclick="switchStats('base',this)">Base (200)</button>
    <button class="tab-btn" onclick="switchStats('advanced',this)">Advanced (100)</button>
  </div>
  <div class="stats-wrap">
  <table class="main" id="stats-table">
    <thead><tr>
      <th style="text-align:left">Model</th><th>Cases</th><th>Perfect</th><th>Accuracy</th>
      <th>Precision</th><th>Recall</th><th>F1</th>
      <th>TP</th><th style="color:#ef4444">FP</th><th style="color:#f59e0b">FN</th>
      <th>Latency</th>
    </tr></thead>
    <tbody id="stats-body"></tbody>
  </table>
  </div>

  <!-- Confusion Matrix -->
  <div class="section-title">Confusion Matrix (Document-Category Level)</div>
  <div style="background:#fff;border-radius:12px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:16px;font-size:13px;color:#475569;line-height:1.9;">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;margin-bottom:10px;">
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:#dcfce7;border:1px solid #166534;margin-right:6px;vertical-align:middle;"></span><strong style="color:#166534;">TP (True Positive)</strong> — 실제 PII를 모델이 정확히 탐지한 건수. 높을수록 탐지 능력이 우수합니다.</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:#f0fdf4;border:1px solid #166534;margin-right:6px;vertical-align:middle;"></span><strong style="color:#166534;">TN (True Negative)</strong> — PII가 없는 항목을 올바르게 무시한 건수. 높을수록 불필요한 알림이 적습니다.</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:#fef2f2;border:1px solid #991b1b;margin-right:6px;vertical-align:middle;"></span><strong style="color:#991b1b;">FP (False Positive)</strong> — PII가 없는데 잘못 탐지한 건수(오탐). 높으면 사용자에게 불필요한 경고를 유발합니다.</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:#fff7ed;border:1px solid #9a3412;margin-right:6px;vertical-align:middle;"></span><strong style="color:#9a3412;">FN (False Negative)</strong> — 실제 PII를 놓친 건수(미탐). 높으면 개인정보 유출 위험이 증가합니다.</div>
    </div>
    <div style="border-top:1px solid #e2e8f0;padding-top:8px;font-size:12px;">
      <strong>Sensitivity</strong> = TP/(TP+FN): 실제 존재하는 PII 중 모델이 얼마나 빠짐없이 찾아내는지 (재현율) &nbsp;|&nbsp;
      <strong>Specificity</strong> = TN/(TN+FP): PII가 없는 항목을 얼마나 정확하게 무시하는지 (특이도)
    </div>
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:12px;">
    <span class="filter-label">Model</span>
    <select class="model-select" id="cm-model-select" onchange="cmModel=this.value;renderCM();"></select>
  </div>
  <div class="tab-row" id="cm-dataset-tabs">
    <button class="tab-btn active" onclick="switchCMDataset('combined',this)">Combined</button>
    <button class="tab-btn" onclick="switchCMDataset('base',this)">Base</button>
    <button class="tab-btn" onclick="switchCMDataset('advanced',this)">Advanced</button>
  </div>
  <div id="cm-area" class="cm-container"></div>

  <!-- Per-Category CM -->
  <div class="section-title">Per-Category Confusion Matrix</div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:12px;">
    <span class="filter-label">Model</span>
    <select class="model-select" id="catcm-model-select" onchange="catcmModel=this.value;renderCatCM();"></select>
  </div>
  <div class="stats-wrap">
  <table class="cat-cm-table" id="cat-cm-table">
    <thead><tr>
      <th style="text-align:left">Category</th>
      <th style="color:#166534">TP</th><th style="color:#4ade80">TN</th>
      <th style="color:#dc2626">FP</th><th style="color:#ea580c">FN</th>
      <th>Sensitivity</th><th>Specificity</th>
      <th style="text-align:left">Description</th>
    </tr></thead>
    <tbody id="cat-cm-body"></tbody>
  </table>
  </div>

  <!-- Case Browser -->
  <div class="section-title">Case Browser</div>
  <div class="filters">
    <span class="filter-label">Model</span>
    <select class="model-select" id="filter-model-select" onchange="curModel=this.value;curPage=0;renderCases();"></select>
    <span class="filter-label">Dataset</span>
    <div class="filter-group" id="filter-dataset">
      <button class="filter-btn active" data-val="combined">Combined</button>
      <button class="filter-btn" data-val="base">Base</button>
      <button class="filter-btn" data-val="advanced">Advanced</button>
    </div>
    <span class="filter-label">Result</span>
    <div class="filter-group" id="filter-type">
      <button class="filter-btn active" data-val="all">All</button>
      <button class="filter-btn" data-val="success">Success</button>
      <button class="filter-btn" data-val="fail">Fail</button>
    </div>
    <span class="filter-label">PII</span>
    <div class="filter-group" id="filter-pii">
      <button class="filter-btn active" data-val="all">All</button>
      <button class="filter-btn" data-val="positive">PII Present</button>
      <button class="filter-btn" data-val="negative">No PII</button>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <span class="case-count" id="case-count"></span>
  </div>
  <div id="case-list"></div>
  <div id="pagination" class="pagination"></div>

  <!-- Case Study: Qwen3-30B-A3B Error Analysis -->
  <div class="section-title">Case Study: Qwen3-30B-A3B Error Analysis</div>
  <div class="cs-summary">
    <h3>Qwen3-30B-A3B (MoE 30B, 3B active) &mdash; 불완전 케이스 심층 분석</h3>
    <div class="cs-kpi" id="cs-kpi"></div>
    <h3 style="margin-top:16px;">실패 패턴 분류</h3>
    <table class="fp-table" id="fp-table">
      <thead><tr><th>실패 유형</th><th>건수</th><th>대표 TC</th><th>설명</th></tr></thead>
      <tbody id="fp-body"></tbody>
    </table>
  </div>
  <div id="cs-cases"></div>
</div>

<script>
const MODELS = {model_list_json};
const CASE_DATA = {case_data_json};
const CATS = {cats_json};
const CS_CASES = {case_study_json};
const FP_PATTERNS = {failure_patterns_json};
const PER_PAGE = 20;
let curModel = MODELS[0]?.id || '', curDataset = 'combined', curType = 'all', curPii = 'all', curPage = 0;
let cmModel = MODELS[0]?.id || '', cmDataset = 'combined';
let catcmModel = MODELS[0]?.id || '', catcmDataset = 'combined';
let barDataset = 'combined';

function getModel(id) {{ return MODELS.find(m => m.id === id); }}

// ── Summary Cards (Qwen3-30B-A3B baseline) ──
function renderSummary() {{
  const m = getModel('qwen3_30b_a3b_fp8') || MODELS[0];
  const s = m.stats.combined;
  const lat = m.latency_mean ? m.latency_mean.toFixed(2) + 's' : '-';
  document.getElementById('summary-cards').innerHTML = `
    <div class="card" style="border-left:4px solid #10b981"><div class="label">Recall</div><div class="value" style="color:#10b981">${{s.r}}%</div><div class="sub">${{m.label}}</div></div>
    <div class="card" style="border-left:4px solid #3b82f6"><div class="label">Precision</div><div class="value" style="color:#3b82f6">${{s.p}}%</div><div class="sub">${{m.label}}</div></div>
    <div class="card" style="border-left:4px solid #2563eb"><div class="label">F1 Score</div><div class="value" style="color:#2563eb">${{s.f1}}%</div><div class="sub">${{m.label}}</div></div>
    <div class="card" style="border-left:4px solid #8b5cf6"><div class="label">Perfect</div><div class="value" style="color:#8b5cf6">${{s.perfect}}/300</div><div class="sub">${{(s.perfect/s.total*100).toFixed(1)}}%</div></div>
    <div class="card" style="border-left:4px solid #f59e0b"><div class="label">Latency</div><div class="value" style="color:#f59e0b">${{lat}}</div><div class="sub">${{m.label}} (mean)</div></div>
  `;
}}

// ── Bar Chart ──
let barMetric = 'r';
const METRIC_DESC = {{
  r: '<strong>Recall (재현율)</strong>: 실제 존재하는 PII 중 모델이 빠뜨리지 않고 찾아낸 비율. 높을수록 개인정보 유출 위험이 낮아집니다.',
  p: '<strong>Precision (정밀도)</strong>: 모델이 PII라고 예측한 것 중 실제로 PII인 비율. 높을수록 불필요한 마스킹/알림이 줄어듭니다.',
  f1: '<strong>F1 Score</strong>: Precision과 Recall의 조화 평균. 둘 사이의 균형을 하나의 숫자로 요약합니다.',
  perfect: '<strong>Perfect Match %</strong>: 완벽하게 모든 정보를 식별하고, 불필요한 정보를 식별하지 않은 비율. FP와 FN이 모두 0인 케이스만 해당됩니다.',
  latency: '<strong>Latency (응답시간)</strong>: 1건 처리 평균 소요시간. 낮을수록 실시간 처리에 유리합니다.',
}};
function getMetricVal(m, dataset, metric) {{
  const s = m.stats[dataset];
  if (metric === 'latency') return m.latency_mean || 0;
  if (metric === 'perfect') return s.total > 0 ? (s.perfect / s.total * 100) : 0;
  return s[metric] || 0;
}}
function renderBarChart(dataset) {{
  const metric = barMetric;
  const isLatency = metric === 'latency';
  const sorted = [...MODELS].sort((a, b) => {{
    const va = getMetricVal(a, dataset, metric);
    const vb = getMetricVal(b, dataset, metric);
    return isLatency ? va - vb : vb - va;
  }});
  const maxVal = isLatency ? Math.max(...sorted.map(m => getMetricVal(m, dataset, metric)), 1) : 100;
  let html = '';
  sorted.forEach((m, i) => {{
    const v = getMetricVal(m, dataset, metric);
    const pct = isLatency ? (v / maxVal * 100) : v;
    const label = isLatency ? v.toFixed(2) + 's' : v.toFixed(1) + '%';
    const hue = isLatency ? Math.max(0, 120 - (v / maxVal * 120)) : v * 1.2;
    const barColor = `hsl(${{hue}}, 70%, 50%)`;
    html += `<div class="bar-row">
      <div class="bar-label">${{m.label}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{Math.max(pct, 3)}}%;background:${{barColor}}">${{label}}</div></div>
    </div>`;
  }});
  document.getElementById('bar-chart').innerHTML = html;
}}
function switchBar(d, btn) {{
  barDataset = d;
  document.querySelectorAll('#bar-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderBarChart(d);
}}
function switchBarMetric(metric, btn) {{
  barMetric = metric;
  document.querySelectorAll('#bar-metric-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('bar-metric-desc').innerHTML = METRIC_DESC[metric] || '';
  renderBarChart(barDataset);
}}

// ── Stats Table ──
function renderStats(dataset) {{
  const sorted = [...MODELS].sort((a, b) => b.stats[dataset].r - a.stats[dataset].r);
  let html = '';
  sorted.forEach((m, i) => {{
    const s = m.stats[dataset];
    const rank = i < 3 ? `<span class="rank r${{i+1}}">${{i+1}}</span>` : '';
    const cls = i === 0 ? ' class="best"' : '';
    const lat = m.latency_mean ? m.latency_mean.toFixed(2) + 's' : '-';
    html += `<tr${{cls}}>
      <td>${{rank}}${{m.label}}</td>
      <td>${{s.total}}</td><td>${{s.perfect}}</td><td>${{s.acc}}%</td>
      <td>${{s.p}}%</td><td><strong>${{s.r}}%</strong></td><td>${{s.f1}}%</td>
      <td>${{s.tp}}</td><td style="color:#ef4444">${{s.fp}}</td><td style="color:#f59e0b">${{s.fn}}</td>
      <td>${{lat}}</td>
    </tr>`;
  }});
  document.getElementById('stats-body').innerHTML = html;
}}
function switchStats(d, btn) {{
  document.querySelectorAll('#stats-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderStats(d);
}}

// ── Confusion Matrix ──
function renderCM() {{
  const m = getModel(cmModel);
  if (!m) return;
  const cm = m.confusion[cmDataset];
  const total = cm.tp + cm.tn + cm.fp + cm.fn;
  const pct = v => total > 0 ? (v/total*100).toFixed(1)+'%' : '0%';
  const tpr = (cm.tp+cm.fn)>0 ? (cm.tp/(cm.tp+cm.fn)*100).toFixed(1) : '0.0';
  const tnr = (cm.tn+cm.fp)>0 ? (cm.tn/(cm.tn+cm.fp)*100).toFixed(1) : '0.0';
  document.getElementById('cm-area').innerHTML = `<div class="cm-box" style="max-width:500px">
    <h3>${{m.label}} - ${{cmDataset}}</h3>
    <div class="cm-grid">
      <div class="cm-corner">Actual \\\\ Pred</div>
      <div class="cm-header">Detected</div><div class="cm-header">Not Detected</div>
      <div class="cm-row-header">PII Exists</div>
      <div class="cm-cell tp">${{cm.tp}}<span class="pct">TP · 정확한 탐지 (${{pct(cm.tp)}})</span></div>
      <div class="cm-cell fn">${{cm.fn}}<span class="pct">FN · 미탐 (${{pct(cm.fn)}})</span></div>
      <div class="cm-row-header">No PII</div>
      <div class="cm-cell fp">${{cm.fp}}<span class="pct">FP · 오탐 (${{pct(cm.fp)}})</span></div>
      <div class="cm-cell tn">${{cm.tn}}<span class="pct">TN · 정확한 무시 (${{pct(cm.tn)}})</span></div>
    </div>
    <div style="margin-top:10px;font-size:12px;color:#475569;text-align:center;line-height:1.8;">
      <strong>Sensitivity:</strong> ${{tpr}}% &nbsp; <strong>Specificity:</strong> ${{tnr}}%
    </div>
  </div>`;
}}
function buildCMModelSelect() {{
  let html = '';
  MODELS.forEach(m => {{
    html += `<option value="${{m.id}}">${{m.label}} (${{m.size}})</option>`;
  }});
  document.getElementById('cm-model-select').innerHTML = html;
}}
function switchCMDataset(d, btn) {{
  cmDataset = d;
  document.querySelectorAll('#cm-dataset-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderCM();
}}

// ── Per-Category CM ──
function sensColor(val) {{
  // 0%=red, 50%=yellow, 100%=green
  if (val === null) return '#888';
  const h = val * 1.2; // 0→0(red), 100→120(green)
  return `hsl(${{h}}, 75%, 42%)`;
}}
const CAT_DESC = {{
  '이름': '문서에 등장하는 인명을 빠짐없이 검출하는지. 동명이인, 한자이름, 부수적 인물 포함.',
  '주소': '도로명/지번 등 물리적 위치 정보 검출. 지명과의 혼동 여부가 관건.',
  '주민등록번호': '13자리 전체 또는 부분 노출된 주민번호 검출. 가장 민감한 PII.',
  '여권번호': '여권번호(M/S + 8자리 등) 검출. 출현 빈도가 낮아 학습 데이터 부족 가능.',
  '운전면허번호': '운전면허번호(2자리-6자리-6자리-2자리) 검출. 지역코드 포함 형식.',
  '이메일': '표준 및 난독화된 이메일 주소 검출. 골뱅이/닷 등 한글 치환 포함.',
  'IP주소': 'IPv4/IPv6 주소 검출. 코드 내 리터럴 IP와 실제 PII 구분이 과제.',
  '전화번호': '휴대폰/유선/내선/국제 전화번호 검출. 대표번호 과탐 주의.',
  '계좌번호': '국내 은행 계좌 및 국제(IBAN) 계좌 검출. 형식 다양성이 도전 과제.',
  '카드번호': '신용/체크카드 번호 및 암호화폐 지갑 주소 검출.',
  '생년월일': '다양한 형식(YYYY.MM.DD, 00년생 등)의 생년월일 검출.',
  '기타_고유식별정보': '사번, 학번, 회원번호 등 문맥 의존적 식별자 검출.',
}};
function renderCatCM() {{
  const m = getModel(catcmModel);
  if (!m) return;
  const cc = m.cat_confusion[catcmDataset];
  let html = '';
  CATS.forEach(cat => {{
    const c = cc[cat] || {{tp:0,tn:0,fp:0,fn:0}};
    const sensVal = (c.tp+c.fn)>0 ? (c.tp/(c.tp+c.fn)*100) : null;
    const specVal = (c.tn+c.fp)>0 ? (c.tn/(c.tn+c.fp)*100) : null;
    const sens = sensVal !== null ? sensVal.toFixed(1)+'%' : '-';
    const spec = specVal !== null ? specVal.toFixed(1)+'%' : '-';
    const sensC = sensColor(sensVal);
    const specC = sensColor(specVal);
    const desc = CAT_DESC[cat] || '';
    html += `<tr><td>${{cat}}</td>
      <td style="color:#166534;font-weight:700">${{c.tp}}</td><td style="color:#4ade80">${{c.tn}}</td>
      <td style="color:#dc2626;font-weight:700">${{c.fp}}</td><td style="color:#ea580c;font-weight:700">${{c.fn}}</td>
      <td style="font-weight:700;color:${{sensC}}">${{sens}}</td><td style="font-weight:700;color:${{specC}}">${{spec}}</td>
      <td style="text-align:left;font-size:11px;color:#64748b;">${{desc}}</td></tr>`;
  }});
  document.getElementById('cat-cm-body').innerHTML = html;
}}
function buildCatCMSelect() {{
  let html = '';
  MODELS.forEach(m => {{
    html += `<option value="${{m.id}}">${{m.label}} (${{m.size}})</option>`;
  }});
  document.getElementById('catcm-model-select').innerHTML = html;
}}

// ── Case Browser ──
function buildModelFilter() {{
  let html = '';
  MODELS.forEach(m => {{
    html += `<option value="${{m.id}}">${{m.label}} (${{m.size}})</option>`;
  }});
  document.getElementById('filter-model-select').innerHTML = html;
}}

function getFilteredCases() {{
  let results = CASE_DATA[curModel] || [];
  if (curDataset === 'base') results = results.filter(r => r.difficulty === 'base');
  else if (curDataset === 'advanced') results = results.filter(r => r.difficulty === 'advanced');
  if (curType === 'success') results = results.filter(r => r.f1 === 1.0);
  else if (curType === 'fail') results = results.filter(r => r.f1 < 1.0);
  if (curPii === 'positive') results = results.filter(r => Object.keys(r.expected).length > 0);
  else if (curPii === 'negative') results = results.filter(r => Object.keys(r.expected).length === 0);
  return results;
}}

function renderCases() {{
  const results = getFilteredCases();
  const total = results.length;
  document.getElementById('case-count').textContent = total + ' cases';
  if (total === 0) {{
    document.getElementById('case-list').innerHTML = '<p style="color:#888;text-align:center;padding:20px;">No data</p>';
    document.getElementById('pagination').innerHTML = '';
    return;
  }}
  const start = curPage * PER_PAGE;
  const pageResults = results.slice(start, start + PER_PAGE);
  let html = '';
  pageResults.forEach(r => {{
    const f1class = r.f1 === 1.0 ? 'perfect' : r.f1 >= 0.7 ? 'good' : 'bad';
    let catHtml = '';
    const hasExp = Object.keys(r.expected).length > 0;
    const hasPred = Object.keys(r.predicted).length > 0;
    if (!hasExp && !hasPred) {{
      catHtml = `<div class="cat-row"><span class="cat-status ok">\\u2713</span><span class="cat-name">No PII</span><div class="cat-values"><div class="exp">Correct: No PII expected or detected</div></div></div>`;
    }} else if (!hasExp && hasPred) {{
      Object.keys(r.predicted).forEach(cat => {{
        catHtml += `<div class="cat-row"><span class="cat-status fp">FP</span><span class="cat-name">${{cat}}</span><div class="cat-values"><div class="missing">Expected: none</div><div class="pred">Predicted: ${{JSON.stringify(r.predicted[cat])}}</div></div></div>`;
      }});
    }} else {{
      CATS.forEach(cat => {{
        const e = r.expected[cat] || [];
        const p = r.predicted[cat] || [];
        if (e.length === 0 && p.length === 0) return;
        const match = e.length === p.length && e.every(v => new Set(p).has(v));
        let st, cls;
        if (match) {{ st = '\\u2713'; cls = 'ok'; }}
        else if (e.length > 0 && p.length === 0) {{ st = 'FN'; cls = 'fn'; }}
        else if (e.length === 0 && p.length > 0) {{ st = 'FP'; cls = 'fp'; }}
        else {{ st = '\\u2260'; cls = 'ne'; }}
        catHtml += `<div class="cat-row"><span class="cat-status ${{cls}}">${{st}}</span><span class="cat-name">${{cat}}</span><div class="cat-values">
          ${{e.length > 0 ? '<div class="exp">Expected: '+JSON.stringify(e)+'</div>' : ''}}
          ${{p.length > 0 ? '<div class="pred">Predicted: '+JSON.stringify(p)+'</div>' : ''}}
          ${{e.length === 0 && p.length > 0 ? '<div class="missing">Expected: none</div>' : ''}}
          ${{e.length > 0 && p.length === 0 ? '<div class="missing">Predicted: none</div>' : ''}}
        </div></div>`;
      }});
    }}
    const docText = (r.document_text||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    html += `<div class="case-card" onclick="this.classList.toggle('expanded')">
      <div class="case-top"><div class="case-meta"><span class="case-id">${{r.id}}</span><span class="badge cat">${{r.category}}</span></div><span class="case-f1 ${{f1class}}">F1: ${{(r.f1*100).toFixed(1)}}%</span></div>
      <div class="case-intent">${{r.intent||''}}</div>
      <div class="case-detail"><div class="doc-text">${{docText}}</div>${{catHtml}}</div>
    </div>`;
  }});
  document.getElementById('case-list').innerHTML = html;
  let pg = '';
  const totalPages = Math.ceil(total / PER_PAGE);
  for (let i = 0; i < totalPages; i++) pg += `<button class="page-btn ${{i===curPage?'active':''}}" onclick="curPage=${{i}};renderCases();window.scrollTo(0,document.getElementById('case-list').offsetTop-80)">${{i+1}}</button>`;
  document.getElementById('pagination').innerHTML = pg;
}}

// Filter event listeners
document.addEventListener('DOMContentLoaded', () => {{
  ['filter-dataset','filter-type','filter-pii'].forEach(gid => {{
    document.getElementById(gid).addEventListener('click', e => {{
      const btn = e.target.closest('.filter-btn');
      if (!btn) return;
      document.querySelectorAll('#'+gid+' .filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (gid === 'filter-dataset') curDataset = btn.dataset.val;
      else if (gid === 'filter-type') curType = btn.dataset.val;
      else if (gid === 'filter-pii') curPii = btn.dataset.val;
      curPage = 0;
      renderCases();
    }});
  }});
}});

// ── Case Study ──
function renderCaseStudy() {{
  // KPI
  const total = 300;
  const imperfect = CS_CASES.length;
  const perfect = total - imperfect;
  const f1Vals = CS_CASES.map(c => c.f1);
  const zeros = f1Vals.filter(v => v === 0).length;
  document.getElementById('cs-kpi').innerHTML = `
    <div class="cs-kpi-item"><div class="num" style="color:#10b981">${{perfect}}</div><div class="lbl">Perfect (F1=1.0)</div></div>
    <div class="cs-kpi-item"><div class="num" style="color:#f59e0b">${{imperfect}}</div><div class="lbl">Imperfect</div></div>
    <div class="cs-kpi-item"><div class="num" style="color:#ef4444">${{zeros}}</div><div class="lbl">F1 = 0</div></div>
    <div class="cs-kpi-item"><div class="num" style="color:#2563eb">${{(perfect/total*100).toFixed(1)}}%</div><div class="lbl">Accuracy</div></div>
  `;
  // Failure patterns table
  let fpHtml = '';
  FP_PATTERNS.forEach(p => {{
    fpHtml += `<tr><td style="font-weight:600">${{p.type}}</td><td>${{p.count}}</td><td style="font-family:monospace;font-size:12px">${{p.cases}}</td><td>${{p.desc}}</td></tr>`;
  }});
  document.getElementById('fp-body').innerHTML = fpHtml;
  // Case cards
  let html = '';
  CS_CASES.forEach(c => {{
    const f1pct = (c.f1 * 100).toFixed(1);
    const f1cls = c.f1 === 0 ? 'zero' : c.f1 < 0.5 ? 'low' : c.f1 < 0.9 ? 'mid' : 'high';
    const docText = (c.document_text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    let rowsHtml = '';
    c.rows.forEach(r => {{
      if (r.status === 'match') {{
        rowsHtml += `<tr><td>${{r.type}}</td><td class="match">${{esc(r.exp)}}</td><td class="match">${{esc(r.pred)}}</td><td class="match">✅</td></tr>`;
      }} else if (r.status === 'miss') {{
        rowsHtml += `<tr><td>${{r.type}}</td><td>${{esc(r.exp)}}</td><td class="miss"><em>(누락)</em></td><td class="miss">❌ Miss</td></tr>`;
      }} else {{
        rowsHtml += `<tr><td>${{r.type}}</td><td class="fp-mark"><em>(없음)</em></td><td class="fp-mark">${{esc(r.pred)}}</td><td class="fp-mark">❌ FP</td></tr>`;
      }}
    }});
    const analysisHtml = c.analysis ? `<div class="cs-analysis"><strong>분석:</strong> ${{esc(c.analysis)}}</div>` : '';
    html += `<div class="cs-card" id="cs-${{c.id}}">
      <div class="cs-header" onclick="this.parentElement.classList.toggle('open')">
        <div class="left">
          <span class="tc-id">${{c.id}}</span>
          <span class="badge cat">${{c.category}}</span>
          <span class="badge" style="background:#f1f5f9;color:#475569">${{c.difficulty}}</span>
        </div>
        <span class="tc-f1 ${{f1cls}}">F1: ${{f1pct}}%</span>
      </div>
      <div class="cs-body">
        <div style="font-size:13px;color:#666;margin-bottom:8px">${{esc(c.intent)}}</div>
        <div class="cs-doc">${{docText}}</div>
        <table class="cs-table"><thead><tr><th>Type</th><th>Expected</th><th>Predicted</th><th>Result</th></tr></thead><tbody>${{rowsHtml}}</tbody></table>
        ${{analysisHtml}}
      </div>
    </div>`;
  }});
  document.getElementById('cs-cases').innerHTML = html;
}}
function esc(s) {{ return s ? String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;') : ''; }}

// Init
renderSummary();
document.getElementById('bar-metric-desc').innerHTML = METRIC_DESC[barMetric] || '';
renderBarChart('combined');
renderStats('combined');
buildCMModelSelect(); renderCM();
buildCatCMSelect(); renderCatCM();
buildModelFilter(); renderCases();
renderCaseStudy();
</script>
</body>
</html>"""
    return html


def main() -> None:
    print("Loading model results...")
    all_models = load_all_models()
    print(f"  Loaded {len(all_models)} models")

    for fname, m in all_models.items():
        s = m["stats"]["combined"]
        lat = m["latency"]
        lat_str = f'{lat["mean_sec"]:.2f}s' if lat else "-"
        print(f"  {m['label']:<18s} F1={s['f1']}%  Perfect={s['perfect']}/300  Latency={lat_str}")

    print("\nGenerating HTML report...")
    html = build_html(all_models)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved to {OUTPUT}")
    print(f"  File size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
