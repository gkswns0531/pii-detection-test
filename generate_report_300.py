#!/usr/bin/env python3
"""HTML report generator for 11-model × 300-case PII benchmark.

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

  <!-- F1 Bar Chart -->
  <div class="section-title">F1 Score Comparison</div>
  <div class="tab-row" id="bar-tabs">
    <button class="tab-btn active" onclick="switchBar('combined',this)">Combined (300)</button>
    <button class="tab-btn" onclick="switchBar('base',this)">Base (200)</button>
    <button class="tab-btn" onclick="switchBar('advanced',this)">Advanced (100)</button>
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
  <div style="background:#fff;border-radius:12px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:16px;font-size:13px;color:#475569;line-height:1.7;">
    <p><strong>TP</strong> = PII exists & detected (good), <strong>TN</strong> = No PII & not detected (good), <strong>FP</strong> = No PII but detected (false alarm), <strong>FN</strong> = PII exists but missed (privacy risk)</p>
    <p><strong>Sensitivity</strong> = TP/(TP+FN), <strong>Specificity</strong> = TN/(TN+FP)</p>
  </div>
  <div class="tab-row" id="cm-model-tabs"></div>
  <div class="tab-row" id="cm-dataset-tabs" style="margin-left:12px;">
    <button class="tab-btn active" onclick="switchCMDataset('combined',this)">Combined</button>
    <button class="tab-btn" onclick="switchCMDataset('base',this)">Base</button>
    <button class="tab-btn" onclick="switchCMDataset('advanced',this)">Advanced</button>
  </div>
  <div id="cm-area" class="cm-container"></div>

  <!-- Per-Category CM -->
  <div class="section-title">Per-Category Confusion Matrix</div>
  <div class="tab-row" id="catcm-tabs"></div>
  <div class="stats-wrap">
  <table class="cat-cm-table" id="cat-cm-table">
    <thead><tr>
      <th style="text-align:left">Category</th>
      <th style="color:#166534">TP</th><th style="color:#4ade80">TN</th>
      <th style="color:#dc2626">FP</th><th style="color:#ea580c">FN</th>
      <th>Sensitivity</th><th>Specificity</th>
    </tr></thead>
    <tbody id="cat-cm-body"></tbody>
  </table>
  </div>

  <!-- Case Browser -->
  <div class="section-title">Case Browser</div>
  <div class="filters">
    <span class="filter-label">Model</span>
    <div class="filter-group" id="filter-model"></div>
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
</div>

<script>
const MODELS = {model_list_json};
const CASE_DATA = {case_data_json};
const CATS = {cats_json};
const PER_PAGE = 20;
let curModel = MODELS[0]?.id || '', curDataset = 'combined', curType = 'all', curPii = 'all', curPage = 0;
let cmModel = MODELS[0]?.id || '', cmDataset = 'combined';
let catcmModel = MODELS[0]?.id || '', catcmDataset = 'combined';
let barDataset = 'combined';

function getModel(id) {{ return MODELS.find(m => m.id === id); }}

// ── Summary Cards ──
function renderSummary() {{
  const best = MODELS.reduce((a, b) => (a.stats.combined.f1 > b.stats.combined.f1 ? a : b));
  const fastest = MODELS.filter(m => m.latency_mean).reduce((a, b) => (a.latency_mean < b.latency_mean ? a : b));
  const bestEfficiency = MODELS.filter(m => m.latency_mean && m.stats.combined.f1 > 70)
    .reduce((a, b) => ((a.stats.combined.f1 / a.latency_mean) > (b.stats.combined.f1 / b.latency_mean) ? a : b), MODELS[0]);
  document.getElementById('summary-cards').innerHTML = `
    <div class="card" style="border-left:4px solid #2563eb"><div class="label">Best F1</div><div class="value" style="color:#2563eb">${{best.stats.combined.f1}}%</div><div class="sub">${{best.label}}</div></div>
    <div class="card" style="border-left:4px solid #10b981"><div class="label">Best Perfect</div><div class="value" style="color:#10b981">${{best.stats.combined.perfect}}/300</div><div class="sub">${{best.label}}</div></div>
    <div class="card" style="border-left:4px solid #f59e0b"><div class="label">Fastest Latency</div><div class="value" style="color:#f59e0b">${{fastest.latency_mean?.toFixed(2)}}s</div><div class="sub">${{fastest.label}}</div></div>
    <div class="card" style="border-left:4px solid #8b5cf6"><div class="label">Best Efficiency</div><div class="value" style="color:#8b5cf6">${{bestEfficiency.stats.combined.f1}}%</div><div class="sub">${{bestEfficiency.label}} (${{bestEfficiency.latency_mean?.toFixed(2)}}s)</div></div>
    <div class="card" style="border-left:4px solid #64748b"><div class="label">Models Tested</div><div class="value" style="color:#64748b">${{MODELS.length}}</div><div class="sub">300 cases each</div></div>
  `;
}}

// ── Bar Chart ──
function renderBarChart(dataset) {{
  const sorted = [...MODELS].sort((a, b) => b.stats[dataset].f1 - a.stats[dataset].f1);
  let html = '';
  sorted.forEach((m, i) => {{
    const s = m.stats[dataset];
    html += `<div class="bar-row">
      <div class="bar-label">${{m.label}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{s.f1}}%;background:${{m.color}}">${{s.f1}}%</div></div>
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

// ── Stats Table ──
function renderStats(dataset) {{
  const sorted = [...MODELS].sort((a, b) => b.stats[dataset].f1 - a.stats[dataset].f1);
  let html = '';
  sorted.forEach((m, i) => {{
    const s = m.stats[dataset];
    const rank = i < 3 ? `<span class="rank r${{i+1}}">${{i+1}}</span>` : '';
    const cls = i === 0 ? ' class="best"' : '';
    const lat = m.latency_mean ? m.latency_mean.toFixed(2) + 's' : '-';
    html += `<tr${{cls}}>
      <td>${{rank}}${{m.label}}</td>
      <td>${{s.total}}</td><td>${{s.perfect}}</td><td>${{s.acc}}%</td>
      <td>${{s.p}}%</td><td>${{s.r}}%</td><td><strong>${{s.f1}}%</strong></td>
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
      <div class="cm-cell tp">${{cm.tp}}<span class="pct">TP (${{pct(cm.tp)}})</span></div>
      <div class="cm-cell fn">${{cm.fn}}<span class="pct">FN (${{pct(cm.fn)}})</span></div>
      <div class="cm-row-header">No PII</div>
      <div class="cm-cell fp">${{cm.fp}}<span class="pct">FP (${{pct(cm.fp)}})</span></div>
      <div class="cm-cell tn">${{cm.tn}}<span class="pct">TN (${{pct(cm.tn)}})</span></div>
    </div>
    <div style="margin-top:10px;font-size:12px;color:#475569;text-align:center;line-height:1.8;">
      <strong>Sensitivity:</strong> ${{tpr}}% &nbsp; <strong>Specificity:</strong> ${{tnr}}%
    </div>
  </div>`;
}}
function buildCMModelTabs() {{
  let html = '';
  MODELS.forEach((m, i) => {{
    html += `<button class="tab-btn ${{i===0?'active':''}}" onclick="cmModel='${{m.id}}';document.querySelectorAll('#cm-model-tabs .tab-btn').forEach(b=>b.classList.remove('active'));this.classList.add('active');renderCM();">${{m.label}}</button>`;
  }});
  document.getElementById('cm-model-tabs').innerHTML = html;
}}
function switchCMDataset(d, btn) {{
  cmDataset = d;
  document.querySelectorAll('#cm-dataset-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderCM();
}}

// ── Per-Category CM ──
function renderCatCM() {{
  const m = getModel(catcmModel);
  if (!m) return;
  const cc = m.cat_confusion[catcmDataset];
  let html = '';
  CATS.forEach(cat => {{
    const c = cc[cat] || {{tp:0,tn:0,fp:0,fn:0}};
    const sens = (c.tp+c.fn)>0 ? ((c.tp/(c.tp+c.fn))*100).toFixed(1)+'%' : '-';
    const spec = (c.tn+c.fp)>0 ? ((c.tn/(c.tn+c.fp))*100).toFixed(1)+'%' : '-';
    html += `<tr><td>${{cat}}</td>
      <td style="color:#166534;font-weight:700">${{c.tp}}</td><td style="color:#4ade80">${{c.tn}}</td>
      <td style="color:#dc2626;font-weight:700">${{c.fp}}</td><td style="color:#ea580c;font-weight:700">${{c.fn}}</td>
      <td>${{sens}}</td><td>${{spec}}</td></tr>`;
  }});
  document.getElementById('cat-cm-body').innerHTML = html;
}}
function buildCatCMTabs() {{
  let html = '';
  MODELS.forEach((m, i) => {{
    html += `<button class="tab-btn ${{i===0?'active':''}}" onclick="catcmModel='${{m.id}}';document.querySelectorAll('#catcm-tabs .tab-btn').forEach(b=>b.classList.remove('active'));this.classList.add('active');renderCatCM();">${{m.label}}</button>`;
  }});
  document.getElementById('catcm-tabs').innerHTML = html;
}}

// ── Case Browser ──
function buildModelFilter() {{
  let html = '';
  MODELS.forEach((m, i) => {{
    html += `<button class="filter-btn ${{i===0?'active':''}}" data-val="${{m.id}}">${{m.label}}</button>`;
  }});
  document.getElementById('filter-model').innerHTML = html;
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
  ['filter-model','filter-dataset','filter-type','filter-pii'].forEach(gid => {{
    document.getElementById(gid).addEventListener('click', e => {{
      const btn = e.target.closest('.filter-btn');
      if (!btn) return;
      document.querySelectorAll('#'+gid+' .filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (gid === 'filter-model') curModel = btn.dataset.val;
      else if (gid === 'filter-dataset') curDataset = btn.dataset.val;
      else if (gid === 'filter-type') curType = btn.dataset.val;
      else if (gid === 'filter-pii') curPii = btn.dataset.val;
      curPage = 0;
      renderCases();
    }});
  }});
}});

// Init
renderSummary();
renderBarChart('combined');
renderStats('combined');
buildCMModelTabs(); renderCM();
buildCatCMTabs(); renderCatCM();
buildModelFilter(); renderCases();
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
