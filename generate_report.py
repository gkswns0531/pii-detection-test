#!/usr/bin/env python3
"""PM-friendly HTML report generator for PII Detection Benchmark.

Reads split JSON files and generates an interactive HTML report with:
- Summary cards
- Statistics table with bar chart visualization
- Confusion matrix (entity-level & category-level)
- Filterable case browser
"""

import json
import os
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "benchmark_results" / "splits"
OUTPUT = Path(__file__).parent / "benchmark_results" / "report.html"

CATS = [
    "이름", "주소", "주민등록번호", "여권번호", "운전면허번호",
    "이메일", "IP주소", "전화번호", "계좌번호", "카드번호",
    "생년월일", "기타_고유식별정보",
]

PROMPTS = ["full", "vanilla"]
DATASETS = ["base", "advanced", "combined"]
TYPES = ["all", "success", "fail"]

PROMPT_LABELS = {"full": "After Optimization", "vanilla": "Before Optimization"}


def load_splits() -> dict:
    """Load all 18 split JSON files."""
    data = {}
    for p in PROMPTS:
        for d in DATASETS:
            for t in TYPES:
                fname = f"{p}_{d}_{t}.json"
                fpath = SPLITS_DIR / fname
                if fpath.exists():
                    with open(fpath, encoding="utf-8") as f:
                        data[f"{p}_{d}_{t}"] = json.load(f)
    return data


def compute_stats(results: list[dict]) -> dict:
    """Compute precision/recall/F1/accuracy from result list."""
    total = len(results)
    perfect = sum(1 for r in results if r["f1"] == 1.0)
    tp = fp = fn = 0
    for r in results:
        exp = r["expected"]
        pred = r["predicted"]
        all_cats = set(list(exp.keys()) + list(pred.keys()))
        for cat in all_cats:
            e_set = set(exp.get(cat, []))
            p_set = set(pred.get(cat, []))
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
    """Compute confusion matrix at document-category level.

    For each (document, category):
    - TP: expected has items AND predicted has items (at least partial overlap)
    - TN: expected empty AND predicted empty
    - FP: expected empty BUT predicted has items
    - FN: expected has items BUT predicted empty
    """
    tp = tn = fp = fn = 0
    for r in results:
        exp = r["expected"]
        pred = r["predicted"]
        for cat in CATS:
            e = exp.get(cat, [])
            p = pred.get(cat, [])
            has_e = len(e) > 0
            has_p = len(p) > 0
            if has_e and has_p:
                tp += 1
            elif not has_e and not has_p:
                tn += 1
            elif not has_e and has_p:
                fp += 1
            else:  # has_e and not has_p
                fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_category_confusion(results: list[dict]) -> dict[str, dict]:
    """Per-category confusion matrix."""
    cm = {cat: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for cat in CATS}
    for r in results:
        exp = r["expected"]
        pred = r["predicted"]
        for cat in CATS:
            e = exp.get(cat, [])
            p = pred.get(cat, [])
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


def build_html(all_data: dict) -> str:
    """Build the complete HTML report."""
    # Compute stats for all splits
    stats = {}
    for key, split_data in all_data.items():
        stats[key] = compute_stats(split_data["results"])

    # Compute confusion matrices for the 6 "all" splits
    confusion = {}
    cat_confusion = {}
    for p in PROMPTS:
        for d in DATASETS:
            key = f"{p}_{d}_all"
            if key in all_data:
                confusion[key] = compute_confusion_matrix(all_data[key]["results"])
                cat_confusion[key] = compute_category_confusion(all_data[key]["results"])

    # Serialize data for JS (only the results arrays)
    js_data = {}
    for key, split_data in all_data.items():
        js_data[key] = split_data["results"]

    stats_json = json.dumps(stats, ensure_ascii=False)
    data_json = json.dumps(js_data, ensure_ascii=False)
    confusion_json = json.dumps(confusion, ensure_ascii=False)
    cat_confusion_json = json.dumps(cat_confusion, ensure_ascii=False)
    cats_json = json.dumps(CATS, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PII Detection Benchmark Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
.container {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
.subtitle {{ color: #666; font-size: 14px; margin-bottom: 24px; }}

/* Summary Cards */
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 28px; }}
.card {{ background: #fff; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }}
.card .label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
.card .value {{ font-size: 32px; font-weight: 800; margin: 4px 0; }}
.card .sub {{ font-size: 12px; color: #aaa; }}
.card.green .value {{ color: #10b981; }}
.card.blue .value {{ color: #3b82f6; }}
.card.orange .value {{ color: #f59e0b; }}
.card.red .value {{ color: #ef4444; }}
.card.purple .value {{ color: #8b5cf6; }}

/* Section */
.section-title {{ font-size: 18px; font-weight: 700; margin: 32px 0 14px; display: flex; align-items: center; gap: 8px; }}
.section-title .icon {{ font-size: 20px; }}

/* Stats Table */
.stats-wrap {{ overflow-x: auto; margin-bottom: 28px; }}
.stats-table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.stats-table th {{ background: #f8fafc; padding: 10px 12px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; font-weight: 600; text-align: right; border-bottom: 2px solid #e2e8f0; white-space: nowrap; }}
.stats-table th:first-child, .stats-table th:nth-child(2), .stats-table th:nth-child(3) {{ text-align: left; }}
.stats-table td {{ padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #f1f5f9; text-align: right; white-space: nowrap; }}
.stats-table td:first-child, .stats-table td:nth-child(2), .stats-table td:nth-child(3) {{ text-align: left; font-weight: 500; }}
.stats-table tr:hover {{ background: #f8fafc; }}
.stats-table tr.highlight {{ background: #eff6ff; }}

/* Badges */
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.badge.full {{ background: #dbeafe; color: #1d4ed8; }}
.badge.vanilla {{ background: #fef3c7; color: #92400e; }}
.badge.base {{ background: #d1fae5; color: #065f46; }}
.badge.advanced {{ background: #fce7f3; color: #9d174d; }}
.badge.combined {{ background: #e0e7ff; color: #3730a3; }}

/* Bar Charts */
.bar-chart {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 28px; }}
.bar-row {{ display: flex; align-items: center; gap: 10px; }}
.bar-label {{ width: 180px; font-size: 13px; font-weight: 600; text-align: right; flex-shrink: 0; }}
.bar-track {{ flex: 1; background: #e5e7eb; border-radius: 8px; height: 28px; position: relative; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 8px; display: flex; align-items: center; padding: 0 10px; font-size: 12px; font-weight: 700; color: #fff; transition: width 0.6s ease; min-width: 40px; }}
.bar-fill.after {{ background: linear-gradient(135deg, #3b82f6, #2563eb); }}
.bar-fill.before {{ background: linear-gradient(135deg, #f59e0b, #d97706); }}
.bar-pair {{ display: flex; flex-direction: column; gap: 3px; }}

/* Confusion Matrix */
.cm-container {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 28px; }}
.cm-box {{ background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); flex: 1; min-width: 340px; }}
.cm-box h3 {{ font-size: 15px; font-weight: 700; margin-bottom: 14px; text-align: center; }}
.cm-grid {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 0; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }}
.cm-corner {{ background: #f8fafc; padding: 10px 14px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; }}
.cm-header {{ background: #f8fafc; padding: 10px 14px; text-align: center; font-size: 12px; font-weight: 700; color: #475569; border-bottom: 2px solid #e2e8f0; }}
.cm-row-header {{ background: #f8fafc; padding: 10px 14px; font-size: 12px; font-weight: 700; color: #475569; display: flex; align-items: center; border-right: 2px solid #e2e8f0; }}
.cm-cell {{ padding: 16px 14px; text-align: center; font-size: 22px; font-weight: 800; }}
.cm-cell .pct {{ font-size: 11px; font-weight: 500; color: #64748b; display: block; margin-top: 2px; }}
.cm-cell.tp {{ background: #dcfce7; color: #166534; }}
.cm-cell.tn {{ background: #f0fdf4; color: #166534; }}
.cm-cell.fp {{ background: #fef2f2; color: #991b1b; }}
.cm-cell.fn {{ background: #fff7ed; color: #9a3412; }}

/* Category CM Table */
.cat-cm-table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 28px; }}
.cat-cm-table th {{ background: #f8fafc; padding: 8px 10px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; color: #64748b; font-weight: 600; border-bottom: 2px solid #e2e8f0; text-align: center; }}
.cat-cm-table th:first-child {{ text-align: left; }}
.cat-cm-table td {{ padding: 8px 10px; font-size: 13px; border-bottom: 1px solid #f1f5f9; text-align: center; }}
.cat-cm-table td:first-child {{ text-align: left; font-weight: 600; }}
.cat-cm-table tr:hover {{ background: #f8fafc; }}
.cat-cm-table .val-tp {{ color: #166534; font-weight: 700; }}
.cat-cm-table .val-tn {{ color: #4ade80; }}
.cat-cm-table .val-fp {{ color: #dc2626; font-weight: 700; }}
.cat-cm-table .val-fn {{ color: #ea580c; font-weight: 700; }}
.mini-bar {{ display: inline-block; height: 6px; border-radius: 3px; vertical-align: middle; margin-left: 4px; }}

/* Filters */
.filters {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
.filter-group {{ display: flex; gap: 4px; background: #fff; border-radius: 8px; padding: 3px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.filter-btn {{ padding: 6px 14px; border: none; background: transparent; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; color: #666; transition: all 0.15s; }}
.filter-btn.active {{ background: #3b82f6; color: #fff; }}
.filter-btn:hover:not(.active) {{ background: #f0f0f0; }}
.filter-label {{ font-size: 12px; color: #888; font-weight: 600; text-transform: uppercase; margin-right: 4px; align-self: center; }}

/* Case List */
.case-list {{ margin-bottom: 24px; }}
.case-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
.case-header h2 {{ font-size: 18px; }}
.case-count {{ font-size: 13px; color: #888; }}
.case-card {{ background: #fff; border-radius: 10px; padding: 16px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); cursor: pointer; transition: all 0.15s; }}
.case-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
.case-card.expanded {{ box-shadow: 0 2px 12px rgba(0,0,0,0.15); }}
.case-top {{ display: flex; justify-content: space-between; align-items: center; }}
.case-meta {{ display: flex; gap: 6px; align-items: center; }}
.case-id {{ font-weight: 700; font-size: 14px; margin-right: 8px; }}
.case-f1 {{ font-weight: 700; font-size: 14px; }}
.case-f1.perfect {{ color: #10b981; }}
.case-f1.good {{ color: #3b82f6; }}
.case-f1.bad {{ color: #ef4444; }}
.case-intent {{ font-size: 13px; color: #666; margin-top: 6px; }}
.case-detail {{ display: none; margin-top: 14px; border-top: 1px solid #f1f5f9; padding-top: 14px; }}
.case-card.expanded .case-detail {{ display: block; }}
.doc-text {{ background: #f8fafc; padding: 12px; border-radius: 8px; font-size: 12px; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; margin-bottom: 12px; border: 1px solid #e2e8f0; }}
.cat-row {{ display: flex; align-items: flex-start; padding: 6px 0; border-bottom: 1px solid #f8fafc; font-size: 13px; }}
.cat-status {{ width: 36px; text-align: center; font-weight: 700; flex-shrink: 0; }}
.cat-status.ok {{ color: #10b981; }}
.cat-status.fp {{ color: #ef4444; }}
.cat-status.fn {{ color: #f59e0b; }}
.cat-status.ne {{ color: #ef4444; }}
.cat-name {{ width: 100px; font-weight: 600; flex-shrink: 0; color: #374151; }}
.cat-values {{ flex: 1; }}
.cat-values .exp {{ color: #059669; font-size: 12px; }}
.cat-values .pred {{ color: #2563eb; font-size: 12px; }}
.cat-values .missing {{ color: #dc2626; font-size: 12px; }}
.pagination {{ display: flex; justify-content: center; gap: 6px; margin-top: 12px; }}
.page-btn {{ padding: 6px 12px; border: 1px solid #e2e8f0; background: #fff; border-radius: 6px; cursor: pointer; font-size: 13px; }}
.page-btn.active {{ background: #3b82f6; color: #fff; border-color: #3b82f6; }}
.page-btn:hover:not(.active) {{ background: #f0f0f0; }}

/* Tabs */
.tab-row {{ display: flex; gap: 4px; margin-bottom: 16px; background: #fff; border-radius: 8px; padding: 3px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); display: inline-flex; }}
.tab-btn {{ padding: 8px 18px; border: none; background: transparent; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; color: #666; transition: all 0.15s; }}
.tab-btn.active {{ background: #3b82f6; color: #fff; }}
.tab-btn:hover:not(.active) {{ background: #f0f0f0; }}

/* Responsive */
@media (max-width: 768px) {{
  .cm-container {{ flex-direction: column; }}
  .bar-label {{ width: 120px; font-size: 12px; }}
}}
</style>
</head>
<body>
<div class="container">
  <h1>PII Detection Benchmark Report</h1>
  <p class="subtitle">Qwen3-30B-A3B-Instruct-2507-FP8 &middot; 300 Test Cases &middot; Before vs After Optimization</p>

  <!-- Summary Cards -->
  <div id="summary-cards" class="summary-grid"></div>

  <!-- F1 Bar Chart Comparison -->
  <div class="section-title">Performance Comparison</div>
  <div id="bar-chart" class="bar-chart"></div>

  <!-- Statistics Table -->
  <div class="section-title">Detailed Statistics</div>
  <div class="stats-wrap">
  <table class="stats-table" id="stats-table">
    <thead>
      <tr>
        <th>Prompt</th><th>Dataset</th><th>Type</th>
        <th>Cases</th><th>Perfect</th><th>Accuracy</th>
        <th>Precision</th><th>Recall</th><th>F1</th>
        <th>FP</th><th>FN</th>
      </tr>
    </thead>
    <tbody id="stats-body"></tbody>
  </table>
  </div>

  <!-- Confusion Matrix Section -->
  <div class="section-title">Confusion Matrix (Document-Category Level)</div>
  <div style="background:#fff;border-radius:12px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:18px;font-size:13px;color:#475569;line-height:1.8;">
    <p style="margin-bottom:10px;font-weight:600;color:#1a1a2e;">
      How to read this matrix
    </p>
    <p style="margin-bottom:8px;">
      Each document is evaluated across 12 PII categories, producing <strong>document &times; 12</strong> cells total.
      For each cell, we check: "Does this category actually contain PII?" vs "Did the model detect PII in this category?"
    </p>
    <table style="border-collapse:collapse;margin:10px 0 14px;font-size:12px;width:100%;max-width:700px;">
      <tr>
        <td style="padding:6px 12px;background:#dcfce7;border:1px solid #e2e8f0;font-weight:700;color:#166534;width:60px;">TP</td>
        <td style="padding:6px 12px;border:1px solid #e2e8f0;"><strong>True Positive</strong> &mdash; PII actually exists, and the model correctly detected it. <span style="color:#166534;font-weight:600;">Higher is better.</span></td>
      </tr>
      <tr>
        <td style="padding:6px 12px;background:#f0fdf4;border:1px solid #e2e8f0;font-weight:700;color:#166534;">TN</td>
        <td style="padding:6px 12px;border:1px solid #e2e8f0;"><strong>True Negative</strong> &mdash; No PII exists, and the model correctly ignored it. <span style="color:#166534;font-weight:600;">Higher is better.</span></td>
      </tr>
      <tr>
        <td style="padding:6px 12px;background:#fef2f2;border:1px solid #e2e8f0;font-weight:700;color:#991b1b;">FP</td>
        <td style="padding:6px 12px;border:1px solid #e2e8f0;"><strong>False Positive</strong> &mdash; No PII exists, but the model incorrectly flagged it as PII. <span style="color:#dc2626;font-weight:600;">Lower is better.</span> Users get unnecessary masking/alerts.</td>
      </tr>
      <tr>
        <td style="padding:6px 12px;background:#fff7ed;border:1px solid #e2e8f0;font-weight:700;color:#9a3412;">FN</td>
        <td style="padding:6px 12px;border:1px solid #e2e8f0;"><strong>False Negative</strong> &mdash; PII actually exists, but the model missed it. <span style="color:#ea580c;font-weight:600;">Lower is better.</span> This is a privacy risk &mdash; sensitive data goes undetected.</td>
      </tr>
    </table>
    <p style="margin-bottom:6px;">
      <strong>Sensitivity (TPR)</strong> = TP / (TP + FN) &mdash; "Of all real PII, how much did the model catch?" Higher = fewer privacy leaks.
    </p>
    <p style="margin-bottom:0;">
      <strong>Specificity (TNR)</strong> = TN / (TN + FP) &mdash; "Of all non-PII, how much did the model correctly leave alone?" Higher = fewer false alarms.
    </p>
  </div>

  <div class="tab-row" id="cm-tabs">
    <button class="tab-btn active" data-dataset="combined" onclick="switchCM('combined',this)">Combined (300)</button>
    <button class="tab-btn" data-dataset="base" onclick="switchCM('base',this)">Base (200)</button>
    <button class="tab-btn" data-dataset="advanced" onclick="switchCM('advanced',this)">Advanced (100)</button>
  </div>
  <div id="cm-area" class="cm-container"></div>

  <!-- Per-Category Confusion Matrix -->
  <div class="section-title">Per-Category Confusion Matrix</div>
  <div style="background:#fff;border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:18px;font-size:13px;color:#475569;line-height:1.7;">
    <p>
      Below shows TP/TN/FP/FN <strong>per PII category</strong> (e.g. Name, Address, SSN...).
      Categories with high <span style="color:#dc2626;font-weight:600;">FP</span> are being over-detected (false alarms).
      Categories with high <span style="color:#ea580c;font-weight:600;">FN</span> are being missed (privacy risk).
      Ideally: high TP &amp; TN, zero FP &amp; FN.
    </p>
  </div>
  <div class="tab-row" id="cat-cm-tabs">
    <button class="tab-btn active" data-dataset="combined" onclick="switchCatCM('combined',this)">Combined</button>
    <button class="tab-btn" data-dataset="base" onclick="switchCatCM('base',this)">Base</button>
    <button class="tab-btn" data-dataset="advanced" onclick="switchCatCM('advanced',this)">Advanced</button>
  </div>
  <div class="stats-wrap">
  <table class="cat-cm-table" id="cat-cm-table">
    <thead>
      <tr>
        <th>Category</th>
        <th colspan="4">After Optimization</th>
        <th colspan="4">Before Optimization</th>
      </tr>
      <tr>
        <th></th>
        <th style="color:#166534">TP</th><th style="color:#4ade80">TN</th>
        <th style="color:#dc2626">FP</th><th style="color:#ea580c">FN</th>
        <th style="color:#166534">TP</th><th style="color:#4ade80">TN</th>
        <th style="color:#dc2626">FP</th><th style="color:#ea580c">FN</th>
      </tr>
    </thead>
    <tbody id="cat-cm-body"></tbody>
  </table>
  </div>

  <!-- Case Browser -->
  <div class="section-title">Case Browser</div>
  <div class="filters">
    <span class="filter-label">Prompt</span>
    <div class="filter-group" id="filter-prompt">
      <button class="filter-btn active" data-val="full">After Opt.</button>
      <button class="filter-btn" data-val="vanilla">Before Opt.</button>
    </div>
    <span class="filter-label">Dataset</span>
    <div class="filter-group" id="filter-dataset">
      <button class="filter-btn active" data-val="base">Base</button>
      <button class="filter-btn" data-val="advanced">Advanced</button>
      <button class="filter-btn" data-val="combined">Combined</button>
    </div>
    <span class="filter-label">Result</span>
    <div class="filter-group" id="filter-type">
      <button class="filter-btn active" data-val="all">All</button>
      <button class="filter-btn" data-val="success">Success</button>
      <button class="filter-btn" data-val="fail">Fail</button>
    </div>
  </div>
  <div class="case-header">
    <h2 id="case-title"></h2>
    <span class="case-count" id="case-count"></span>
  </div>
  <div id="case-list" class="case-list"></div>
  <div id="pagination" class="pagination"></div>
</div>

<script>
const STATS = {stats_json};
const DATA = {data_json};
const CONFUSION = {confusion_json};
const CAT_CONFUSION = {cat_confusion_json};
const CATS = {cats_json};
const PROMPT_LABELS = {{"full": "After Optimization", "vanilla": "Before Optimization"}};

const PER_PAGE = 20;
let currentPrompt = 'full', currentDataset = 'base', currentType = 'all', currentPage = 0;

function getKey() {{ return currentPrompt + '_' + currentDataset + '_' + currentType; }}

// ── Summary Cards ──
function renderSummaryCards() {{
  const after = STATS['full_combined_all'];
  const before = STATS['vanilla_combined_all'];
  const delta = (after.f1 - before.f1).toFixed(1);
  document.getElementById('summary-cards').innerHTML = `
    <div class="card green"><div class="label">After Opt. F1</div><div class="value">${{after.f1}}%</div><div class="sub">${{after.perfect}}/${{after.total}} perfect</div></div>
    <div class="card orange"><div class="label">Before Opt. F1</div><div class="value">${{before.f1}}%</div><div class="sub">${{before.perfect}}/${{before.total}} perfect</div></div>
    <div class="card blue"><div class="label">Optimization Effect</div><div class="value">+${{delta}}%p</div><div class="sub">F1 improvement</div></div>
    <div class="card purple"><div class="label">After Precision</div><div class="value">${{after.p}}%</div><div class="sub">FP: ${{after.fp}}</div></div>
    <div class="card red"><div class="label">Before FP Count</div><div class="value">${{before.fp}}</div><div class="sub">vs After: ${{after.fp}}</div></div>
  `;
}}

// ── Bar Chart ──
function renderBarChart() {{
  const datasets = [
    ['Base (200)', 'base'], ['Advanced (100)', 'advanced'], ['Combined (300)', 'combined']
  ];
  const metrics = [['F1', 'f1'], ['Precision', 'p'], ['Recall', 'r']];
  let html = '';
  metrics.forEach(([metricLabel, metricKey]) => {{
    html += `<div style="font-size:13px;font-weight:700;color:#475569;margin:12px 0 6px;">${{metricLabel}}</div>`;
    datasets.forEach(([label, dkey]) => {{
      const after = STATS['full_'+dkey+'_all'];
      const before = STATS['vanilla_'+dkey+'_all'];
      const aVal = after[metricKey];
      const bVal = before[metricKey];
      html += `<div class="bar-row">
        <div class="bar-label">${{label}}</div>
        <div class="bar-pair" style="flex:1">
          <div class="bar-track"><div class="bar-fill after" style="width:${{aVal}}%">After ${{aVal}}%</div></div>
          <div class="bar-track"><div class="bar-fill before" style="width:${{bVal}}%">Before ${{bVal}}%</div></div>
        </div>
      </div>`;
    }});
  }});
  document.getElementById('bar-chart').innerHTML = html;
}}

// ── Statistics Table ──
function renderStatsTable() {{
  const rows = [
    ['full','base','all'],['full','base','fail'],
    ['full','advanced','all'],['full','advanced','fail'],
    ['full','combined','all'],['full','combined','fail'],
    ['vanilla','base','all'],['vanilla','base','fail'],
    ['vanilla','advanced','all'],['vanilla','advanced','fail'],
    ['vanilla','combined','all'],['vanilla','combined','fail'],
  ];
  let html = '';
  rows.forEach(([p, d, t]) => {{
    const key = p+'_'+d+'_'+t;
    const s = STATS[key];
    if (!s) return;
    const hl = t === 'all' ? ' class="highlight"' : '';
    const pLabel = PROMPT_LABELS[p] || p;
    html += `<tr${{hl}}>
      <td><span class="badge ${{p}}">${{pLabel}}</span></td>
      <td><span class="badge ${{d}}">${{d}}</span></td>
      <td>${{t}}</td>
      <td>${{s.total}}</td><td>${{s.perfect}}</td>
      <td>${{s.acc}}%</td><td>${{s.p}}%</td><td>${{s.r}}%</td>
      <td><strong>${{s.f1}}%</strong></td>
      <td style="color:#ef4444">${{s.fp}}</td><td style="color:#f59e0b">${{s.fn}}</td>
    </tr>`;
  }});
  document.getElementById('stats-body').innerHTML = html;
}}

// ── Confusion Matrix ──
function renderCM(dataset) {{
  const afterKey = 'full_'+dataset+'_all';
  const beforeKey = 'vanilla_'+dataset+'_all';
  const aCM = CONFUSION[afterKey] || {{tp:0,tn:0,fp:0,fn:0}};
  const bCM = CONFUSION[beforeKey] || {{tp:0,tn:0,fp:0,fn:0}};

  function cmBox(title, cm) {{
    const total = cm.tp + cm.tn + cm.fp + cm.fn;
    const pct = v => total > 0 ? (v/total*100).toFixed(1)+'%' : '0%';
    const tpr = (cm.tp+cm.fn)>0 ? (cm.tp/(cm.tp+cm.fn)*100).toFixed(1) : '0.0';
    const tnr = (cm.tn+cm.fp)>0 ? (cm.tn/(cm.tn+cm.fp)*100).toFixed(1) : '0.0';
    return `<div class="cm-box">
      <h3>${{title}}</h3>
      <div class="cm-grid">
        <div class="cm-corner">Actual \\\\ Pred</div>
        <div class="cm-header">Detected<br><span style="font-size:10px;font-weight:400;color:#94a3b8;">model said PII</span></div>
        <div class="cm-header">Not Detected<br><span style="font-size:10px;font-weight:400;color:#94a3b8;">model said no PII</span></div>
        <div class="cm-row-header">PII Exists<br><span style="font-size:10px;font-weight:400;color:#94a3b8;">actual PII present</span></div>
        <div class="cm-cell tp">${{cm.tp}}<span class="pct">TP (${{pct(cm.tp)}})</span></div>
        <div class="cm-cell fn">${{cm.fn}}<span class="pct">FN (${{pct(cm.fn)}})</span></div>
        <div class="cm-row-header">No PII<br><span style="font-size:10px;font-weight:400;color:#94a3b8;">no actual PII</span></div>
        <div class="cm-cell fp">${{cm.fp}}<span class="pct">FP (${{pct(cm.fp)}})</span></div>
        <div class="cm-cell tn">${{cm.tn}}<span class="pct">TN (${{pct(cm.tn)}})</span></div>
      </div>
      <div style="margin-top:12px;font-size:12px;color:#475569;text-align:center;line-height:1.8;">
        <strong>Sensitivity(TPR):</strong> ${{tpr}}% <span style="color:#94a3b8;">&mdash; catches ${{tpr}}% of real PII</span><br>
        <strong>Specificity(TNR):</strong> ${{tnr}}% <span style="color:#94a3b8;">&mdash; correctly ignores ${{tnr}}% of non-PII</span>
      </div>
    </div>`;
  }}

  document.getElementById('cm-area').innerHTML =
    cmBox('After Optimization', aCM) + cmBox('Before Optimization', bCM);
}}

function switchCM(dataset, btn) {{
  document.querySelectorAll('#cm-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderCM(dataset);
}}

// ── Per-Category Confusion Matrix ──
function renderCatCM(dataset) {{
  const afterKey = 'full_'+dataset+'_all';
  const beforeKey = 'vanilla_'+dataset+'_all';
  const aCM = CAT_CONFUSION[afterKey] || {{}};
  const bCM = CAT_CONFUSION[beforeKey] || {{}};

  let html = '';
  CATS.forEach(cat => {{
    const a = aCM[cat] || {{tp:0,tn:0,fp:0,fn:0}};
    const b = bCM[cat] || {{tp:0,tn:0,fp:0,fn:0}};
    html += `<tr>
      <td>${{cat}}</td>
      <td class="val-tp">${{a.tp}}</td><td class="val-tn">${{a.tn}}</td>
      <td class="val-fp">${{a.fp}}</td><td class="val-fn">${{a.fn}}</td>
      <td class="val-tp">${{b.tp}}</td><td class="val-tn">${{b.tn}}</td>
      <td class="val-fp">${{b.fp}}</td><td class="val-fn">${{b.fn}}</td>
    </tr>`;
  }});
  document.getElementById('cat-cm-body').innerHTML = html;
}}

function switchCatCM(dataset, btn) {{
  document.querySelectorAll('#cat-cm-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderCatCM(dataset);
}}

// ── Case Browser ──
function renderCases() {{
  const key = getKey();
  const results = DATA[key];
  if (!results || results.length === 0) {{
    document.getElementById('case-list').innerHTML = '<p style="color:#888;text-align:center;padding:20px;">No data for this filter</p>';
    document.getElementById('case-count').textContent = '0 cases';
    document.getElementById('pagination').innerHTML = '';
    return;
  }}
  const total = results.length;
  const start = currentPage * PER_PAGE;
  const end = Math.min(start + PER_PAGE, total);
  const pageResults = results.slice(start, end);
  const pLabel = PROMPT_LABELS[currentPrompt] || currentPrompt;
  document.getElementById('case-title').textContent =
    pLabel + ' / ' + currentDataset.toUpperCase() + ' / ' + currentType.toUpperCase();
  document.getElementById('case-count').textContent = total + ' cases';

  let html = '';
  pageResults.forEach(r => {{
    const f1class = r.f1 === 1.0 ? 'perfect' : r.f1 >= 0.7 ? 'good' : 'bad';
    const f1pct = (r.f1 * 100).toFixed(1);
    let catHtml = '';
    CATS.forEach(cat => {{
      const e = r.expected[cat] || [];
      const p = r.predicted[cat] || [];
      if (e.length === 0 && p.length === 0) return;
      const pSet = new Set(p);
      const isMatch = e.length === p.length && e.every(v => pSet.has(v));
      let status, statusCls;
      if (isMatch) {{ status = '\\u2713'; statusCls = 'ok'; }}
      else if (e.length > 0 && p.length === 0) {{ status = 'FN'; statusCls = 'fn'; }}
      else if (e.length === 0 && p.length > 0) {{ status = 'FP'; statusCls = 'fp'; }}
      else {{ status = '\\u2260'; statusCls = 'ne'; }}
      catHtml += `<div class="cat-row">
        <span class="cat-status ${{statusCls}}">${{status}}</span>
        <span class="cat-name">${{cat}}</span>
        <div class="cat-values">
          ${{e.length > 0 ? '<div class="exp">Expected: ' + JSON.stringify(e) + '</div>' : ''}}
          ${{p.length > 0 ? '<div class="pred">Predicted: ' + JSON.stringify(p) + '</div>' : ''}}
          ${{e.length === 0 && p.length > 0 ? '<div class="missing">Expected: (none)</div>' : ''}}
          ${{e.length > 0 && p.length === 0 ? '<div class="missing">Predicted: (none)</div>' : ''}}
        </div>
      </div>`;
    }});
    const docText = r.document_text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    html += `<div class="case-card" onclick="this.classList.toggle('expanded')">
      <div class="case-top">
        <div class="case-meta">
          <span class="case-id">${{r.id}}</span>
          <span class="badge ${{r.difficulty === 'EASY' ? 'base' : r.difficulty === 'HARD' ? 'advanced' : 'combined'}}">${{r.difficulty}}</span>
          <span class="badge vanilla">${{r.category}}</span>
        </div>
        <span class="case-f1 ${{f1class}}">F1: ${{f1pct}}%</span>
      </div>
      <div class="case-intent">${{r.intent}}</div>
      <div class="case-detail">
        <div class="doc-text">${{docText}}</div>
        ${{catHtml}}
      </div>
    </div>`;
  }});
  document.getElementById('case-list').innerHTML = html;

  const totalPages = Math.ceil(total / PER_PAGE);
  let pgHtml = '';
  for (let i = 0; i < totalPages; i++) {{
    pgHtml += `<button class="page-btn ${{i === currentPage ? 'active' : ''}}" onclick="goPage(${{i}})">${{i+1}}</button>`;
  }}
  document.getElementById('pagination').innerHTML = pgHtml;
}}

function goPage(p) {{ currentPage = p; renderCases(); window.scrollTo(0, document.getElementById('case-list').offsetTop - 80); }}

document.querySelectorAll('.filter-group').forEach(group => {{
  group.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      group.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const id = group.id;
      if (id === 'filter-prompt') currentPrompt = btn.dataset.val;
      else if (id === 'filter-dataset') currentDataset = btn.dataset.val;
      else if (id === 'filter-type') currentType = btn.dataset.val;
      currentPage = 0;
      renderCases();
    }});
  }});
}});

// Init
renderSummaryCards();
renderBarChart();
renderStatsTable();
renderCM('combined');
renderCatCM('combined');
renderCases();
</script>
</body>
</html>"""
    return html


def main() -> None:
    print("Loading split files...")
    all_data = load_splits()
    print(f"  Loaded {len(all_data)} splits")

    print("Generating HTML report...")
    html = build_html(all_data)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved to {OUTPUT}")
    print(f"  File size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
