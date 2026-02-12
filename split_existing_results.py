#!/usr/bin/env python3
"""기존 결과 파일을 base/advanced로 분할 저장."""

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_pii_evaluation import PII_CATEGORIES, compute_metrics, print_report

RESULTS_DIR = Path(__file__).parent / "benchmark_results" / "300"


def split_result_file(name: str) -> None:
    path = RESULTS_DIR / f"results_{name}.json"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct internal format from saved format
    internal_results: list[dict] = []
    for r in data["results"]:
        expected = {cat: r["expected"].get(cat) for cat in PII_CATEGORIES}
        predicted = {cat: r["predicted"].get(cat) for cat in PII_CATEGORIES}
        metrics = compute_metrics(expected, predicted)
        internal_results.append({
            "id": r["id"],
            "category": r["category"],
            "difficulty": r["difficulty"],
            "intent": r["intent"],
            "expected": expected,
            "predicted": predicted,
            "metrics": metrics,
            "raw_response": r.get("raw_response", ""),
        })

    for suffix, filter_fn in [
        ("_base", lambda r: r["difficulty"] == "EASY"),
        ("_advanced", lambda r: r["difficulty"] in ("MEDIUM", "HARD")),
    ]:
        subset = [r for r in internal_results if filter_fn(r)]
        if not subset:
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            split_summary = print_report(subset)

        split_data = {
            "model": data["model"],
            "api_url": data.get("api_url", ""),
            "concurrency": data.get("concurrency", 0),
            "inference_time_sec": data.get("inference_time_sec", 0),
            "timestamp": data.get("timestamp", ""),
            "summary": split_summary,
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
                for r in subset
            ],
        }
        out_path = RESULTS_DIR / f"results_{name}{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        s = split_summary
        print(f"  {out_path.name}: {len(subset)}개, F1={s['overall_f1']*100:.2f}%, Perfect={s['perfect_cases']}/{s['total_cases']}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["qwen3_30b_a3b_fp8", "qwen3next_80b_a3b_int4"]
    for name in targets:
        print(f"[{name}]")
        split_result_file(name)
    print("Done!")
