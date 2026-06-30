#!/usr/bin/env python
"""Aggregate the Hill 395 ablation sweep into mean+/-std tables + paired tests.

Reads results/ablation/results.jsonl (written by run_ablation_hill395.py) and
writes, for dev and test separately:
    results/ablation/summary_<split>.csv   variant x metric, "mean+/-std"
    results/ablation/summary.md             formatted tables + paired-test section
and prints the test-split table + paired tests to the console.

Paired test: each variant vs m3_full on per-seed F1 (paired by seed). With only
3 seeds (df=2) statistical power is very low -- a paired t-test cannot reach
p<0.05 unless the effect is large and consistent -- so the sign-consistency
(all seeds agree on direction) and the mean delta are reported alongside p, and
should be read together. (Wilcoxon signed-rank is omitted: at n=3 its smallest
attainable two-sided p is 0.25.)

    python scripts/summarize_ablation.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "ablation"
RESULTS_JSONL = RESULTS / "results.jsonl"

METRICS = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
           "trajectory_consistency_rate", "impossible_transition_rate",
           "dangling_precision", "dangling_recall"]
# Display order: 1A ladder, then 1B leave-one-out, then 1C loss.
ORDER = ["m1", "m2", "m3_full",
         "no_time", "no_motion", "no_state", "no_rel", "no_src",
         "no_ltraj", "no_lassign"]
BASELINE_VARIANT = "m3_full"


# ---- Student's t two-sided p-value via the regularized incomplete beta ----
def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def paired_t(a: list[float], b: list[float]) -> dict:
    """Paired t-test on a-b (a=variant, b=baseline), aligned by seed order."""
    d = np.array(a, float) - np.array(b, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        t = 0.0 if mean == 0.0 else math.inf
        p = 1.0 if mean == 0.0 else 0.0
    else:
        t = mean / (sd / math.sqrt(n))
        p = t_two_sided_p(t, n - 1)
    signs = {np.sign(x) for x in d if x != 0.0}
    consistent = len(signs) == 1 and len(d) > 0
    return {"mean_delta": mean, "t": t, "p": p, "n": n,
            "sign_consistent": bool(consistent)}


def load() -> dict:
    if not RESULTS_JSONL.exists():
        raise SystemExit(f"no results at {RESULTS_JSONL} -- run run_ablation_hill395.py first")
    # data[split][variant][metric] -> {seed: value} (keep seed alignment for pairing)
    data = {"dev": defaultdict(lambda: defaultdict(dict)),
            "test": defaultdict(lambda: defaultdict(dict))}
    for ln in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        for split in ("dev", "test"):
            for mkey in METRICS:
                if mkey in r[split]:
                    data[split][r["variant"]][mkey][int(r["seed"])] = float(r[split][mkey])
    return data


def present_order(data: dict) -> list[str]:
    seen = set(data["test"].keys()) | set(data["dev"].keys())
    ordered = [v for v in ORDER if v in seen]
    ordered += sorted(v for v in seen if v not in ORDER)
    return ordered


def cell(values_by_seed: dict) -> tuple[str, list[float]]:
    vals = [values_by_seed[s] for s in sorted(values_by_seed)]
    if not vals:
        return "-", []
    arr = np.array(vals, float)
    return f"{arr.mean():.4f}±{arr.std(ddof=1) if len(arr) > 1 else 0.0:.4f}", vals


def build_split(data: dict, split: str, variants: list[str]):
    rows = []
    for v in variants:
        row = {"variant": v}
        for mkey in METRICS:
            txt, _ = cell(data[split][v].get(mkey, {}))
            row[mkey] = txt
        rows.append(row)
    return rows


def paired_section(data: dict, split: str, variants: list[str]) -> list[dict]:
    base = data[split].get(BASELINE_VARIANT, {})
    out = []
    base_f1 = base.get("f1", {})
    for v in variants:
        if v == BASELINE_VARIANT:
            continue
        vf1 = data[split][v].get("f1", {})
        seeds = sorted(set(vf1) & set(base_f1))
        if len(seeds) < 2:
            continue
        res = paired_t([vf1[s] for s in seeds], [base_f1[s] for s in seeds])
        out.append({"variant": v, **res})
    return out


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + METRICS)
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict]) -> list[str]:
    head = "| variant | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    lines = [head, sep]
    for r in rows:
        lines.append("| " + r["variant"] + " | " + " | ".join(r[m] for m in METRICS) + " |")
    return lines


def main() -> None:
    data = load()
    variants = present_order(data)
    RESULTS.mkdir(parents=True, exist_ok=True)

    md = ["# 백마고지(Hill 395) — 구성요소 Ablation 결과\n",
          f"변형 {len(variants)}개, 시드 3개 평균±표준편차. paired t-test는 시드별 F1을 "
          f"`{BASELINE_VARIANT}` 기준과 시드 단위로 짝지어 계산(df=2, 저검정력 — 부호 일관성·평균차와 함께 해석).\n"]

    for split in ("dev", "test"):
        rows = build_split(data, split, variants)
        write_csv(RESULTS / f"summary_{split}.csv", rows)
        md.append(f"\n## {split} 집계 (mean±std, n=3 seeds)\n")
        md += md_table(rows)

        md.append(f"\n### {split}: F1 paired t-test vs {BASELINE_VARIANT}\n")
        md.append("| variant | mean ΔF1 | t | p (two-sided) | n | 부호일관 |")
        md.append("|---|---|---|---|---|---|")
        for r in paired_section(data, split, variants):
            md.append(f"| {r['variant']} | {r['mean_delta']:+.4f} | {r['t']:.3f} | "
                      f"{r['p']:.4f} | {r['n']} | {'예' if r['sign_consistent'] else '아니오'} |")

    (RESULTS / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- console: test table + paired tests ----
    print("\n=== TEST split (mean±std over seeds) ===")
    test_rows = build_split(data, "test", variants)
    wv = max(len(r["variant"]) for r in test_rows)
    print(f"{'variant':{wv}}  " + "  ".join(f"{m[:9]:>13}" for m in ("precision", "recall", "f1",
          "wrong_merge_rate", "impossible_transition_rate", "dangling_precision")))
    show = ["precision", "recall", "f1", "wrong_merge_rate", "impossible_transition_rate", "dangling_precision"]
    for r in test_rows:
        print(f"{r['variant']:{wv}}  " + "  ".join(f"{r[m]:>13}" for m in show))

    print(f"\n=== TEST F1 paired t-test vs {BASELINE_VARIANT} (df=2, low power) ===")
    for r in paired_section(data, "test", variants):
        flag = "  *" if (r["p"] < 0.05) else ""
        print(f"  {r['variant']:12} ΔF1={r['mean_delta']:+.4f}  t={r['t']:+.3f}  "
              f"p={r['p']:.4f}  sign_consistent={r['sign_consistent']}{flag}")

    print(f"\nwrote -> {(RESULTS / 'summary.md').relative_to(REPO)}, summary_dev.csv, summary_test.csv")


if __name__ == "__main__":
    main()
