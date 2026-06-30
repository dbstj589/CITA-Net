#!/usr/bin/env python
"""Aggregate + analyse the Hill 395 robustness (difficulty) sweep.

Reads results/robustness/results.jsonl (the 36 new runs) and MERGES the frozen
baseline column D0 from results/ablation/results.jsonl (m3_full/no_motion/no_time
x 3 seeds, which were trained on the frozen suite = noise 1x, dangling 0.2).

Builds, per metric and per variant, mean+/-std(n=3) curves along two axes:
  noise axis   : D0(1x) -> noise2x(2x) -> noise4x(4x)   (dangling fixed 0.2)
  dangling axis: D0(0.2) -> dang035(0.35) -> dang050(0.5) (noise fixed 1x)
and answers the pre-registered questions (a) F1 gap m3_full vs no_motion/no_time
across difficulty (reversal?), (b) impossible_transition specificity of no_motion
under noise, (c) degradation curves. Saves CSV + markdown + (if matplotlib) PNGs.

    python scripts/summarize_robustness.py

Stats: paired t-test on per-seed deltas (df=2, LOW power) via pure-numpy
incomplete-beta (no scipy). p reported WITH sign-consistency + mean delta + std.
per-source-pair precision is NOT in the metrics -> reported as "not measured".
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROB_JSONL = REPO / "results" / "robustness" / "results.jsonl"
ABL_JSONL = REPO / "results" / "ablation" / "results.jsonl"
OUT = REPO / "results" / "robustness"

VARIANTS = ["m3_full", "no_motion", "no_time"]
METRICS = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
           "impossible_transition_rate", "dangling_precision", "dangling_recall"]
NOISE_AXIS = [("D0", 1.0), ("noise2x", 2.0), ("noise4x", 4.0)]      # dangling 0.2
DANG_AXIS = [("D0", 0.2), ("dang035", 0.35), ("dang050", 0.5)]      # noise 1x
BASE = "m3_full"


# ---- Student's t two-sided p via regularized incomplete beta (no scipy) ----
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = FPMIN if abs(d) < FPMIN else d
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def paired_delta(a_by_seed, b_by_seed):
    """a - b paired by seed. Returns mean delta, p (two-sided t), sign-consistency."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in seeds], float)
    n = len(d)
    if n == 0:
        return {"mean": float("nan"), "p": float("nan"), "n": 0, "consistent": False}
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        t, p = (0.0, 1.0) if mean == 0.0 else (math.inf, 0.0)
    else:
        t = mean / (sd / math.sqrt(n)); p = t_two_sided_p(t, n - 1)
    signs = {np.sign(x) for x in d if x != 0.0}
    return {"mean": mean, "p": p, "n": n, "consistent": len(signs) == 1 and len(d) > 0}


def load():
    # data[split][dataset][variant][metric] -> {seed: value}
    data = {sp: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) for sp in ("dev", "test")}
    # D0 from the ablation sweep (frozen suite = noise 1x, dangling 0.2)
    if ABL_JSONL.exists():
        for ln in ABL_JSONL.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["variant"] not in VARIANTS:
                continue
            for sp in ("dev", "test"):
                for mk in METRICS:
                    if mk in r[sp]:
                        data[sp]["D0"][r["variant"]][mk][int(r["seed"])] = float(r[sp][mk])
    # robustness runs
    if ROB_JSONL.exists():
        for ln in ROB_JSONL.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            for sp in ("dev", "test"):
                for mk in METRICS:
                    if mk in r[sp]:
                        data[sp][r["dataset"]][r["variant"]][mk][int(r["seed"])] = float(r[sp][mk])
    return data


def mstd(byseed):
    vals = [byseed[s] for s in sorted(byseed)]
    if not vals:
        return None, None, 0
    a = np.array(vals, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0), len(a)


def fmt(byseed):
    m, s, n = mstd(byseed)
    return "-" if m is None else f"{m:.4f}+/-{s:.4f}"


def write_axis_csv(path, data, split, axis):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "x", "variant"] + METRICS)
        for ds, x in axis:
            for v in VARIANTS:
                w.writerow([ds, x, v] + [fmt(data[split][ds][v][mk]) for mk in METRICS])


def axis_md(data, split, axis, xlabel):
    lines = [f"\n### {split} -- {xlabel} 축\n",
             "| metric | variant | " + " | ".join(f"{ds}({x})" for ds, x in axis) + " |",
             "|" + "---|" * (len(axis) + 2)]
    for mk in METRICS:
        for v in VARIANTS:
            cells = " | ".join(fmt(data[split][ds][v][mk]) for ds, _ in axis)
            lines.append(f"| {mk} | {v} | {cells} |")
    return lines


def analysis_a(data, split, axis, xlabel, md):
    md.append(f"\n#### (a) {xlabel}: F1 격차 m3_full - no_* (음수=m3_full이 낮음; 0 교차=역전)\n")
    md.append("| x | m3_full F1 | no_motion F1 | gap(m3-noM) [p,SC] | no_time F1 | gap(m3-noT) [p,SC] |")
    md.append("|---|---|---|---|---|---|")
    crossed = {"no_motion": None, "no_time": None}
    prev = {"no_motion": None, "no_time": None}
    for ds, x in axis:
        f_full = data[split][ds][BASE]["f1"]
        mf, _, _ = mstd(f_full)
        row = [f"{x}", f"{mf:.4f}" if mf is not None else "-"]
        for nv in ("no_motion", "no_time"):
            f_nv = data[split][ds][nv]["f1"]
            mnv, _, _ = mstd(f_nv)
            pr = paired_delta(f_full, f_nv)  # m3_full - no_v
            sc = "SC" if pr["consistent"] else "--"
            row += [f"{mnv:.4f}" if mnv is not None else "-",
                    f"{pr['mean']:+.4f} [p={pr['p']:.3f},{sc}]" if pr["n"] else "-"]
            if pr["n"]:
                if prev[nv] is not None and prev[nv] < 0 <= pr["mean"]:
                    crossed[nv] = x
                prev[nv] = pr["mean"]
        md.append("| " + " | ".join(row) + " |")
    for nv in ("no_motion", "no_time"):
        if crossed[nv] is not None:
            md.append(f"\n- **{nv}: gap이 x={crossed[nv]}에서 음->양 교차 = m3_full이 역전.**")
        else:
            md.append(f"\n- **{nv}: 역전 관찰 안 됨** (gap이 부호를 바꾸지 않음, 위 표 참조).")
    return md


def analysis_b(data, split, md):
    md.append(f"\n#### (b) {split}: impossible_transition_rate, no_motion - m3_full (노이즈 축; +면 no_motion이 더 나쁨)\n")
    md.append("| noise x | m3_full | no_motion | delta(noM-m3) [p,SC] |")
    md.append("|---|---|---|---|")
    for ds, x in NOISE_AXIS:
        full = data[split][ds][BASE]["impossible_transition_rate"]
        nm = data[split][ds]["no_motion"]["impossible_transition_rate"]
        mf, _, _ = mstd(full); mn, _, _ = mstd(nm)
        pr = paired_delta(nm, full)  # no_motion - m3_full
        sc = "SC" if pr["consistent"] else "--"
        md.append(f"| {x} | {mf:.4f} | {mn:.4f} | {pr['mean']:+.4f} [p={pr['p']:.3f},{sc}] |"
                  if mf is not None and pr["n"] else f"| {x} | - | - | - |")
    md.append("\n- per-source-pair precision은 metrics에 **없음 -> 직접 측정 안 됨** (전체 precision 표만 제공).")
    return md


def make_figures(data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable: {e}; skipping figures)")
        return
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_metrics = ["f1", "recall", "impossible_transition_rate", "wrong_merge_rate"]
    for axis, xlabel, fname in [(NOISE_AXIS, "noise_mult", "noise"), (DANG_AXIS, "dangling_ratio", "dangling")]:
        for split in ("dev", "test"):
            fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.6 * len(plot_metrics), 4.0))
            for ax, mk in zip(axes, plot_metrics):
                for v in VARIANTS:
                    xs, ys, es = [], [], []
                    for ds, x in axis:
                        m, s, n = mstd(data[split][ds][v][mk])
                        if m is not None:
                            xs.append(x); ys.append(m); es.append(s)
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=v)
                ax.set_title(mk); ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            fig.suptitle(f"Hill395 robustness -- {xlabel} axis ({split})")
            fig.tight_layout()
            fig.savefig(figdir / f"robust_{fname}_{split}.png", dpi=110)
            plt.close(fig)
    print(f"figures -> {figdir.relative_to(REPO)}")


def main():
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    # coverage report
    print("=== cells present (variant x dataset, test, n seeds) ===")
    all_ds = ["D0", "noise2x", "noise4x", "dang035", "dang050"]
    for v in VARIANTS:
        print(f"  {v:9}: " + "  ".join(f"{ds}={mstd(data['test'][ds][v]['f1'])[2]}" for ds in all_ds))

    md = ["# 백마고지(Hill 395) 난이도 Sweep (robustness) — 분석\n",
          "변형 3종(m3_full/no_motion/no_time, 전부 Sinkhorn) × 난이도 × 3시드. D0=동결(노이즈1×,"
          " dangling0.2)은 직전 ablation 결과 재사용. 각 칸 mean±std(n=3).\n",
          "노이즈축: D0(1×)→noise2x(2×)→noise4x(4×); dangling축: D0(0.2)→dang035(0.35)→dang050(0.5).\n",
          "> 통계: paired t-test(df=2, 저검정력) — p 단독 의존 금지, 부호일관성[SC]·평균차와 함께 해석.\n"]

    for split in ("dev", "test"):
        write_axis_csv(OUT / f"summary_{split}_noise.csv", data, split, NOISE_AXIS)
        write_axis_csv(OUT / f"summary_{split}_dangling.csv", data, split, DANG_AXIS)
        md.append(f"\n## {split} 집계")
        md += axis_md(data, split, NOISE_AXIS, "노이즈")
        md += axis_md(data, split, DANG_AXIS, "dangling")

    md.append("\n## 핵심 분석")
    for split in ("test", "dev"):
        md.append(f"\n### [{split}]")
        analysis_a(data, split, NOISE_AXIS, "노이즈", md)
        analysis_a(data, split, DANG_AXIS, "dangling", md)
        analysis_b(data, split, md)

    (OUT / "robustness_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    make_figures(data)

    # console: test F1 by axis
    print("\n=== TEST F1 by variant (mean) ===")
    for label, axis in [("noise", NOISE_AXIS), ("dangling", DANG_AXIS)]:
        print(f"  [{label}]")
        for v in VARIANTS:
            pts = "  ".join(f"{ds}({x}):{(mstd(data['test'][ds][v]['f1'])[0] or float('nan')):.4f}" for ds, x in axis)
            print(f"    {v:9} {pts}")
    print(f"\nwrote -> {(OUT / 'robustness_analysis.md').relative_to(REPO)}, summary_*_{{noise,dangling}}.csv")


if __name__ == "__main__":
    main()
