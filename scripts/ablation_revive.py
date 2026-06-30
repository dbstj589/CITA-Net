#!/usr/bin/env python
"""Re-score the component ablation with the REVIVE decoder (fixed margin) -- NO RETRAIN.

The original ablation (results/ablation) was scored with the default decode_full.
Here we reload the SAME trained checkpoints and re-decode every variant with the
argmax-∅ revive decoder at a FIXED margin (0.70), so each constraint/component's
contribution is re-read under one common, less-conservative decode condition.

No retraining: each variant's checkpoint is loaded with ITS OWN config (so the
architecture matches the trained weights), forwarded once per sector, then decoded
with decode_full_revive(margin=0.70) and scored by evaluate_full.

Variants: m3_full, no_time, no_motion, no_state, no_rel, no_src
  (m1/m2 are the greedy path -- not comparable under this Sinkhorn decoder.)

Fidelity: we ALSO decode at margin=0 and check it reproduces the recorded
results/ablation numbers per variant (the revive decoder == decode_full at m=0).

decode.py is untouched: the revive decoder is imported from redecode_revive.py.

    python scripts/ablation_revive.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import torch

# reuse the revive decoder (decode.py left untouched); this import works because
# scripts/ is on sys.path[0] when running `python scripts/ablation_revive.py`.
from redecode_revive import decode_full_revive

from citanet.config import load_config
from citanet.engine_large import load_large_bundle
from citanet.eval import aggregate, evaluate_full
from citanet.data.stream import featurize_sector, read_split

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "configs" / "ablation"
RUNS = REPO / "runs" / "ablation"
ABL_JSONL = REPO / "results" / "ablation" / "results.jsonl"
DEFAULT_SEEDS = [19521006, 19521007, 19521008]
VARIANTS = ["m3_full", "no_time", "no_motion", "no_state", "no_rel", "no_src"]
REF = "m3_full"
MARGIN = 0.70

PANEL = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
         "trajectory_consistency_rate", "impossible_transition_rate",
         "dangling_precision", "dangling_recall"]
# which diagnostic each constraint term targets (question b)
DIAGNOSTIC = {
    "no_motion": [("impossible_transition_rate", "↑")],
    "no_state":  [("wrong_merge_rate", "↑")],
    "no_rel":    [("recall", "↓"), ("fragmentation_rate", "↑")],
    "no_time":   [("wrong_merge_rate", "↑"), ("impossible_transition_rate", "↑")],
}


# ---- pure-numpy paired t-test (copied from summarize_robustness.py) ----
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
    """a - b paired by seed. mean delta, two-sided t p, sign-consistency."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in seeds], float)
    n = len(d)
    if n == 0:
        return {"mean": float("nan"), "p": float("nan"), "n": 0, "consistent": False}
    mean = float(d.mean())
    if n == 1 or d.std(ddof=1) == 0.0:
        p = 0.0 if mean != 0.0 else 1.0
    else:
        sd = float(d.std(ddof=1))
        t = mean / (sd / math.sqrt(n)); p = t_two_sided_p(t, n - 1)
    signs = {np.sign(x) for x in d if x != 0.0}
    return {"mean": mean, "p": p, "n": n, "consistent": len(signs) == 1 and len(d) > 0}


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    return (st.fmean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


def recorded_baseline():
    """per-variant recorded dev/test metrics from the original ablation sweep."""
    if not ABL_JSONL.exists():
        return {}
    rows = [json.loads(l) for l in ABL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for v in VARIANTS:
        vr = [r for r in rows if r.get("variant") == v]
        if not vr:
            continue
        out[v] = {}
        for split in ("dev", "test"):
            for met in PANEL:
                out[v].setdefault(split, {})[met] = mean_std(
                    [r[split][met] for r in vr if split in r and met in r[split]])[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "ablation_revive"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    margins = [0.0, args.margin]   # 0.0 = fidelity, margin = revive condition

    print(f"== component ablation re-scored with REVIVE decoder (margin={args.margin}, NO retrain) ==")
    print(f"   variants={args.variants}  seeds={args.seeds}")

    # rows: per (variant, seed, split, margin) aggregated metric dict
    rows = []
    for v in args.variants:
        cfg = load_config(CFG_DIR / f"{v}.yaml")
        for seed in args.seeds:
            run_dir = RUNS / f"{v}_seed{seed}"
            if not (run_dir / "model.pt").exists():
                raise SystemExit(f"missing checkpoint: {run_dir/'model.pt'} -- abort (no retrain).")
            model, fs, ont = load_large_bundle(cfg, run_dir)
            for split in ("dev", "test"):
                cache = []
                for sid in read_split(cfg.data_root, split):
                    sd = Path(cfg.data_root) / "sectors" / sid
                    feats = featurize_sector(sd, fs, ont, cfg)
                    with torch.no_grad():
                        out = model(feats)
                    cache.append((out, feats, sd))
                for mg in margins:
                    per = [evaluate_full(decode_full_revive(o, f, ont, margin=mg), f, sd)
                           for (o, f, sd) in cache]
                    rows.append({"variant": v, "seed": seed, "split": split,
                                 "margin": mg, **aggregate(per)})
                del cache
            print(f"   [{v} seed{seed}] done")

    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    # ---- fidelity: margin=0 vs recorded ablation (per variant, test F1) ----
    base = recorded_baseline()
    fid_lines, fid_ok = [], True
    for v in args.variants:
        got = mean_std([r["f1"] for r in rows if r["variant"] == v and r["split"] == "test"
                        and r["margin"] == 0.0])[0]
        ref = base.get(v, {}).get("test", {}).get("f1")
        if ref is None:
            continue
        d = abs(got - ref); ok = d < 5e-3; fid_ok &= ok
        fid_lines.append(f"  {v:10} test F1: m0={got:.4f} vs recorded={ref:.4f} "
                         f"(Δ={d:.4f}) {'OK' if ok else 'MISMATCH'}")

    # ---- aggregate at revive margin: per (variant, split, metric) mean±std ----
    def cell(v, split, met, mg=args.margin):
        return mean_std([r[met] for r in rows if r["variant"] == v and r["split"] == split
                         and r["margin"] == mg and met in r])

    def by_seed(v, split, met, mg=args.margin):
        return {r["seed"]: r[met] for r in rows if r["variant"] == v and r["split"] == split
                and r["margin"] == mg and met in r}

    # ---- CSV (revive margin) ----
    for split in ("dev", "test"):
        with open(out_dir / f"ablation_revive_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            head = ["variant"]
            for m in PANEL:
                head += [f"{m}_mean", f"{m}_std"]
            head += ["dF1_vs_full", "F1_sign_consistent", "F1_p"]
            w.writerow(head)
            for v in args.variants:
                row = [v]
                for m in PANEL:
                    mu, sd = cell(v, split, m)
                    row += [round(mu, 4), round(sd, 4)]
                if v == REF:
                    row += [0.0, "ref", "ref"]
                else:
                    pd = paired_delta(by_seed(v, split, "f1"), by_seed(REF, split, "f1"))
                    row += [round(pd["mean"], 4), pd["consistent"], round(pd["p"], 4)]
                w.writerow(row)

    md = build_md(args.margin, args.variants, cell, by_seed, base, fid_lines, fid_ok)
    (out_dir / "ablation_revive_analysis.md").write_text(md, encoding="utf-8")

    try:
        plot(out_dir, args.margin, args.variants, cell)
        fignote = f"figures -> {out_dir/'figures'}"
    except Exception as e:  # noqa: BLE001
        fignote = f"(no figures: {e})"

    print("\n--- fidelity (margin=0 vs recorded) ---")
    print("\n".join(fid_lines))
    print(f"fidelity: {'PASS' if fid_ok else 'FAIL'}")
    print("\n" + md)
    print("\nwrote -> results.jsonl, ablation_revive_{dev,test}.csv, ablation_revive_analysis.md")
    print(fignote)


def build_md(margin, variants, cell, by_seed, base, fid_lines, fid_ok):
    L = [f"# 구성요소 ablation — 되살리기 디코더 재채점 (margin={margin}, 재학습 없음)\n",
         "각 변형의 학습 체크포인트를 자기 config로 로드, Sinkhorn 배정 불변, "
         f"argmax-∅ 되살리기 디코더(margin={margin} 고정)로만 재디코드+재채점.\n",
         f"**충실성 (margin=0 == 기록된 decode_full):** {'통과' if fid_ok else '실패'}"]
    L += fid_lines
    L.append("")
    if not fid_ok:
        L.append("> ⚠️ 충실성 실패 — 아래 해석 보류.\n")

    for split in ("dev", "test"):
        L.append(f"\n## {split} — 변형별 전체 지표 (mean±std, n=3, 되살리기 margin={margin})\n")
        L.append("| variant | P | R | F1 | ΔF1 vs full [SC,p] | wrong_merge | frag | traj_cons | impossible | dang_P | dang_R |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for v in variants:
            c = {m: cell(v, split, m) for m in PANEL}
            if v == REF:
                df1 = "— (ref)"
            else:
                pd = paired_delta(by_seed(v, split, "f1"), by_seed(REF, split, "f1"))
                sc = "SC" if pd["consistent"] else "--"
                df1 = f"{pd['mean']:+.4f} [{sc},p={pd['p']:.3f}]"
            L.append("| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {} | "
                     "{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                         v, *c["precision"], *c["recall"], *c["f1"], df1,
                         c["wrong_merge_rate"][0], c["fragmentation_rate"][0],
                         c["trajectory_consistency_rate"][0],
                         c["impossible_transition_rate"][0],
                         c["dangling_precision"][0], c["dangling_recall"][0]))

    # ---- (a) does removing a term drop F1 (test, report split) ----
    L.append("\n## 핵심 질문 (test 보고; margin 고정 0.70)\n")
    L.append("**(a) 제약 항 제거 시 F1이 m3_full 대비 떨어지나/오르나? (ΔF1=variant−full)**\n")
    L.append("| variant | test ΔF1 | 방향 | 부호일관성 | p |")
    L.append("|---|---|---|---|---|")
    for v in variants:
        if v == REF:
            continue
        pd = paired_delta(by_seed(v, "test", "f1"), by_seed(REF, "test", "f1"))
        dirn = "하락(full이 우위)" if pd["mean"] < 0 else ("상승(변형이 우위)" if pd["mean"] > 0 else "동일")
        L.append(f"| {v} | {pd['mean']:+.4f} | {dirn} | {'SC' if pd['consistent'] else '--'} | "
                 f"{pd['p']:.3f} |")

    # ---- (b) targeted-diagnostic specificity ----
    L.append("\n**(b) 각 제약 항의 겨냥 진단지표가 그 항 제거 시 특이적으로 악화되나? "
             "(Δ=variant−full; ↑=악화 기대)**\n")
    L.append("| variant | 진단지표 | 기대 | test Δ | 부호일관성 | 관찰 |")
    L.append("|---|---|---|---|---|---|")
    for v in variants:
        if v not in DIAGNOSTIC:
            continue
        for met, exp in DIAGNOSTIC[v]:
            pd = paired_delta(by_seed(v, "test", met), by_seed(REF, "test", met))
            worse = (pd["mean"] > 0) if exp == "↑" else (pd["mean"] < 0)
            obs = ("예상대로 악화" if (worse and pd["consistent"]) else
                   ("악화 방향이나 부호 불일치" if worse else "관찰 안 됨(반대/무변화)"))
            L.append(f"| {v} | {met} | {exp} | {pd['mean']:+.4f} | "
                     f"{'SC' if pd['consistent'] else '--'} | {obs} |")

    L.append("\n**(c) per-source-pair precision:** metrics에 **없음** → 측정 안 됨 "
             "(전체 precision만 제공).")
    L.append("\n> 통계: paired t-test(df=2, 저검정력). p 단독 의존 금지 — ΔF1·부호일관성[SC]·std와 함께 해석.")
    return "\n".join(L)


def plot(out_dir, margin, variants, cell):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for split in ("dev", "test"):
        mus = [cell(v, split, "f1")[0] for v in variants]
        sds = [cell(v, split, "f1")[1] for v in variants]
        ref = cell(REF, split, "f1")[0]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        colors = ["tab:gray" if v == REF else "tab:blue" for v in variants]
        ax.bar(variants, mus, yerr=sds, capsize=4, color=colors, alpha=0.85)
        ax.axhline(ref, ls="--", color="tab:gray", label=f"m3_full={ref:.3f}")
        for i, (mu, sd) in enumerate(zip(mus, sds)):
            ax.text(i, mu + sd + 0.005, f"{mu:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("F1")
        ax.set_ylim(0, max(mus) * 1.18)
        ax.set_title(f"Component ablation under REVIVE decoder (margin={margin}) -- {split} (n=3)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"ablation_revive_{split}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
