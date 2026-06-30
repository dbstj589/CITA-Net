#!/usr/bin/env python
"""Evaluate term_gating / m3_full / no_motion / no_rel on the hard-ambiguity suite.

New data => no recorded baselines; all variants are trained fresh on amb80 and
scored here with the SAME default decode_full (decoder-on variants), n=3 seeds.
Answers: (a) do gating/m3_full overtake no_motion here? (b) does m3_full beat
no_motion (constraints finally help)? (c) gate differential + ambiguity recheck.

dev for selection, test reported. No retraining here.

    python scripts/eval_hard_ambiguity.py --data-root data/hard_ambiguity/amb80 --tag amb80
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

from citanet.config import load_config
from citanet.decode import decode_full
from citanet.engine_large import load_large_bundle
from citanet.eval import aggregate, evaluate_full
from citanet.data.stream import featurize_sector, read_split
from citanet.model.cta import CTA

REPO = Path(__file__).resolve().parents[1]
SEEDS = [19521006, 19521007, 19521008]
TERMS = ["sem", "time", "motion", "state", "rel", "src"]
PANEL = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
         "trajectory_consistency_rate", "impossible_transition_rate",
         "dangling_precision", "dangling_recall"]
# model -> config (run dirs are <runs>/<model>_seed<seed>)
MODELS = {
    "term_gating": REPO / "configs" / "term_gating.yaml",
    "m3_full":     REPO / "configs" / "ablation" / "m3_full.yaml",
    "no_motion":   REPO / "configs" / "ablation" / "no_motion.yaml",
    "no_rel":      REPO / "configs" / "ablation" / "no_rel.yaml",
}


# ---- pure-numpy paired t (from summarize_robustness.py) ----
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap
    d = FPMIN if abs(d) < FPMIN else d; d = 1.0 / d; h = d
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
    return 1.0 if t == 0.0 else _betai(df / 2.0, 0.5, df / (df + t * t))


def paired_delta(a_by_seed, b_by_seed):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in seeds], float)
    n = len(d)
    if n == 0:
        return {"mean": float("nan"), "p": float("nan"), "consistent": False}
    mean = float(d.mean())
    if n == 1 or d.std(ddof=1) == 0.0:
        p = 0.0 if mean != 0.0 else 1.0
    else:
        p = t_two_sided_p(mean / (d.std(ddof=1) / math.sqrt(n)), n - 1)
    signs = {np.sign(x) for x in d if x != 0.0}
    return {"mean": mean, "p": p, "consistent": len(signs) == 1 and len(d) > 0}


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    return (st.fmean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/hard_ambiguity/amb80")
    ap.add_argument("--ontology-dir", default=None)
    ap.add_argument("--runs-dir", default="runs/hard_ambiguity/amb80")
    ap.add_argument("--tag", default="amb80")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    data_root = args.data_root
    onto = args.ontology_dir or (data_root + "/ontology")
    runs_dir = REPO / args.runs_dir if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "results" / "hard_ambiguity" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []                         # per (model, seed, split)
    gate_acc = {b: {t: [] for t in TERMS} for b in ("easy", "ambig", "all")}
    fid_lines, fid_ok = [], True

    for model_name, cfg_path in MODELS.items():
        cfg = load_config(cfg_path)
        cfg.data_root = data_root
        cfg.ontology_dir = onto
        for seed in args.seeds:
            run_dir = runs_dir / f"{model_name}_seed{seed}"
            if not (run_dir / "model.pt").exists():
                raise SystemExit(f"missing checkpoint {run_dir/'model.pt'} -- train first.")
            model, fs, ont = load_large_bundle(cfg, run_dir)
            for split in ("dev", "test"):
                per = []
                for sid in read_split(cfg.data_root, split):
                    sd = Path(cfg.data_root) / "sectors" / sid
                    feats = featurize_sector(sd, fs, ont, cfg)
                    with torch.no_grad():
                        out = model(feats)
                    per.append(evaluate_full(decode_full(out, feats, ont), feats, sd))
                    if model_name == "term_gating" and split == "dev" and getattr(model.cta, "last_gate", None) is not None:
                        g, sim, cross = model.cta.last_gate, model.cta.last_raw_sim, model.cta.last_cross.bool()
                        if cross.any():
                            gc, sc = g[cross], sim[cross]; easy = sc > 0.5
                            for ti, t in enumerate(TERMS):
                                gate_acc["all"][t].extend(gc[:, ti].tolist())
                                gate_acc["easy"][t].extend(gc[easy, ti].tolist())
                                gate_acc["ambig"][t].extend(gc[~easy, ti].tolist())
                rows.append({"model": model_name, "seed": seed, "split": split, **aggregate(per)})
            # fidelity recheck for the gated model
            if model_name == "term_gating":
                sd = Path(cfg.data_root) / "sectors" / read_split(cfg.data_root, "dev")[0]
                feats = featurize_sector(sd, fs, ont, cfg)
                with torch.no_grad():
                    h = model.encoder(feats)
                    if model.graph is not None:
                        h = model.graph(h, feats)
                    base = CTA.forward(model.cta, h, feats)
                    model.cta.force_gate_one = True
                    g1 = model.cta(h, feats); model.cta.force_gate_one = False
                    dd = max((getattr(base, n) - getattr(g1, n)).abs().max().item()
                             for n in ("score", "sim_sem", "b_motion", "b_rel"))
                ok = dd < 1e-6; fid_ok &= ok
                fid_lines.append(f"  term_gating seed{seed}: gate==1 vs base max|Δ|={dd:.1e} {'OK' if ok else 'FAIL'}")
            print(f"[{model_name} seed{seed}] done")

    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    def cell(model, split, met):
        return mean_std([r[met] for r in rows if r["model"] == model and r["split"] == split and met in r])

    def by_seed(model, split, met):
        return {r["seed"]: r[met] for r in rows if r["model"] == model and r["split"] == split and met in r}

    # CSV
    for split in ("dev", "test"):
        with open(out_dir / f"compare_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["model"] + [f"{m}_mean" for m in PANEL] + [f"{m}_std" for m in PANEL])
            for mdl in MODELS:
                w.writerow([mdl] + [round(cell(mdl, split, m)[0], 4) for m in PANEL]
                           + [round(cell(mdl, split, m)[1], 4) for m in PANEL])

    md = build_md(args.tag, cell, by_seed, gate_acc, fid_lines, fid_ok)
    (out_dir / "analysis.md").write_text(md, encoding="utf-8")
    try:
        plot(out_dir, args.tag, cell, gate_acc)
        fignote = f"figures -> {out_dir/'figures'}"
    except Exception as e:  # noqa: BLE001
        fignote = f"(no figures: {e})"

    print("\n--- fidelity ---"); print("\n".join(fid_lines))
    print("\n" + md)
    print(f"\nwrote -> {out_dir}")
    print(fignote)


def build_md(tag, cell, by_seed, gate_acc, fid_lines, fid_ok):
    L = [f"# 어려운(쌍 모호성↑) 데이터에서 게이팅 vs 비교군 — {tag}\n",
         "동종 밀도 2x(identities/sector=80), dangling 0.2 고정, noise 1.0. 4모델 새로 학습, "
         "기본 decode_full로 채점. dev 선택·test 보고, n=3.\n",
         f"**게이팅 충실성 (gate==1 == 기존 CTA):** {'통과' if fid_ok else '실패'}"]
    L += fid_lines
    for split in ("dev", "test"):
        L.append(f"\n## {split} (mean±std, n=3)\n")
        L.append("| model | P | R | F1 | wrong_merge | frag | impossible | dang_R |")
        L.append("|---|---|---|---|---|---|---|---|")
        for mdl in MODELS:
            c = {m: cell(mdl, split, m) for m in PANEL}
            L.append("| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                mdl, *c["precision"], *c["recall"], *c["f1"], c["wrong_merge_rate"][0],
                c["fragmentation_rate"][0], c["impossible_transition_rate"][0], c["dangling_recall"][0]))

    def verdict(a, b, split):
        pd = paired_delta(by_seed(a, split, "f1"), by_seed(b, split, "f1"))
        return pd
    L.append("\n## 핵심 질문 (dev·test 둘 다 보고; ΔF1=좌−우, n=3 저검정력)\n")
    for split in ("dev", "test"):
        tg_nm = verdict("term_gating", "no_motion", split)
        m3_nm = verdict("m3_full", "no_motion", split)
        tg_m3 = verdict("term_gating", "m3_full", split)
        L.append(f"**[{split}]** term_gating F1={cell('term_gating',split,'f1')[0]:.3f}, "
                 f"m3_full={cell('m3_full',split,'f1')[0]:.3f}, no_motion={cell('no_motion',split,'f1')[0]:.3f}, "
                 f"no_rel={cell('no_rel',split,'f1')[0]:.3f}")
        L.append(f"- (a) term_gating−no_motion = {tg_nm['mean']:+.4f} "
                 f"[{'SC' if tg_nm['consistent'] else '--'},p={tg_nm['p']:.3f}] "
                 f"({'넘음' if tg_nm['mean']>0 else '못넘음'})")
        L.append(f"- (b) m3_full−no_motion = {m3_nm['mean']:+.4f} "
                 f"[{'SC' if m3_nm['consistent'] else '--'},p={m3_nm['p']:.3f}] "
                 f"({'제약 순기여' if m3_nm['mean']>0 else '제약 순기여 아님'})")
        L.append(f"- term_gating−m3_full = {tg_m3['mean']:+.4f} "
                 f"[{'SC' if tg_m3['consistent'] else '--'},p={tg_m3['p']:.3f}]\n")

    # dev/test consistency note
    a_dev = verdict("m3_full", "no_motion", "dev")["mean"]
    a_test = verdict("m3_full", "no_motion", "test")["mean"]
    consistent = (a_dev > 0) == (a_test > 0)
    L.append(f"> m3_full−no_motion 부호: dev {a_dev:+.4f} / test {a_test:+.4f} → "
             f"{'일치(신뢰 가능 방향)' if consistent else '불일치 → 신뢰 불가'}.")

    L.append("\n## (c) 게이트 차등 작동 (dev cross-KG, 쉬운 sim>0.5 vs 모호 sim≤0.5)\n")
    L.append("| term | gate(전체) | gate(쉬운) | gate(모호) | 쉬운−모호 |")
    L.append("|---|---|---|---|---|")
    for t in TERMS:
        if not gate_acc["all"][t]:
            continue
        ga, ge, gm = (st.fmean(gate_acc[k][t]) if gate_acc[k][t] else float("nan")
                      for k in ("all", "easy", "ambig"))
        L.append(f"| {t} | {ga:.3f} | {ge:.3f} | {gm:.3f} | {ge-gm:+.3f} |")
    L.append("\n> n=3 저검정력. dev/test 부호가 갈리는 항목은 '신뢰 불가'로 해석.")
    return "\n".join(L)


def plot(out_dir, tag, cell, gate_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    models = list(MODELS)
    for split in ("dev", "test"):
        f1 = [cell(m, split, "f1")[0] for m in models]
        er = [cell(m, split, "f1")[1] for m in models]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        colors = ["tab:green", "tab:purple", "tab:blue", "tab:orange"]
        ax.bar(models, f1, yerr=er, capsize=4, color=colors, alpha=0.85)
        ax.axhline(cell("no_motion", split, "f1")[0], ls="--", color="tab:blue", alpha=0.6,
                   label=f"no_motion={cell('no_motion',split,'f1')[0]:.3f}")
        for i, (v, e) in enumerate(zip(f1, er)):
            ax.text(i, v + e + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylabel(f"{split} F1"); ax.set_ylim(0, max(f1) * 1.2)
        ax.set_title(f"Hard-ambiguity ({tag}) {split} F1, default decode (n=3)")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / f"f1_{split}.png", dpi=130); plt.close(fig)
    # gate by difficulty
    if gate_acc["easy"]["motion"]:
        x = np.arange(len(TERMS)); w = 0.35
        ge = [st.fmean(gate_acc["easy"][t]) for t in TERMS]
        gm = [st.fmean(gate_acc["ambig"][t]) for t in TERMS]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, ge, w, label="easy (sim>0.5)", color="tab:green", alpha=0.85)
        ax.bar(x + w / 2, gm, w, label="ambiguous (sim<=0.5)", color="tab:orange", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(TERMS); ax.set_ylabel("mean gate"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Term gates by pair difficulty -- {tag} (dev cross-KG)")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "gates.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
