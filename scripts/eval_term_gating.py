#!/usr/bin/env python
"""Evaluate the learned term-gating model vs the recorded ablation variants.

Loads the 3 trained term-gating checkpoints (runs/term_gating/seed*), scores them
with the SAME default decode_full used to produce the recorded ablation numbers
(so the comparison is apples-to-apples), and contrasts against the recorded
results/ablation values (m3_full / no_motion / no_time / no_state / no_rel / no_src).

Also introspects the learned gates: mean gate per term, and per term split by pair
difficulty (easy = high semantic sim vs ambiguous = low sim) over cross-KG
candidates -- the mechanism evidence for "gate constraints off on easy pairs".

dev is for selection; test is reported. No retraining here.

    python scripts/eval_term_gating.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.decode import decode_full
from citanet.engine_large import load_large_bundle
from citanet.eval import aggregate, evaluate_full
from citanet.data.stream import featurize_sector, read_split
from citanet.model.cta import CTA

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "term_gating.yaml"
RUNS = REPO / "runs" / "term_gating"
ABL_JSONL = REPO / "results" / "ablation" / "results.jsonl"
SEEDS = [19521006, 19521007, 19521008]
BASE_VARIANTS = ["m3_full", "no_motion", "no_time", "no_state", "no_rel", "no_src"]
TERMS = ["sem", "time", "motion", "state", "rel", "src"]
PANEL = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
         "trajectory_consistency_rate", "impossible_transition_rate",
         "dangling_precision", "dangling_recall"]


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    return (st.fmean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


def recorded_baselines():
    rows = [json.loads(l) for l in ABL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for v in BASE_VARIANTS:
        vr = [r for r in rows if r.get("variant") == v]
        out[v] = {}
        for split in ("dev", "test"):
            for met in PANEL:
                out[v].setdefault(split, {})[met] = mean_std(
                    [r[split][met] for r in vr if split in r and met in r[split]])[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "term_gating"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CFG)

    rows = []                      # per (seed, split) aggregated metrics
    fid_lines, fid_ok = [], True
    gate_acc = {b: {t: [] for t in TERMS} for b in ("easy", "ambig", "all")}
    gate_seed_means = []           # per-seed overall gate means (for std)

    for seed in args.seeds:
        run_dir = RUNS / f"seed{seed}"
        if not (run_dir / "model.pt").exists():
            raise SystemExit(f"missing checkpoint {run_dir/'model.pt'} -- train first.")
        model, fs, ont = load_large_bundle(cfg, run_dir)
        assert type(model.cta).__name__ == "GatedCTA", "loaded model is not gated!"

        # per-seed gate accumulation
        gseed = {t: [] for t in TERMS}
        for split in ("dev", "test"):
            per = []
            for sid in read_split(cfg.data_root, split):
                sd = Path(cfg.data_root) / "sectors" / sid
                feats = featurize_sector(sd, fs, ont, cfg)
                with torch.no_grad():
                    out = model(feats)
                per.append(evaluate_full(decode_full(out, feats, ont), feats, sd))
                # gate introspection on DEV cross-KG candidates only
                if split == "dev":
                    g = model.cta.last_gate            # (P,6)
                    sim = model.cta.last_raw_sim       # (P,)
                    cross = model.cta.last_cross.bool()
                    if cross.any():
                        gc, sc = g[cross], sim[cross]
                        easy = sc > 0.5
                        for ti, t in enumerate(TERMS):
                            col = gc[:, ti]
                            gate_acc["all"][t].extend(col.tolist())
                            gate_acc["easy"][t].extend(col[easy].tolist())
                            gate_acc["ambig"][t].extend(col[~easy].tolist())
                            gseed[t].extend(col.tolist())
            rows.append({"seed": seed, "split": split, **aggregate(per)})

        gate_seed_means.append({t: st.fmean(gseed[t]) for t in TERMS})

        # light fidelity recheck (gate==1 == base CTA) on one dev sector
        sd = Path(cfg.data_root) / "sectors" / read_split(cfg.data_root, "dev")[0]
        feats = featurize_sector(sd, fs, ont, cfg)
        with torch.no_grad():
            h = model.encoder(feats)
            if model.graph is not None:
                h = model.graph(h, feats)
            base = CTA.forward(model.cta, h, feats)
            model.cta.force_gate_one = True
            g1 = model.cta(h, feats)
            model.cta.force_gate_one = False
            d = max((getattr(base, n) - getattr(g1, n)).abs().max().item()
                    for n in ("score", "sim_sem", "b_motion", "b_state", "b_rel", "b_src"))
        ok = d < 1e-6; fid_ok &= ok
        fid_lines.append(f"  seed{seed}: gate==1 vs base CTA max|Δ|={d:.2e} {'OK' if ok else 'FAIL'}")
        print(f"[seed {seed}] evaluated. fidelity Δ={d:.1e}")

    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    base = recorded_baselines()

    def cell(split, met):
        return mean_std([r[met] for r in rows if r["split"] == split and met in r])

    # ---- CSV: gating vs recorded baselines (test + dev) ----
    for split in ("dev", "test"):
        with open(out_dir / f"compare_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model"] + PANEL)
            row = ["term_gating"]
            for m in PANEL:
                mu, sd = cell(split, m)
                row.append(f"{mu:.4f}±{sd:.4f}")
            w.writerow(row)
            for v in BASE_VARIANTS:
                w.writerow([v] + [f"{base[v][split][m]:.4f}" for m in PANEL])

    # ---- gate summary CSV ----
    with open(out_dir / "gate_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "gate_all_mean", "gate_easy(sim>0.5)", "gate_ambig(sim<=0.5)",
                    "easy_minus_ambig"])
        for t in TERMS:
            ga = st.fmean(gate_acc["all"][t]) if gate_acc["all"][t] else float("nan")
            ge = st.fmean(gate_acc["easy"][t]) if gate_acc["easy"][t] else float("nan")
            gm = st.fmean(gate_acc["ambig"][t]) if gate_acc["ambig"][t] else float("nan")
            w.writerow([t, round(ga, 4), round(ge, 4), round(gm, 4), round(ge - gm, 4)])

    md = build_md(cfg, cell, base, fid_lines, fid_ok, gate_acc, gate_seed_means)
    (out_dir / "term_gating_analysis.md").write_text(md, encoding="utf-8")
    try:
        plot(out_dir, cell, base, gate_acc)
        fignote = f"figures -> {out_dir/'figures'}"
    except Exception as e:  # noqa: BLE001
        fignote = f"(no figures: {e})"

    print("\n--- fidelity ---"); print("\n".join(fid_lines))
    print("\n" + md)
    print("\nwrote -> results.jsonl, compare_{dev,test}.csv, gate_summary.csv, "
          "term_gating_analysis.md")
    print(fignote)


def build_md(cfg, cell, base, fid_lines, fid_ok, gate_acc, gate_seed_means):
    L = ["# 학습된 항별 게이팅(term gating) — 평가/비교 (D0, 재학습은 게이팅 모델만)\n",
         "CTA 합산을 score=Σ g_k(쌍맥락)·term_k 로 바꾼 단일 모델 변경. 3시드 40epoch, "
         "기본 decode_full로 채점(기록된 ablation과 동일 조건).\n",
         f"**충실성 (gate==1 == 기존 CTA):** {'통과' if fid_ok else '실패'}"]
    L += fid_lines
    L.append("\n## 비교표 (mean±std, n=3 / 기록값은 ablation 그대로)\n")
    L.append("| model | test F1 | test P | test R | dev F1 | wrong_merge | impossible | dang_R |")
    L.append("|---|---|---|---|---|---|---|---|")
    tg = {m: cell("test", m) for m in PANEL}
    dgf = cell("dev", "f1")
    L.append("| **term_gating** | **{:.3f}±{:.3f}** | {:.3f} | {:.3f} | {:.3f}±{:.3f} | "
             "{:.3f} | {:.3f} | {:.3f} |".format(
                 *tg["f1"], tg["precision"][0], tg["recall"][0], *dgf,
                 tg["wrong_merge_rate"][0], tg["impossible_transition_rate"][0],
                 tg["dangling_recall"][0]))
    for v in BASE_VARIANTS:
        b = base[v]
        L.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            v, b["test"]["f1"], b["test"]["precision"], b["test"]["recall"], b["dev"]["f1"],
            b["test"]["wrong_merge_rate"], b["test"]["impossible_transition_rate"],
            b["test"]["dangling_recall"]))

    tgf, tgf_s = cell("test", "f1")
    m3 = base["m3_full"]["test"]["f1"]; nm = base["no_motion"]["test"]["f1"]
    L.append("\n## 핵심 질문 (dev로 선택, test 보고)\n")
    L.append(f"**(a) F1:** term_gating test F1 = {tgf:.3f}±{tgf_s:.3f}. "
             f"m3_full({m3:.3f}) 대비 {tgf-m3:+.3f} ({'넘음' if tgf>m3 else '못 넘음'}); "
             f"no_motion({nm:.3f}) 대비 {tgf-nm:+.3f} ({'넘음' if tgf>nm else '못 넘음'}).")
    tgr = cell("test", "recall")[0]; m3r = base["m3_full"]["test"]["recall"]
    L.append(f"\n**(b) recall:** {tgr:.3f} vs m3_full {m3r:.3f} ({tgr-m3r:+.3f}). "
             f"trade-off: precision {tg['precision'][0]:.3f} (m3 {base['m3_full']['test']['precision']:.3f}), "
             f"wrong_merge {tg['wrong_merge_rate'][0]:.3f} (m3 {base['m3_full']['test']['wrong_merge_rate']:.3f}), "
             f"impossible {tg['impossible_transition_rate'][0]:.3f} "
             f"(m3 {base['m3_full']['test']['impossible_transition_rate']:.3f}).")

    L.append("\n**(c) 학습된 게이트 (dev cross-KG 후보, 쉬운 쌍 sim>0.5 vs 모호 sim≤0.5):**\n")
    L.append("| term | gate(전체) | gate(쉬운) | gate(모호) | 쉬운−모호 |")
    L.append("|---|---|---|---|---|")
    for t in TERMS:
        ga = st.fmean(gate_acc["all"][t]); ge = st.fmean(gate_acc["easy"][t])
        gm = st.fmean(gate_acc["ambig"][t])
        L.append(f"| {t} | {ga:.3f} | {ge:.3f} | {gm:.3f} | {ge-gm:+.3f} |")
    L.append("\n*해석: '쉬운−모호'가 음수면 쉬운 쌍에서 해당 항을 더 끔(가설 부합). "
             "sem이 높게 유지되고 제약 항이 낮으면 '쉬운 쌍은 sem 위주' 메커니즘 증거.*")
    L.append("\n**(c-보충) per-source-pair precision:** metrics에 없음 → 측정 안 됨.")
    L.append("\n> 시도한 게이팅 변형: 1개(6항 전부 게이트, 근사-항등 초기화, 11차원 쌍맥락 MLP). "
             "설계·하이퍼파라미터는 사전 고정(dev 튜닝 없음). 단일 변형이라 dev 선택은 자명.")
    L.append("> 통계: n=3 저검정력. ΔF1·std와 함께 해석.")
    return "\n".join(L)


def plot(out_dir, cell, base, gate_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # F1 comparison bar
    models = ["term_gating"] + BASE_VARIANTS
    f1s = [cell("test", "f1")[0]] + [base[v]["test"]["f1"] for v in BASE_VARIANTS]
    errs = [cell("test", "f1")[1]] + [0] * len(BASE_VARIANTS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:green"] + ["tab:gray"] * len(BASE_VARIANTS)
    ax.bar(models, f1s, yerr=errs, capsize=4, color=colors, alpha=0.85)
    ax.axhline(base["m3_full"]["test"]["f1"], ls="--", color="tab:red", alpha=0.6,
               label=f"m3_full={base['m3_full']['test']['f1']:.3f}")
    ax.axhline(base["no_motion"]["test"]["f1"], ls=":", color="tab:blue", alpha=0.6,
               label=f"no_motion={base['no_motion']['test']['f1']:.3f}")
    for i, (v, e) in enumerate(zip(f1s, errs)):
        ax.text(i, v + e + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("test F1"); ax.set_ylim(0, max(f1s) * 1.18)
    ax.set_title("Term gating vs recorded ablation variants (test F1, default decode, n=3)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "term_gating_f1.png", dpi=130); plt.close(fig)

    # gate by difficulty
    import numpy as np
    x = np.arange(len(TERMS)); w = 0.35
    ge = [st.fmean(gate_acc["easy"][t]) for t in TERMS]
    gm = [st.fmean(gate_acc["ambig"][t]) for t in TERMS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, ge, w, label="easy (sim>0.5)", color="tab:green", alpha=0.85)
    ax.bar(x + w / 2, gm, w, label="ambiguous (sim<=0.5)", color="tab:orange", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(TERMS); ax.set_ylabel("mean gate g_k")
    ax.set_title("Learned gate per term by pair difficulty (dev cross-KG candidates)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(fig_dir / "term_gating_gates.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
