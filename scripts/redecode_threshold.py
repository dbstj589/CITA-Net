#!/usr/bin/env python
"""∅-slot (null) threshold RE-DECODE sweep for CITA-Net m3_full -- NO RETRAIN.

Loads the already-trained m3_full checkpoints (runs/ablation/m3_full_seed*),
runs the forward pass ONCE per sector, then RE-DECODES the *same* model output
with decode_full() at several null_threshold values and re-scores via
evaluate_full(). Only the inference-time decode threshold changes; model weights
are untouched. (See decode.py:decode_full(null_threshold=0.5).)

Question (robustness sweep follow-up): is m3_full's lower recall a decode
conservatism artifact that a different ∅ threshold can recover, without retraining?

We deliberately sweep BOTH below 0.5 (the spec grid 0.4/0.35/0.3/0.25) AND above
(0.6..0.9) so the empirical recall direction is measured, not assumed -- because
decode.py:196 routes an entity to ∅ when  `argmax==null_slot  OR  null_mass>thr`.
The argmax-null clause is independent of `thr`, so the *measured* effect of `thr`
is reported as fact below, not predicted.

Baselines (no_motion / no_time / m3_full@0.5) are read from the ACTUAL recorded
results/ablation/results.jsonl -- no hardcoded numbers.

    python scripts/redecode_threshold.py            # D0 (frozen suite), test+dev
    python scripts/redecode_threshold.py --tag noise2x \
        --runs-dir runs/robustness --variant-suffix _m3_full \
        --data-root data/robustness/noise2x \
        --ontology-dir data/robustness/noise2x/ontology   # step 3
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

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "ablation" / "m3_full.yaml"
DEFAULT_SEEDS = [19521006, 19521007, 19521008]
DEFAULT_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BASELINE_VARIANTS = ["m3_full", "no_motion", "no_time"]
ABL_JSONL = REPO / "results" / "ablation" / "results.jsonl"

METRICS = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
           "impossible_transition_rate", "dangling_precision", "dangling_recall"]


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    return (st.fmean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


def read_ablation_baselines():
    """mean F1 (n seeds) per baseline variant from the recorded ablation sweep."""
    if not ABL_JSONL.exists():
        return {}
    rows = [json.loads(l) for l in ABL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for v in BASELINE_VARIANTS:
        vr = [r for r in rows if r.get("variant") == v]
        if not vr:
            continue
        out[v] = {}
        for split in ("dev", "test"):
            for met in ("precision", "recall", "f1"):
                m, s = mean_std([r[split][met] for r in vr if split in r])
                out[v][f"{split}_{met}"] = m
                out[v][f"{split}_{met}_std"] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--runs-dir", default=str(REPO / "runs" / "ablation"),
                    help="dir holding <variant><suffix>_seed<seed> checkpoints")
    ap.add_argument("--variant", default="m3_full")
    ap.add_argument("--variant-suffix", default="",
                    help="e.g. '_m3_full' for runs/robustness/noise2x_m3_full_seed*")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--data-root", default=None, help="override cfg.data_root")
    ap.add_argument("--ontology-dir", default=None, help="override cfg.ontology_dir")
    ap.add_argument("--tag", default="D0", help="condition label for outputs")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "threshold_sweep"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.data_root:
        cfg.data_root = args.data_root
    if args.ontology_dir:
        cfg.ontology_dir = args.ontology_dir

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(set(args.thresholds))
    base_thr = 0.5 if 0.5 in thresholds else min(thresholds, key=lambda t: abs(t - 0.5))

    print(f"== ∅-threshold re-decode sweep (NO retrain) :: tag={args.tag} ==")
    print(f"   variant={args.variant}{args.variant_suffix}  seeds={args.seeds}")
    print(f"   data_root={cfg.data_root}")
    print(f"   thresholds={thresholds}  (baseline={base_thr})")

    # rows: one per (seed, threshold, split) holding the aggregated metric dict
    rows = []
    for seed in args.seeds:
        run_dir = runs_dir / f"{args.variant}{args.variant_suffix}_seed{seed}"
        if not (run_dir / "model.pt").exists():
            raise SystemExit(f"missing checkpoint: {run_dir/'model.pt'} -- aborting (no retrain).")
        model, fs, ont = load_large_bundle(cfg, run_dir)
        print(f"\n[seed {seed}] loaded {run_dir.relative_to(REPO)}")
        for split in ("dev", "test"):
            sids = read_split(cfg.data_root, split)
            # forward ONCE per sector, cache (out, feats, sd); re-decode per threshold
            cache = []
            for sid in sids:
                sd = Path(cfg.data_root) / "sectors" / sid
                feats = featurize_sector(sd, fs, ont, cfg)
                with torch.no_grad():
                    out = model(feats)
                cache.append((out, feats, sd))
            for thr in thresholds:
                per = []
                for out, feats, sd in cache:
                    res = decode_full(out, feats, ont, null_threshold=thr)
                    per.append(evaluate_full(res, feats, sd))
                agg = aggregate(per)
                rows.append({"tag": args.tag, "variant": args.variant, "seed": seed,
                             "threshold": thr, "split": split, **agg})
            r05 = next(r for r in rows if r["seed"] == seed and r["split"] == split
                       and r["threshold"] == base_thr)
            print(f"   {split}: @{base_thr} P={r05['precision']:.3f} "
                  f"R={r05['recall']:.3f} F1={r05['f1']:.3f}")
            del cache

    # ---- results.jsonl ----
    jsonl = out_dir / f"results_{args.tag}.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- aggregate across seeds: per (split, threshold) mean+/-std ----
    def cell(split, thr, met):
        vals = [r[met] for r in rows if r["split"] == split and r["threshold"] == thr
                and met in r]
        return mean_std(vals)

    baselines = read_ablation_baselines()

    for split in ("dev", "test"):
        csv_path = out_dir / f"summary_{split}_{args.tag}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            head = ["threshold"]
            for m in METRICS:
                head += [f"{m}_mean", f"{m}_std"]
            w.writerow(head)
            for thr in thresholds:
                row = [thr]
                for m in METRICS:
                    mu, sd = cell(split, thr, m)
                    row += [round(mu, 4), round(sd, 4)]
                w.writerow(row)

    # ---- markdown analysis (answers a-d) ----
    md = analysis_md(args.tag, thresholds, base_thr, cell, baselines)
    (out_dir / f"threshold_analysis_{args.tag}.md").write_text(md, encoding="utf-8")

    # ---- figures ----
    try:
        plot(out_dir, args.tag, thresholds, cell, baselines)
        fig_note = f"figures -> {out_dir/'figures'}"
    except Exception as e:  # noqa: BLE001
        fig_note = f"(no figures: {e})"

    print("\n" + md)
    print(f"\nwrote -> {jsonl.name}, summary_*_{args.tag}.csv, "
          f"threshold_analysis_{args.tag}.md")
    print(fig_note)


def analysis_md(tag, thresholds, base_thr, cell, baselines):
    L = [f"# ∅-슬롯(null) 임계 재디코드 스윕 — {tag} (재학습 없음)\n",
         "m3_full 3시드 학습 체크포인트를 그대로 로드, **디코드 임계만** 바꿔 재추론.",
         "임계를 0.5 아래(과보수 강화)·위(완화 시도) 양방향으로 측정 — recall 방향은 "
         "예측이 아니라 실측. (decode.py:196 `argmax==null OR null_mass>thr`)\n",
         "> ∅ 질량이 임계를 넘으면 dangling 기권. **임계↓ = 더 많이 기권(보수적)**, "
         "임계↑ = 기권 완화 시도.\n"]
    nm = baselines.get("no_motion", {})
    nt = baselines.get("no_time", {})
    for split in ("dev", "test"):
        L.append(f"\n## {split} — 임계별 지표 (m3_full, mean±std, n=3)\n")
        L.append("| thr | precision | recall | f1 | wrong_merge | frag | impossible | dang_P | dang_R |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for thr in thresholds:
            c = {m: cell(split, thr, m) for m in METRICS}
            star = " ⟵base" if thr == base_thr else ""
            L.append("| {:.2f}{} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | "
                     "{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                         thr, star,
                         *c["precision"], *c["recall"], *c["f1"],
                         c["wrong_merge_rate"][0], c["fragmentation_rate"][0],
                         c["impossible_transition_rate"][0],
                         c["dangling_precision"][0], c["dangling_recall"][0]))
        nmf = nm.get(f"{split}_f1"); ntf = nt.get(f"{split}_f1")
        if nmf is not None:
            L.append(f"\n*baseline(실측 ablation): no_motion {split} F1={nmf:.3f}, "
                     f"no_time F1={ntf:.3f}.*")

    # ---- answers on test (report split); threshold chosen on dev (no leakage) ----
    L.append("\n## 핵심 질문 (test로 보고, 임계는 dev로 선택)\n")
    r_base = cell("test", base_thr, "recall")[0]
    f_base = cell("test", base_thr, "f1")[0]
    p_base = cell("test", base_thr, "precision")[0]
    lows = [t for t in thresholds if t < base_thr]
    highs = [t for t in thresholds if t > base_thr]

    # (a) does lowering raise recall?
    L.append("**(a) 임계를 낮추면 recall/F1이 오르는가?**")
    if lows:
        rl = cell("test", min(lows), "recall")[0]
        fl = cell("test", min(lows), "f1")[0]
        dirn = "오른다" if rl > r_base + 1e-9 else ("내려간다" if rl < r_base - 1e-9 else "변화 없음")
        L.append(f"- 임계 {base_thr:.2f}→{min(lows):.2f}: recall {r_base:.3f}→{rl:.3f} ({dirn}), "
                 f"F1 {f_base:.3f}→{fl:.3f}. → 낮출수록 recall은 **{dirn}**.")
    if highs:
        rh = cell("test", max(highs), "recall")[0]
        fh = cell("test", max(highs), "f1")[0]
        same = abs(rh - r_base) < 1e-6
        L.append(f"- 임계 {base_thr:.2f}→{max(highs):.2f}(완화): recall {r_base:.3f}→{rh:.3f}, "
                 f"F1 {f_base:.3f}→{fh:.3f}. "
                 f"{'→ 변화 없음(argmax==null 바닥조건 때문).' if same else ''}")

    # (b) does any threshold reach no_motion F1?
    L.append("\n**(b) m3_full F1이 no_motion을 따라잡는 임계가 있는가?**")
    nmf = nm.get("test_f1")
    if nmf is not None:
        reach = [t for t in thresholds if cell("test", t, "f1")[0] >= nmf - 1e-9]
        if reach:
            L.append(f"- no_motion test F1={nmf:.3f} 이상을 내는 임계: "
                     f"{', '.join(f'{t:.2f}' for t in reach)}.")
        else:
            best_t = max(thresholds, key=lambda t: cell("test", t, "f1")[0])
            L.append(f"- **따라잡지 못함.** 최고 F1은 임계 {best_t:.2f}의 "
                     f"{cell('test',best_t,'f1')[0]:.3f} (< no_motion {nmf:.3f}).")

    # (c) trade-off cost
    L.append("\n**(c) recall을 얻는 대신 잃는 것 (trade-off):**")
    if lows:
        t = min(lows)
        L.append(f"- 임계 {t:.2f}: precision {cell('test',t,'precision')[0]:.3f} "
                 f"(base {p_base:.3f}), wrong_merge {cell('test',t,'wrong_merge_rate')[0]:.3f} "
                 f"(base {cell('test',base_thr,'wrong_merge_rate')[0]:.3f}), "
                 f"impossible {cell('test',t,'impossible_transition_rate')[0]:.3f} "
                 f"(base {cell('test',base_thr,'impossible_transition_rate')[0]:.3f}).")

    # (d) F1-max threshold chosen on dev, reported on test
    best_dev = max(thresholds, key=lambda t: cell("dev", t, "f1")[0])
    L.append(f"\n**(d) F1 최대 임계(dev 기준 선택): {best_dev:.2f}** — "
             f"dev F1={cell('dev',best_dev,'f1')[0]:.3f}, "
             f"해당 임계 test F1={cell('test',best_dev,'f1')[0]:.3f} "
             f"(test P={cell('test',best_dev,'precision')[0]:.3f}, "
             f"R={cell('test',best_dev,'recall')[0]:.3f}).")
    return "\n".join(L)


def plot(out_dir, tag, thresholds, cell, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for split in ("dev", "test"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for met, c in (("precision", "tab:blue"), ("recall", "tab:green"), ("f1", "tab:red")):
            mu = [cell(split, t, met)[0] for t in thresholds]
            sd = [cell(split, t, met)[1] for t in thresholds]
            ax.errorbar(thresholds, mu, yerr=sd, marker="o", capsize=3, color=c, label=met)
        nm = baselines.get("no_motion", {}).get(f"{split}_f1")
        nt = baselines.get("no_time", {}).get(f"{split}_f1")
        if nm is not None:
            ax.axhline(nm, ls="--", color="tab:red", alpha=0.5, label=f"no_motion F1={nm:.3f}")
        if nt is not None:
            ax.axhline(nt, ls=":", color="tab:orange", alpha=0.7, label=f"no_time F1={nt:.3f}")
        ax.axvline(0.5, ls="-", color="gray", alpha=0.3)
        ax.set_xlabel("null-slot threshold  (lower = more abstain  |  higher = looser)")
        ax.set_ylabel("score")
        ax.set_title(f"m3_full null-threshold re-decode -- {tag} / {split} (n=3)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"threshold_{tag}_{split}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
