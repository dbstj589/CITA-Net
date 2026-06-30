#!/usr/bin/env python
"""Alternative decoder: REVIVE argmax-∅ entities by a margin rule -- NO RETRAIN.

Background (continues the null-threshold experiment): m3_full's low recall is NOT
fixable via decode_full's null_threshold, because once null mass > 0.5 the argmax
is mathematically the ∅ slot, so the true gate is "Sinkhorn routed the entity's
argmax to the ∅ slot" (decode.py:196 `s == K`).

Hypothesis tested here: some argmax-∅ entities are *near-miss abstentions* whose
best REAL slot mass is very close to their ∅ mass. Reviving those (post-hoc, model
weights + Sinkhorn assignment untouched) should lift recall.

This file does NOT modify decode.py. It reimplements decode_full as
`decode_full_revive(..., margin)`, reusing decode.py's helpers/dataclasses, and
adds ONE rule to the routing block:

    original: argmax==∅ (s==K) OR null_mass>null_threshold  -> abstain
    added   : if argmax==∅ AND (null_mass - top_real_mass) < margin
              -> revive into that top real slot (cancel abstain)

margin=0 reproduces decode_full exactly (since null_mass-top_real_mass >= 0 when
argmax is ∅, so `< 0` never fires) -- verified against the recorded ablation
numbers before any interpretation.

    python scripts/redecode_revive.py                     # D0, dev+test
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.decode import (FullDecodeResult, FullIdentity, Transition,
                            _entity_obs, _pair_lookup)
from citanet.data.kinematics import feasibility
from citanet.engine_large import load_large_bundle
from citanet.eval import aggregate, evaluate_full
from citanet.data.stream import featurize_sector, read_split

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "ablation" / "m3_full.yaml"
DEFAULT_SEEDS = [19521006, 19521007, 19521008]
DEFAULT_MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
BASELINE_VARIANTS = ["m3_full", "no_motion", "no_time"]
ABL_JSONL = REPO / "results" / "ablation" / "results.jsonl"

METRICS = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
           "impossible_transition_rate", "dangling_precision", "dangling_recall"]


# ===========================================================================
# Alternative decoder = decode.py:decode_full with the revive rule added.
# Body copied faithfully from decode_full; ONLY the routing block differs
# (marked REVIVE). decode.py itself is left untouched.
# ===========================================================================
def decode_full_revive(out, feats, ontology, null_threshold: float = 0.5,
                       margin: float = 0.0):
    assert out.assign is not None, "decode_full requires the M3 decoder (assign)"
    A = out.assign.detach()
    M, S = A.shape
    K = S - 1
    obs = feats.obs
    p_same = torch.sigmoid(out.pair_logits.detach())
    plut = _pair_lookup(feats)
    cta = out.cta

    groups = _entity_obs(feats)
    ent_slot: dict[str, int] = {}
    ent_nullmass: dict[str, float] = {}
    ent_mass: dict[str, torch.Tensor] = {}
    for key, members in groups.items():
        mean_assign = A[members].mean(dim=0)
        ent_mass[key] = mean_assign
        ent_slot[key] = int(mean_assign.argmax())
        ent_nullmass[key] = float(mean_assign[K])

    # --- routing (REVIVE rule added) ---
    slot_entities: dict[int, list[str]] = {}
    dangling_keys: list[str] = []
    n_revived = 0
    for key in groups:
        s = ent_slot[key]
        nullmass = ent_nullmass[key]
        real = ent_mass[key][:K]                      # real-slot masses
        top_real_slot = int(real.argmax())
        top_real_mass = float(real[top_real_slot])
        original_dangling = (s == K) or (nullmass > null_threshold)
        # REVIVE: argmax is ∅ but a real slot is within `margin` of ∅ mass
        if original_dangling and s == K and (nullmass - top_real_mass) < margin:
            slot_entities.setdefault(top_real_slot, []).append(key)
            n_revived += 1
        elif original_dangling:
            dangling_keys.append(key)
        else:
            slot_entities.setdefault(s, []).append(key)

    ent_of = feats.entity_of
    cross_by_entpair: dict[tuple[str, str], list[int]] = {}
    ent_best_cross: dict[str, tuple[float, str]] = {}
    for q, p in enumerate(feats.pairs):
        if not feats.pair_cross[q]:
            continue
        ka, kb = ent_of[obs[p.i].obs_id], ent_of[obs[p.j].obs_id]
        a_key, b_key = (ka, kb) if ka.startswith("A:") else (kb, ka)
        cross_by_entpair.setdefault((a_key, b_key), []).append(q)
        pv = float(p_same[q])
        for me, other in ((ka, kb), (kb, ka)):
            if pv > ent_best_cross.get(me, (-1.0, ""))[0]:
                ent_best_cross[me] = (pv, other)

    identities: list[FullIdentity] = []
    for slot, keys in sorted(slot_entities.items()):
        a_keys = [k for k in keys if k.startswith("A:")]
        b_keys = [k for k in keys if k.startswith("B:")]

        def best(keys_):
            return max(keys_, key=lambda k: float(ent_mass[k][slot])) if keys_ else None
        a_best, b_best = best(a_keys), best(b_keys)
        conflicts = []
        for k in a_keys + b_keys:
            if k not in (a_best, b_best):
                conflicts.append({"local_entity_id": k.split(":", 1)[1],
                                  "kg_id": k.split(":", 1)[0],
                                  "reason": "extra same-KG entity in slot (possible merge)"})
        local_ids = {}
        if a_best:
            local_ids["A"] = a_best.split(":", 1)[1]
        if b_best:
            local_ids["B"] = b_best.split(":", 1)[1]

        member_idx = [m for k in keys for m in groups[k]]
        member_idx.sort(key=lambda m: obs[m].t)
        members = [(obs[m].obs_id, float(A[m, slot])) for m in member_idx]

        types = [obs[m].type for m in member_idx if obs[m].type != "UNKNOWN"]
        typ = max(set(types), key=types.count) if types else "UNKNOWN"
        tconf = float(sum(obs[m].type_confidence for m in member_idx) / len(member_idx))

        traj = [{"obs_id": obs[m].obs_id, "time": obs[m].time_iso,
                 "easting": obs[m].easting, "northing": obs[m].northing,
                 "state": obs[m].state, "kg_id": obs[m].kg_id,
                 "source": obs[m].source} for m in member_idx]

        transitions = []
        for m1, m2 in zip(member_idx, member_idx[1:]):
            o1, o2 = obs[m1], obs[m2]
            feas = feasibility(o1, o2, ontology)
            q = plut.get(frozenset((o1.obs_id, o2.obs_id)))
            prob = float(cta.p_transition.detach()[q]) if q is not None else \
                (1.0 if feas.feasible else 0.0)
            transitions.append(Transition(
                from_obs=o1.obs_id, to_obs=o2.obs_id, prob=prob,
                feasible_motion=bool(feas.feasible),
                required_speed_mps=round(feas.required_speed_mps, 3)))

        cross_qs = cross_by_entpair.get((a_best, b_best), []) if (a_best and b_best) else []
        def cmean(t):
            return float(t.detach()[cross_qs].mean()) if cross_qs else 0.0
        rationale = {"sim_sem": cmean(cta.sim_sem), "b_time": cmean(cta.b_time),
                     "b_motion": cmean(cta.b_motion), "b_state": cmean(cta.b_state),
                     "b_rel": cmean(cta.b_rel), "b_src": cmean(cta.b_src)}
        match_conf = float(p_same[cross_qs].mean()) if cross_qs else float(A[member_idx, slot].mean())

        if member_idx:
            fe = sum(obs[m].easting for m in member_idx) / len(member_idx)
            fn = sum(obs[m].northing for m in member_idx) / len(member_idx)
            rep = {"type": typ, "easting": round(fe, 2), "northing": round(fn, 2),
                   "last_state": obs[member_idx[-1]].state,
                   "last_time": obs[member_idx[-1]].time_iso}
        else:
            rep = {}

        identities.append(FullIdentity(
            slot=slot, type=typ, type_confidence=round(tconf, 3),
            match_confidence=round(match_conf, 3), local_entity_ids=local_ids,
            member_obs=members, trajectory=traj, transitions=transitions,
            rationale={k: round(v, 4) for k, v in rationale.items()},
            fused_representative=rep, conflicts=conflicts))

    p_dang = torch.sigmoid(out.dangling_logits.detach())
    dangling = []
    for key in dangling_keys:
        kg, local = key.split(":", 1)
        members = groups[key]
        d_prob = max(ent_nullmass[key], float(p_dang[members].mean()))
        nearest = None
        if key in ent_best_cross:
            bp, other_key = ent_best_cross[key]
            okg, olocal = other_key.split(":", 1)
            nearest = {"kg_id": okg, "local_entity_id": olocal, "pair_prob": round(bp, 3)}
        dangling.append({
            "kg_id": kg, "local_entity_id": local,
            "obs_ids": [obs[m].obs_id for m in members],
            "dangling_prob": round(d_prob, 3),
            "nearest_candidate": nearest,
            "decision": "abstain",
            "reason": ("routed to null slot" if ent_slot[key] == K
                       else "weak cross-KG support / high dangling probability"),
        })
    res = FullDecodeResult(identities=identities, dangling=dangling)
    res.n_revived = n_revived          # diagnostic (dynamic attr; not in baseline)
    return res


# ===========================================================================
# Harness: load m3_full checkpoints, forward once per sector, re-decode per margin
# ===========================================================================
def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    return (st.fmean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


def read_ablation_baselines(jsonl_path=ABL_JSONL, dataset=None):
    """Recorded baselines. ablation jsonl (dataset=None) gives D0; robustness jsonl
    with dataset='noise2x'/'dang035' gives the condition-specific recorded numbers."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return {}
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if dataset is not None:
        rows = [r for r in rows if r.get("dataset") == dataset]
    out = {}
    for v in BASELINE_VARIANTS:
        vr = [r for r in rows if r.get("variant") == v]
        if not vr:
            continue
        out[v] = {}
        for split in ("dev", "test"):
            for met in ("precision", "recall", "f1"):
                m, _ = mean_std([r[split][met] for r in vr if split in r])
                out[v][f"{split}_{met}"] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--runs-dir", default=str(REPO / "runs" / "ablation"))
    ap.add_argument("--variant", default="m3_full")
    ap.add_argument("--variant-suffix", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--margins", type=float, nargs="+", default=DEFAULT_MARGINS)
    ap.add_argument("--null-threshold", type=float, default=0.5)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ontology-dir", default=None)
    ap.add_argument("--tag", default="D0")
    ap.add_argument("--baseline-jsonl", default=str(ABL_JSONL),
                    help="recorded results for fidelity + no_motion/no_time lines")
    ap.add_argument("--baseline-dataset", default=None,
                    help="filter baseline rows by dataset (e.g. noise2x, dang035)")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "revive_decode"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.data_root:
        cfg.data_root = args.data_root
    if args.ontology_dir:
        cfg.ontology_dir = args.ontology_dir

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = REPO / runs_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    margins = sorted(set(args.margins))
    if 0.0 not in margins:
        margins = [0.0] + margins

    print(f"== argmax-∅ REVIVE re-decode (NO retrain) :: tag={args.tag} ==")
    print(f"   variant={args.variant}{args.variant_suffix}  seeds={args.seeds}")
    print(f"   data_root={cfg.data_root}  null_threshold={args.null_threshold}")
    print(f"   margins={margins}")

    rows = []
    for seed in args.seeds:
        run_dir = runs_dir / f"{args.variant}{args.variant_suffix}_seed{seed}"
        if not (run_dir / "model.pt").exists():
            raise SystemExit(f"missing checkpoint: {run_dir/'model.pt'} -- aborting (no retrain).")
        model, fs, ont = load_large_bundle(cfg, run_dir)
        print(f"\n[seed {seed}] loaded {run_dir.relative_to(REPO)}")
        for split in ("dev", "test"):
            sids = read_split(cfg.data_root, split)
            cache = []
            for sid in sids:
                sd = Path(cfg.data_root) / "sectors" / sid
                feats = featurize_sector(sd, fs, ont, cfg)
                with torch.no_grad():
                    out = model(feats)
                cache.append((out, feats, sd))
            for mg in margins:
                per, revived = [], 0
                for out, feats, sd in cache:
                    res = decode_full_revive(out, feats, ont,
                                             null_threshold=args.null_threshold, margin=mg)
                    revived += getattr(res, "n_revived", 0)
                    per.append(evaluate_full(res, feats, sd))
                agg = aggregate(per)
                rows.append({"tag": args.tag, "variant": args.variant, "seed": seed,
                             "margin": mg, "split": split, "n_revived": revived, **agg})
            r0 = next(r for r in rows if r["seed"] == seed and r["split"] == split
                      and r["margin"] == 0.0)
            print(f"   {split}: margin0 P={r0['precision']:.3f} R={r0['recall']:.3f} "
                  f"F1={r0['f1']:.3f}")
            del cache

    jsonl = out_dir / f"results_{args.tag}.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def cell(split, mg, met):
        vals = [r[met] for r in rows if r["split"] == split and r["margin"] == mg and met in r]
        return mean_std(vals)

    for split in ("dev", "test"):
        with open(out_dir / f"summary_{split}_{args.tag}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            head = ["margin", "n_revived"]
            for m in METRICS:
                head += [f"{m}_mean", f"{m}_std"]
            w.writerow(head)
            for mg in margins:
                row = [mg, round(cell(split, mg, "n_revived")[0], 1)]
                for m in METRICS:
                    mu, sd = cell(split, mg, m)
                    row += [round(mu, 4), round(sd, 4)]
                w.writerow(row)

    baselines = read_ablation_baselines(args.baseline_jsonl, args.baseline_dataset)

    # fidelity gate: margin=0 must reproduce the recorded m3_full numbers
    fid = fidelity_check(cell, baselines)

    md = analysis_md(args.tag, margins, cell, baselines, fid)
    (out_dir / f"revive_analysis_{args.tag}.md").write_text(md, encoding="utf-8")

    try:
        plot(out_dir, args.tag, margins, cell, baselines)
        fig_note = f"figures -> {out_dir/'figures'}"
    except Exception as e:  # noqa: BLE001
        fig_note = f"(no figures: {e})"

    print("\n" + md)
    print(f"\nwrote -> {jsonl.name}, summary_*_{args.tag}.csv, revive_analysis_{args.tag}.md")
    print(fig_note)


def fidelity_check(cell, baselines):
    """margin=0 vs recorded m3_full ablation (test P~0.874 R~0.524 F1~0.652)."""
    base = baselines.get("m3_full", {})
    out = {"ok": True, "lines": []}
    for split in ("dev", "test"):
        for met in ("precision", "recall", "f1"):
            got = cell(split, 0.0, met)[0]
            ref = base.get(f"{split}_{met}")
            if ref is None:
                continue
            d = abs(got - ref)
            ok = d < 5e-3
            out["ok"] &= ok
            if split == "test":
                out["lines"].append(
                    f"  {split} {met}: margin0={got:.4f} vs ablation={ref:.4f} "
                    f"(Δ={d:.4f}) {'OK' if ok else 'MISMATCH'}")
    return out


def analysis_md(tag, margins, cell, baselines, fid):
    nm = baselines.get("no_motion", {})
    nt = baselines.get("no_time", {})
    L = [f"# argmax-∅ 되살리기(revive) 디코더 — {tag} (재학습 없음)\n",
         "m3_full 3시드 체크포인트 로드, Sinkhorn 배정 불변. argmax가 ∅인 엔티티 중 "
         "(∅질량 − 최고실슬롯질량) < margin 인 것만 그 실슬롯으로 되살림.\n",
         f"**충실성 검증 (margin=0 == baseline):** {'통과' if fid['ok'] else '실패'}"]
    L += fid["lines"]
    L.append("")
    if not fid["ok"]:
        L.append("> ⚠️ 충실성 검증 실패 — 아래 수치는 해석 금지.\n")
    for split in ("dev", "test"):
        L.append(f"\n## {split} — margin별 지표 (m3_full, mean±std, n=3)\n")
        L.append("| margin | n_revived | precision | recall | f1 | wrong_merge | frag | impossible | dang_P | dang_R |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for mg in margins:
            c = {m: cell(split, mg, m) for m in METRICS}
            nrev = cell(split, mg, "n_revived")[0]
            tag0 = " ⟵base" if mg == 0.0 else ""
            L.append("| {:.2f}{} | {:.1f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | "
                     "{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                         mg, tag0, nrev, *c["precision"], *c["recall"], *c["f1"],
                         c["wrong_merge_rate"][0], c["fragmentation_rate"][0],
                         c["impossible_transition_rate"][0],
                         c["dangling_precision"][0], c["dangling_recall"][0]))
        nmf, ntf = nm.get(f"{split}_f1"), nt.get(f"{split}_f1")
        if nmf is not None:
            L.append(f"\n*baseline(실측): no_motion {split} F1={nmf:.3f}, no_time F1={ntf:.3f}.*")

    L.append("\n## 핵심 질문 (test로 보고, margin은 dev로 선택)\n")
    r0 = cell("test", 0.0, "recall")[0]; f0 = cell("test", 0.0, "f1")[0]
    p0 = cell("test", 0.0, "precision")[0]
    biggest = max(margins)
    rb = cell("test", biggest, "recall")[0]; fb = cell("test", biggest, "f1")[0]

    L.append("**(a) margin을 키우면 recall/F1이 오르는가?**")
    rdir = "오른다" if rb > r0 + 1e-9 else ("내려간다" if rb < r0 - 1e-9 else "변화 없음")
    fdir = "오른다" if fb > f0 + 1e-9 else ("내려간다" if fb < f0 - 1e-9 else "변화 없음")
    L.append(f"- margin 0→{biggest:.2f}: recall {r0:.3f}→{rb:.3f} (**{rdir}**), "
             f"F1 {f0:.3f}→{fb:.3f} (**{fdir}**). "
             f"되살린 매칭 평균 {cell('test',biggest,'n_revived')[0]:.1f}건.")

    nmf = nm.get("test_f1")
    L.append("\n**(b) m3_full F1이 no_motion을 따라잡는 margin이 있는가?**")
    if nmf is not None:
        reach = [m for m in margins if cell("test", m, "f1")[0] >= nmf - 1e-9]
        if reach:
            L.append(f"- no_motion test F1={nmf:.3f} 이상: margin "
                     f"{', '.join(f'{m:.2f}' for m in reach)}.")
        else:
            bm = max(margins, key=lambda m: cell("test", m, "f1")[0])
            L.append(f"- **따라잡지 못함.** 최고 test F1 = margin {bm:.2f}의 "
                     f"{cell('test',bm,'f1')[0]:.3f} (< no_motion {nmf:.3f}).")

    L.append("\n**(c) recall을 얻는 대가 (trade-off):**")
    L.append(f"- margin {biggest:.2f}: precision {p0:.3f}→{cell('test',biggest,'precision')[0]:.3f}, "
             f"wrong_merge {cell('test',0.0,'wrong_merge_rate')[0]:.3f}→"
             f"{cell('test',biggest,'wrong_merge_rate')[0]:.3f}, "
             f"impossible {cell('test',0.0,'impossible_transition_rate')[0]:.3f}→"
             f"{cell('test',biggest,'impossible_transition_rate')[0]:.3f}.")

    best_dev = max(margins, key=lambda m: cell("dev", m, "f1")[0])
    L.append(f"\n**(d) F1 최대 margin(dev 기준): {best_dev:.2f}** — dev F1="
             f"{cell('dev',best_dev,'f1')[0]:.3f}, 해당 margin test F1="
             f"{cell('test',best_dev,'f1')[0]:.3f} "
             f"(test P={cell('test',best_dev,'precision')[0]:.3f}, "
             f"R={cell('test',best_dev,'recall')[0]:.3f}).")
    if nmf is not None:
        verdict = ("**no_motion 교차 성공**" if cell("test", best_dev, "f1")[0] >= nmf - 1e-9
                   else "**no_motion 교차 실패**")
        L.append(f"\n→ {verdict} (dev 선택 margin {best_dev:.2f}의 test F1 vs no_motion {nmf:.3f}).")
    return "\n".join(L)


def plot(out_dir, tag, margins, cell, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for split in ("dev", "test"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for met, c in (("precision", "tab:blue"), ("recall", "tab:green"), ("f1", "tab:red")):
            mu = [cell(split, m, met)[0] for m in margins]
            sd = [cell(split, m, met)[1] for m in margins]
            ax.errorbar(margins, mu, yerr=sd, marker="o", capsize=3, color=c, label=met)
        nm = baselines.get("no_motion", {}).get(f"{split}_f1")
        nt = baselines.get("no_time", {}).get(f"{split}_f1")
        if nm is not None:
            ax.axhline(nm, ls="--", color="tab:red", alpha=0.5, label=f"no_motion F1={nm:.3f}")
        if nt is not None:
            ax.axhline(nt, ls=":", color="tab:orange", alpha=0.7, label=f"no_time F1={nt:.3f}")
        ax.set_xlabel("revive margin  (0 = baseline decode_full)")
        ax.set_ylabel("score")
        ax.set_title(f"m3_full argmax-null revive -- {tag} / {split} (n=3)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"revive_{tag}_{split}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
