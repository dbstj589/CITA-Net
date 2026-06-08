#!/usr/bin/env python
"""M3 ablation: CTA b_motion (kinematic feasibility) ON vs OFF.

Trains the full model with and without the b_motion term and compares the
decoded trajectories on the frozen dev+test set. Removing b_motion should let
the decoder link kinematically impossible transitions (and merge incompatible
observations) more often -> higher Impossible Transition Rate and Wrong-Merge
Rate.

    python scripts/ablation_m3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.engine import evaluate_full_split, featurize_split, read_split, save_bundle, train

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ["f1", "precision", "recall", "wrong_merge_rate", "fragmentation_rate",
           "trajectory_consistency_rate", "impossible_transition_rate",
           "dangling_precision", "dangling_recall",
           "n_infeasible_pairs", "infeasible_mean_p_transition",
           "infeasible_admission_rate"]


def run(config_name: str, reach_extra_m: float = 6000.0) -> dict:
    cfg = load_config(REPO_ROOT / "configs" / config_name)
    # RELAX blocking (data unchanged): let kinematically-impossible pairs -- in
    # particular any pairing of a towed weapon (v_max=0) across distance --
    # survive to the CTA, so the soft b_motion term (not the hard reach gate) is
    # what must reject them. Applied identically to both ablation arms.
    cfg.blocking.reach_extra_m = reach_extra_m
    print(f"\n== training {cfg.stage} (b_motion "
          f"{'ON' if 'motion' in cfg.cta.enabled_terms else 'OFF'}; "
          f"relaxed blocking +{reach_extra_m:.0f} m) ==")
    bundle = train(cfg, verbose=False)
    save_bundle(bundle, REPO_ROOT / "runs" / cfg.stage)

    rows = {}
    for split in ("dev", "test"):
        feats = bundle.dev_feats if split == "dev" else \
            featurize_split(cfg, bundle.fs, bundle.ontology, "test")
        ids = read_split(cfg.data_root, split)
        agg, per = evaluate_full_split(cfg, bundle.model, bundle.ontology, feats, ids)
        agg.update(impossible_pair_admission(bundle.model, feats))
        rows[split] = {"aggregate": agg, "per_scenario": per}
    return {"stage": cfg.stage, "splits": rows}


@torch.no_grad()
def impossible_pair_admission(model, feats_list) -> dict:
    """Direct b_motion diagnostic: over candidate pairs that are kinematically
    INFEASIBLE (required speed > feasible speed), what transition probability
    does the model assign? b_motion's job is to drive these toward 0. Reports
    the mean p_transition and the admission rate (p_transition > 0.5) on
    infeasible pairs."""
    model.eval()
    n_inf = 0
    sum_p = 0.0
    admitted = 0
    for f in feats_list:
        out = model(f)
        infeasible = f.p_req_speed > f.p_feas_speed
        if not bool(infeasible.any()):
            continue
        p = out.cta.p_transition[infeasible]
        n_inf += int(infeasible.sum())
        sum_p += float(p.sum())
        admitted += int((p > 0.5).sum())
    return {
        "n_infeasible_pairs": n_inf,
        "infeasible_mean_p_transition": (sum_p / n_inf) if n_inf else 0.0,
        "infeasible_admission_rate": (admitted / n_inf) if n_inf else 0.0,
    }


def main() -> None:
    on = run("cita_full.yaml")
    off = run("cita_full_nomotion.yaml")

    print("\n" + "=" * 64)
    print("M3 ABLATION -- b_motion ON vs OFF (dev+test macro avg)")
    print("=" * 64)
    print(f"  {'metric':30} {'b_motion ON':>12} {'b_motion OFF':>13}")
    for split in ("dev", "test"):
        print(f"\n[{split}]")
        a_on = on["splits"][split]["aggregate"]
        a_off = off["splits"][split]["aggregate"]
        for k in METRICS:
            mark = ""
            if k in ("impossible_transition_rate", "wrong_merge_rate",
                     "infeasible_mean_p_transition", "infeasible_admission_rate") \
                    and a_off[k] > a_on[k] + 1e-6:
                mark = "   <-- worse without b_motion"
            print(f"  {k:30} {a_on[k]:>12.4f} {a_off[k]:>13.4f}{mark}")

    out_dir = REPO_ROOT / "runs" / "ablation_m3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(
        json.dumps({"b_motion_on": on, "b_motion_off": off}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nsaved to {out_dir / 'ablation.json'}")


if __name__ == "__main__":
    main()
