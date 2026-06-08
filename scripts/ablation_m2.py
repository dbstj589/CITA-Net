#!/usr/bin/env python
"""M2 ablation: relation encoder ON vs OFF on the SAME data.

Trains two models that differ ONLY in the relation encoder (GNN + CTA b_rel):
  * cita_g        -- relation encoder ON
  * cita_g_norel  -- relation encoder OFF (ablation)
then evaluates both on the ambiguity probe(s) where the two T-72s overlap in
space and only relational context separates them. Reports the id_0003 hard
positive outcome (correct recovery vs wrong-merge / fragmentation) for each.

    python scripts/ablation_m2.py
"""
from __future__ import annotations

import json
from pathlib import Path

from citanet.config import load_config
from citanet.decode import decode_entities
from citanet.engine import (
    evaluate,
    featurize_split,
    read_split,
    save_bundle,
    scenario_dir,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# All held-out ambiguity probes (dev + test). The two T-72s are exchangeable on
# every non-relational channel, so a relation-off model can only guess; the
# id_0003 recovery RATE over many such probes is the robust ablation signal.
AMB_DEV = ["scn_amb01", "scn_amb02", "scn_amb03", "scn_amb04", "scn_amb05"]
AMB_TEST = ["scn_eamb01", "scn_eamb02", "scn_eamb03", "scn_eamb04"]
AMB_SCENARIOS = AMB_DEV + AMB_TEST


def id0003_diagnosis(model, feats, sdir: Path) -> dict:
    """Inspect what KG_A/KG_B id_0003 entities were matched to."""
    out = model(feats)
    res = decode_entities(out, feats)
    gold = json.loads((sdir / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
    g3 = next(i for i in gold["identities"] if i["gold_identity_id"] == "id_0003")
    a3, b3 = g3["member_local_entities"]["A"], g3["member_local_entities"]["B"]

    a_match = next((idn.b_entity for idn in res.identities if idn.a_entity == a3), None)
    b_match = next((idn.a_entity for idn in res.identities if idn.b_entity == b3), None)
    correct = (a_match == b3 and b_match == a3)
    # wrong-merge: a3 matched to a B entity that is NOT b3 (i.e. the lead tank)
    wrong_merge = a_match is not None and a_match != b3
    # fragmentation: a3 left unmatched (abstained) though it is a true match
    fragmented = a_match is None
    return {"a3": a3, "b3": b3, "a3_matched_to": a_match, "b3_matched_to": b_match,
            "id_0003_correct": correct, "id_0003_wrong_merge": wrong_merge,
            "id_0003_fragmented": fragmented}


def run_one(config_path: str) -> dict:
    cfg = load_config(config_path)
    print(f"\n===== training {cfg.stage} (relation encoder "
          f"{'ON' if cfg.graph.enabled else 'OFF'}) =====")
    bundle = train(cfg, verbose=True)
    save_bundle(bundle, REPO_ROOT / "runs" / cfg.stage)

    dev_ids = read_split(cfg.data_root, "dev")
    feats_by_id = dict(zip(dev_ids, bundle.dev_feats))
    # full dev metrics (macro)
    agg, per = evaluate(cfg, bundle.model, bundle.dev_feats, dev_ids)

    test_feats = featurize_split(cfg, bundle.fs, bundle.ontology, "test")
    test_ids = read_split(cfg.data_root, "test")
    feats_by_id.update(dict(zip(test_ids, test_feats)))

    diag = {}
    for sid in AMB_SCENARIOS:
        f = feats_by_id.get(sid)
        if f is None:
            continue
        diag[sid] = id0003_diagnosis(bundle.model, f, scenario_dir(cfg, sid))
    return {"stage": cfg.stage, "graph": cfg.graph.enabled,
            "dev_aggregate": agg, "amb_diagnosis": diag}


def main() -> None:
    on = run_one(str(REPO_ROOT / "configs" / "cita_g.yaml"))
    off = run_one(str(REPO_ROOT / "configs" / "cita_g_norel.yaml"))

    print("\n" + "=" * 70)
    print("M2 ABLATION SUMMARY -- id_0003 hard positive on ambiguity probes")
    print("(two T-72s exchangeable except for relation; only the relation")
    print(" encoder can separate them)")
    print("=" * 70)
    hdr = f"{'scenario':12} | {'relation ON':^26} | {'relation OFF':^26}"
    print(hdr)
    print("-" * len(hdr))

    def fmt(d):
        if not d:
            return "n/a"
        if d["id_0003_correct"]:
            return "OK (recovered)"
        if d["id_0003_wrong_merge"]:
            return f"WRONG-MERGE ->{d['a3_matched_to']}"
        if d["id_0003_fragmented"]:
            return "FRAGMENTED (abstained)"
        return "?"

    rec_on = rec_off = wm_off = fr_off = tot = 0
    for sid in AMB_SCENARIOS:
        d_on = on["amb_diagnosis"].get(sid, {})
        d_off = off["amb_diagnosis"].get(sid, {})
        if d_on and d_off:
            tot += 1
            rec_on += int(d_on["id_0003_correct"])
            rec_off += int(d_off["id_0003_correct"])
            wm_off += int(d_off["id_0003_wrong_merge"])
            fr_off += int(d_off["id_0003_fragmented"])
        print(f"{sid:12} | {fmt(d_on):^26} | {fmt(d_off):^26}")
    print("-" * len(hdr))
    print(f"id_0003 recovered : relation ON {rec_on}/{tot}   relation OFF {rec_off}/{tot}"
          f"  (OFF: {wm_off} wrong-merge, {fr_off} fragmented)")

    print("\nDev aggregate (macro avg):")
    print(f"  {'metric':22} {'relation ON':>12} {'relation OFF':>13}")
    for k in ["f1", "precision", "recall", "wrong_merge_rate",
              "fragmentation_rate", "dangling_precision", "dangling_recall"]:
        print(f"  {k:22} {on['dev_aggregate'][k]:>12.4f} {off['dev_aggregate'][k]:>13.4f}")

    out = {"relation_on": on, "relation_off": off}
    out_dir = REPO_ROOT / "runs" / "ablation_m2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\nsaved ablation to {out_dir / 'ablation.json'}")


if __name__ == "__main__":
    main()
