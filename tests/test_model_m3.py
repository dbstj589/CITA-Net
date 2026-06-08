"""M3 tests: Sinkhorn row/column convergence, row-stochastic assignment, full
decode two-level invariants, Part-B schema validation, and an end-to-end
acceptance check on scn_0001 (5 identities recovered, two T-72s not merged,
danglings abstained)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from citanet.config import load_config
from citanet.decode import decode_full
from citanet.engine import (
    build_feature_space,
    build_model,
    featurize_split,
    read_split,
    scenario_dir,
    train,
)
from citanet.model.decoder import sinkhorn_assign, sinkhorn_log
from citanet.output_schema import SchemaError, validate_output
from citanet.serialize import build_part_b

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sinkhorn_log_converges_doubly_stochastic():
    """On a square matrix, sinkhorn_log must converge so BOTH row and column
    sums approach 1 (doubly stochastic)."""
    torch.manual_seed(0)
    Z = torch.randn(6, 6) * 2.0
    A = sinkhorn_log(Z, n_iters=200, tau=1.0)
    row = A.sum(dim=1)
    col = A.sum(dim=0)
    assert torch.allclose(row, torch.ones(6), atol=1e-4), f"row sums {row}"
    assert torch.allclose(col, torch.ones(6), atol=1e-2), f"col sums {col}"


def test_sinkhorn_log_row_sums_one_rectangular():
    torch.manual_seed(1)
    Z = torch.randn(10, 4)
    A = sinkhorn_log(Z, n_iters=50, tau=1.0)
    assert torch.allclose(A.sum(dim=1), torch.ones(10), atol=1e-5)


def test_sinkhorn_assign_rows_sum_one_and_nonneg():
    """The decoder's row-stochastic assignment: each observation's mass over
    (K+1) slots sums to 1, all entries in [0,1]."""
    torch.manual_seed(2)
    Z = torch.randn(15, 9)          # 8 slots + null
    A = sinkhorn_assign(Z, n_iters=20, tau=0.5, col_strength=0.1)
    assert torch.allclose(A.sum(dim=1), torch.ones(15), atol=1e-5)
    assert torch.all(A >= 0) and torch.all(A <= 1 + 1e-6)


def test_decoder_assignment_rows_sum_one_in_model():
    cfg = load_config(REPO_ROOT / "configs" / "cita_full.yaml")
    fs, ont = build_feature_space(cfg)
    feats = featurize_split(cfg, fs, ont, "dev")
    ids = read_split(cfg.data_root, "dev")
    f = dict(zip(ids, feats))["scn_0001"]
    model = build_model(cfg, fs, ont)
    with torch.no_grad():
        out = model(f)
    assert out.assign is not None
    assert torch.allclose(out.assign.sum(dim=1), torch.ones(out.assign.shape[0]), atol=1e-4)


def test_part_b_schema_validation_roundtrip():
    cfg = load_config(REPO_ROOT / "configs" / "cita_full.yaml")
    fs, ont = build_feature_space(cfg)
    feats = featurize_split(cfg, fs, ont, "dev")
    ids = read_split(cfg.data_root, "dev")
    f = dict(zip(ids, feats))["scn_0001"]
    model = build_model(cfg, fs, ont)
    with torch.no_grad():
        out = model(f)
    res = decode_full(out, f, ont)
    doc = build_part_b("scn_0001", cfg, res, f)
    validate_output(doc)            # must not raise

    # a malformed document must be rejected
    bad = dict(doc)
    del bad["stats"]
    with pytest.raises(SchemaError):
        validate_output(bad)


def test_dangling_decisions_are_abstain():
    cfg = load_config(REPO_ROOT / "configs" / "cita_full.yaml")
    fs, ont = build_feature_space(cfg)
    feats = featurize_split(cfg, fs, ont, "dev")
    ids = read_split(cfg.data_root, "dev")
    f = dict(zip(ids, feats))["scn_0001"]
    model = build_model(cfg, fs, ont)
    with torch.no_grad():
        out = model(f)
    res = decode_full(out, f, ont)
    for d in res.dangling:
        assert d["decision"] == "abstain"


def test_bmotion_suppresses_impossible_transitions():
    """With b_motion enabled, kinematically-infeasible candidate pairs (which
    relaxed blocking admits) must receive a much lower transition probability
    than with b_motion disabled -- the core ablation behaviour."""
    cfg = load_config(REPO_ROOT / "configs" / "cita_full.yaml")
    cfg.blocking.reach_extra_m = 6000.0           # relax: admit impossible pairs
    fs, ont = build_feature_space(cfg)
    feats = featurize_split(cfg, fs, ont, "dev")
    ids = read_split(cfg.data_root, "dev")
    f = dict(zip(ids, feats))["scn_amb01"]        # has towed danglings (v_max 0)

    model_on = build_model(cfg, fs, ont)
    cfg_off = load_config(REPO_ROOT / "configs" / "cita_full.yaml")
    cfg_off.cta.enabled_terms = ["sem", "time", "state", "rel", "src"]  # b_motion off
    model_off = build_model(cfg_off, fs, ont)

    with torch.no_grad():
        out_on = model_on(f)
        out_off = model_off(f)
    infeasible = f.p_req_speed > f.p_feas_speed
    assert bool(infeasible.any()), "relaxed blocking admitted no infeasible pairs"
    p_on = float(out_on.cta.p_transition[infeasible].mean())
    p_off = float(out_off.cta.p_transition[infeasible].mean())
    assert p_on < p_off, f"b_motion did not suppress impossible transitions: {p_on} !< {p_off}"


@pytest.mark.slow
def test_scn_0001_full_acceptance_after_training():
    """End-to-end: train the full model and verify scn_0001 recovers all 5 gold
    identities, does NOT merge the two T-72s (wrong-merge 0), and abstains on
    both danglings."""
    import json

    cfg = load_config(REPO_ROOT / "configs" / "cita_full.yaml")
    cfg.train.epochs = 150
    bundle = train(cfg, verbose=False)
    f = dict(zip(read_split(cfg.data_root, "dev"), bundle.dev_feats))["scn_0001"]
    with torch.no_grad():
        out = bundle.model(f)
    res = decode_full(out, f, bundle.ontology)

    # 5 identities, each a clean one-to-one A<->B
    paired = [idn for idn in res.identities
              if "A" in idn.local_entity_ids and "B" in idn.local_entity_ids]
    assert len(paired) == 5, f"expected 5 identities, got {len(paired)}"

    gold = json.loads((scenario_dir(cfg, "scn_0001") / "labels" /
                       "gold_identities.json").read_text(encoding="utf-8"))
    gold_pairs = {(i["member_local_entities"]["A"], i["member_local_entities"]["B"])
                  for i in gold["identities"]}
    pred_pairs = {(idn.local_entity_ids["A"], idn.local_entity_ids["B"]) for idn in paired}
    assert pred_pairs == gold_pairs, f"matches differ: {pred_pairs ^ gold_pairs}"

    # two T-72s must be in different identities (no wrong-merge)
    t72 = [idn for idn in paired if idn.type == "T-72"]
    assert len(t72) == 2
    assert t72[0].local_entity_ids["A"] != t72[1].local_entity_ids["A"]

    # both danglings abstained
    dang = {(d["kg_id"], d["local_entity_id"]) for d in res.dangling}
    assert ("A", "ent_A_006") in dang and ("B", "ent_B_006") in dang
