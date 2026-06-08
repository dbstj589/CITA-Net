"""M2 tests: relation-context GNN behaviour, clean b_rel ablation gating, the
relational signal that separates the exchangeable T-72s, and the b_state
penalty on an impossible Destroyed->Moving transition."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.engine import build_feature_space, build_model, featurize_split, read_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(stage_config, sid="scn_amb01", split="dev"):
    cfg = load_config(REPO_ROOT / "configs" / stage_config)
    fs, ont = build_feature_space(cfg)
    feats = featurize_split(cfg, fs, ont, split)
    ids = read_split(cfg.data_root, split)
    f = dict(zip(ids, feats))[sid]
    model = build_model(cfg, fs, ont)
    return cfg, model, f, fs, ont


def test_gnn_changes_embeddings_via_relations():
    """The graph encoder must actually move embeddings, and removing the edges
    must change its output (so the effect is relational, not a no-op)."""
    cfg, model, f, fs, ont = _load("cita_g.yaml")
    assert model.graph is not None
    h0 = model.encoder(f)
    h1 = model.graph(h0, f)
    assert not torch.allclose(h0, h1), "GNN did not change embeddings"

    import copy
    f_noedge = copy.copy(f)
    f_noedge.edges = []
    h_noedge = model.graph(h0, f_noedge)
    assert not torch.allclose(h1, h_noedge), "edges had no effect on GNN output"


def test_brel_gated_by_enabled_terms():
    """b_rel must be active under cita_g and identically zero under the ablation
    (cita_g_norel) -- no relational leak into the pair head."""
    with torch.no_grad():
        _, model_g, f_g, _, _ = _load("cita_g.yaml")
        out_g = model_g(f_g)
        assert float(out_g.cta.b_rel.abs().max()) > 0

        _, model_n, f_n, _, _ = _load("cita_g_norel.yaml")
        out_n = model_n(f_n)
        assert torch.all(out_n.cta.b_rel == 0)


def _pair_between(f, ent_a, ent_b):
    """Indices of candidate pairs whose endpoints are the given two entities."""
    out = []
    for q, p in enumerate(f.pairs):
        ka = f"{f.kg_ids[p.i]}:{f.local_entity_ids[p.i]}"
        kb = f"{f.kg_ids[p.j]}:{f.local_entity_ids[p.j]}"
        if {ka, kb} == {ent_a, ent_b}:
            out.append(q)
    return out


def test_relational_signal_favours_correct_match():
    """On scn_amb01 the two T-72s are identical except for their relation. The
    relational Jaccard feature must score the CORRECT cross pair
    (A_003 'follows' <-> B_003 'follows') above the WRONG one
    (A_003 'follows' <-> B_001 'near')."""
    _, _, f, _, _ = _load("cita_g.yaml")
    correct = _pair_between(f, "A:ent_A_003", "B:ent_B_003")
    wrong = _pair_between(f, "A:ent_A_003", "B:ent_B_001")
    assert correct and wrong
    j_correct = max(float(f.p_rel_jaccard[q]) for q in correct)
    j_wrong = max(float(f.p_rel_jaccard[q]) for q in wrong)
    assert j_correct > j_wrong, f"relational signal not discriminative: {j_correct} !> {j_wrong}"


def test_destroyed_to_moving_is_penalised_by_bstate():
    """A candidate pair linking the Destroyed T-72 wreck (dangling A) to a live
    Moving T-72 must receive a strongly negative b_state; with the state term
    disabled the penalty vanishes (showing it IS b_state doing the work)."""
    cfg, model, f, fs, ont = _load("cita_g.yaml")
    out = model(f)
    # find pairs touching the wreck entity (ent_A_006, Destroyed)
    wreck_pairs = [q for q, p in enumerate(f.pairs)
                   if "ent_A_006" in (f.local_entity_ids[p.i], f.local_entity_ids[p.j])]
    assert wreck_pairs, "no candidate pair touches the wreck (blocking too strict?)"
    # at least one wreck pair must be a Destroyed<->Moving link, strongly penalised
    bstate = out.cta.b_state[wreck_pairs]
    assert float(bstate.min()) < -2.0, f"b_state did not penalise Destroyed link: {float(bstate.min())}"

    # ablation: state term off -> b_state contribution is zero
    cfg2 = load_config(REPO_ROOT / "configs" / "cita_g_norel.yaml")
    cfg2.cta.enabled_terms = ["sem", "time", "motion", "src"]   # drop 'state'
    model2 = build_model(cfg2, fs, ont)
    out2 = model2(f)
    assert torch.all(out2.cta.b_state[wreck_pairs] == 0)


def test_state_compat_matrix_has_destroyed():
    cfg, model, f, fs, ont = _load("cita_g.yaml")
    # Destroyed -> Moving must be near-impossible
    assert ont.state_compat("Destroyed", "Moving") < 0.05
    assert ont.state_compat("Moving", "Moving") == 1.0
    assert "Destroyed" in ont.state_names
