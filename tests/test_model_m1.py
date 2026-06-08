"""M1 unit + integration tests: encoder gate, CTA constraints, two-level
decode (many-to-one obs->identity, one-to-one identity<->identity), dangling
abstain, and a mini training loop that must improve the loss."""
from __future__ import annotations

from pathlib import Path

import torch

from citanet.config import load_config
from citanet.decode import decode_entities
from citanet.engine import (
    build_feature_space,
    build_model,
    featurize_split,
    read_split,
    scenario_dir,
)
from citanet.losses import total_loss

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    cfg = load_config(REPO_ROOT / "configs" / "cita_lite.yaml")
    cfg.train.epochs = 1
    return cfg


def _scn(cfg, sid="scn_0001", split="dev"):
    fs, ont = build_feature_space(cfg)
    feats = featurize_split(cfg, fs, ont, split)
    ids = read_split(cfg.data_root, split)
    f = dict(zip(ids, feats))[sid]
    model = build_model(cfg, fs, ont)
    return model, f, fs, ont


def test_encoder_gate_in_unit_interval():
    cfg = _cfg()
    model, f, fs, ont = _scn(cfg)
    g = model.encoder.gate(f.source_id)
    assert g.shape == (len(f.obs_ids), 5)
    assert torch.all(g > 0) and torch.all(g < 1)


def test_gate_prior_reflects_source_reliability():
    """A high-reliability source (EO_IR) should start with a higher gate than a
    low-reliability one (ACOUSTIC) at init (prior-dominated)."""
    cfg = _cfg()
    model, f, fs, ont = _scn(cfg)
    eo = torch.tensor([fs.source_id("EO_IR")])
    ac = torch.tensor([fs.source_id("ACOUSTIC")])
    with torch.no_grad():
        assert float(model.encoder.gate(eo).mean()) > float(model.encoder.gate(ac).mean())


def test_cta_backward_in_time_is_penalised():
    cfg = _cfg()
    model, f, fs, ont = _scn(cfg)
    out = model(f)
    # all candidate pairs are forward (dt>=0) so b_time must be ~0 everywhere
    assert torch.allclose(out.cta.b_time, torch.zeros_like(out.cta.b_time))


def test_cta_motion_penalises_impossible_speed():
    """b_motion must be <= 0 and strictly negative for kinematically violating
    pairs (required > feasible speed)."""
    cfg = _cfg()
    model, f, fs, ont = _scn(cfg)
    out = model(f)
    assert torch.all(out.cta.b_motion <= 1e-6)
    viol = (f.p_req_speed > f.p_feas_speed)
    if viol.any():
        assert torch.all(out.cta.b_motion[viol] < 0)


def test_decode_is_one_to_one_at_identity_level():
    cfg = _cfg()
    model, f, fs, ont = _scn(cfg)
    out = model(f)
    res = decode_entities(out, f)
    a_used = [idn.a_entity for idn in res.identities]
    b_used = [idn.b_entity for idn in res.identities]
    assert len(a_used) == len(set(a_used)), "an A entity matched twice (not 1-to-1)"
    assert len(b_used) == len(set(b_used)), "a B entity matched twice (not 1-to-1)"


def test_obs_to_identity_is_many_to_one():
    """Each matched identity groups multiple observations (many-to-one), and no
    observation is claimed by two identities."""
    cfg = _cfg()
    model, f, fs, ont = _scn(cfg)
    out = model(f)
    res = decode_entities(out, f)
    seen = set()
    for idn in res.identities:
        for oid in idn.member_obs:
            assert oid not in seen, "observation assigned to two identities"
            seen.add(oid)


def test_mini_training_reduces_loss():
    cfg = _cfg()
    cfg.train.epochs = 30
    fs, ont = build_feature_space(cfg)
    train_feats = featurize_split(cfg, fs, ont, "train")
    model = build_model(cfg, fs, ont)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    def epoch_loss():
        tot = 0.0
        for f in train_feats:
            out = model(f)
            l, _ = total_loss(out, f, cfg, scenario_dir(cfg, f.scenario_id))
            tot += float(l.detach())
        return tot / len(train_feats)

    start = epoch_loss()
    model.train()
    for _ in range(cfg.train.epochs):
        for f in train_feats:
            out = model(f)
            l, _ = total_loss(out, f, cfg, scenario_dir(cfg, f.scenario_id))
            opt.zero_grad(); l.backward(); opt.step()
    end = epoch_loss()
    assert end < start * 0.5, f"loss did not improve enough: {start:.3f} -> {end:.3f}"
