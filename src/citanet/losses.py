"""Loss components (§4.6).

L = lambda_pair * L_pair + lambda_transition * L_transition
  + lambda_trajectory * L_trajectory + lambda_dangling * L_dangling
  + lambda_assign * L_assign

M1 uses L_pair + L_dangling. M2 adds L_transition. M3 adds L_trajectory +
L_assign (assignment loss is computed inside the decoder wrapper). Each term is
gated by its lambda so a stage simply zeroes the weights it does not use.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .model.citanet import ModelOutput
from .model.featurize import ScenarioFeatures


def pair_loss(out: ModelOutput, feats: ScenarioFeatures, pos_weight: float) -> torch.Tensor:
    """BCE over cross-KG candidate pairs (the only pairs L_pair supervises)."""
    cross = feats.pair_cross
    if not cross.any():
        return torch.zeros((), requires_grad=True)
    logits = out.pair_logits[cross]
    labels = feats.pair_label[cross]
    pw = torch.tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)


def dangling_loss(out: ModelOutput, feats: ScenarioFeatures) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(out.dangling_logits, feats.dangling_label)


def _pair_index(feats: ScenarioFeatures) -> dict[frozenset, int]:
    return {frozenset((feats.obs_ids[p.i], feats.obs_ids[p.j])): q
            for q, p in enumerate(feats.pairs)}


def transition_loss(out: ModelOutput, feats: ScenarioFeatures,
                    scenario_dir: str | Path) -> torch.Tensor:
    """BCE on CTA transition scores. Uses ``transition_level.json`` when present
    (small suite); otherwise (large suite, no exhaustive labels) derives the
    supervision from the gold assignment: a candidate pair is a real transition
    iff its two observations belong to the same gold identity."""
    sdir = Path(scenario_dir)
    tl_path = sdir / "labels" / "transition_level.json"
    if tl_path.exists():
        tl = json.loads(tl_path.read_text(encoding="utf-8"))
        pidx = _pair_index(feats)
        logits, labels = [], []
        for rec, lab in [(tl["positives"], 1.0), (tl["hard_negatives"], 0.0)]:
            for e in rec:
                q = pidx.get(frozenset((e["src"], e["dst"])))
                if q is not None:
                    logits.append(out.cta.score[q])
                    labels.append(lab)
        if not logits:
            return torch.zeros((), requires_grad=True)
        return F.binary_cross_entropy_with_logits(torch.stack(logits), torch.tensor(labels))

    # derive from gold assignment over candidate pairs
    gold = json.loads((sdir / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
    obs_to_gold = {a["obs_id"]: a["gold_identity_id"] for a in gold.get("assignment", [])}
    if not feats.pairs:
        return torch.zeros((), requires_grad=True)
    labels = torch.tensor(
        [1.0 if (obs_to_gold.get(feats.obs_ids[p.i]) is not None
                 and obs_to_gold.get(feats.obs_ids[p.i]) == obs_to_gold.get(feats.obs_ids[p.j]))
         else 0.0 for p in feats.pairs])
    pos = labels.sum().clamp(min=1.0)
    pos_weight = ((labels.numel() - pos) / pos).clamp(max=20.0)
    return F.binary_cross_entropy_with_logits(out.cta.score, labels, pos_weight=pos_weight)


def assign_loss(out: ModelOutput, feats: ScenarioFeatures,
                scenario_dir: str | Path) -> torch.Tensor:
    """Slot assignment loss (§4.6 L_assign).

    Slots are permutation-free, so we first align predicted slots to gold
    identities by greedy maximum overlap (on the detached Sinkhorn assignment),
    then cross-entropy each observation against its gold identity's matched slot
    (dangling observations target the ∅ slot)."""
    if out.assign_logits is None:
        return torch.zeros((), requires_grad=True)
    Z = out.assign_logits                       # (M, K+1)
    A = out.assign.detach()                     # (M, K+1)
    M, S = Z.shape
    K = S - 1                                   # ∅ slot index = K
    sdir = Path(scenario_dir)
    gold = json.loads((sdir / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
    # obs -> gold id, from the canonical assignment (large) or member lists (small)
    if "assignment" in gold:
        obs_to_gold = {a["obs_id"]: a["gold_identity_id"] for a in gold["assignment"]}
    else:
        obs_to_gold = {oid: i["gold_identity_id"]
                       for i in gold["identities"] for oid in i["member_observations"]}
    dangling = json.loads((sdir / "labels" / "dangling.json").read_text(encoding="utf-8"))
    dang_obs = set(dangling["dangling_observations"])
    idx = {oid: k for k, oid in enumerate(feats.obs_ids)}

    # gold identity -> member observation indices (non-dangling)
    gold_groups: dict[str, list[int]] = {}
    for oid, g in obs_to_gold.items():
        if oid in dang_obs or oid not in idx:
            continue
        gold_groups.setdefault(g, []).append(idx[oid])

    golds = list(gold_groups.keys())
    # overlap matrix O[g, k] over real slots
    overlap = torch.zeros(len(golds), K)
    for gi, g in enumerate(golds):
        overlap[gi] = A[gold_groups[g], :K].sum(dim=0)

    # greedy max-overlap slot assignment (one slot per gold)
    gold_slot: dict[str, int] = {}
    used_slots: set[int] = set()
    flat = [(float(overlap[gi, k]), gi, k) for gi in range(len(golds)) for k in range(K)]
    flat.sort(reverse=True)
    assigned_golds: set[int] = set()
    for _, gi, k in flat:
        if gi in assigned_golds or k in used_slots:
            continue
        gold_slot[golds[gi]] = k
        assigned_golds.add(gi)
        used_slots.add(k)
        if len(assigned_golds) == len(golds):
            break

    # build per-observation target slot
    targets = torch.full((M,), -1, dtype=torch.long)
    for oid in feats.obs_ids:
        m = idx[oid]
        if oid in dang_obs:
            targets[m] = K                       # ∅ slot
        else:
            g = obs_to_gold.get(oid)
            if g in gold_slot:
                targets[m] = gold_slot[g]
    valid = targets >= 0
    if valid.sum() == 0:
        return torch.zeros((), requires_grad=True)
    return F.cross_entropy(Z[valid], targets[valid])


def trajectory_loss(out: ModelOutput, feats: ScenarioFeatures) -> torch.Tensor:
    """Soft kinematic penalty: candidate pairs the model believes are the same
    object (high p) should not require an impossible speed. Weights the motion
    violation by the predicted same-probability so confident-but-impossible
    links are penalised (§4.6 L_trajectory)."""
    viol = torch.clamp(feats.p_req_speed - feats.p_feas_speed, min=0.0)
    p_same = torch.sigmoid(out.pair_logits)
    if viol.numel() == 0:
        return torch.zeros((), requires_grad=True)
    return (p_same * viol).mean() / 10.0   # /10 keeps it on a comparable scale


def total_loss(out: ModelOutput, feats: ScenarioFeatures, cfg,
               scenario_dir: str | Path) -> tuple[torch.Tensor, dict[str, float]]:
    lc = cfg.loss
    parts: dict[str, torch.Tensor] = {}
    if lc.lambda_pair > 0:
        parts["pair"] = lc.lambda_pair * pair_loss(out, feats, lc.pair_pos_weight)
    if lc.lambda_dangling > 0:
        parts["dangling"] = lc.lambda_dangling * dangling_loss(out, feats)
    if lc.lambda_transition > 0:
        parts["transition"] = lc.lambda_transition * transition_loss(out, feats, scenario_dir)
    if lc.lambda_trajectory > 0:
        parts["trajectory"] = lc.lambda_trajectory * trajectory_loss(out, feats)
    if lc.lambda_assign > 0 and out.assign_logits is not None:
        parts["assign"] = lc.lambda_assign * assign_loss(out, feats, scenario_dir)

    total = sum(parts.values()) if parts else torch.zeros((), requires_grad=True)
    return total, {k: float(v.detach()) for k, v in parts.items()}
