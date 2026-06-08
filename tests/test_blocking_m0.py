"""Blocking + kinematic-feasibility unit tests."""
from __future__ import annotations

import json
from pathlib import Path

from citanet.data.blocking import generate_candidates, identity_coverage
from citanet.data.kinematics import feasibility

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "battlefield_stkg_dataset"


def _identities(scenario_id: str) -> list[dict]:
    gold = json.loads((DATA_ROOT / "scenarios" / scenario_id / "labels" /
                       "gold_identities.json").read_text(encoding="utf-8"))
    return gold["identities"]


def test_blocking_keeps_every_identity_connectable(scn_0001, ontology):
    pairs = generate_candidates(scn_0001.observations, ontology,
                                dt_max_s=180.0, theta_text=0.3)
    cov = identity_coverage(pairs, scn_0001.observations, _identities("scn_0001"))
    assert cov == 1.0, f"blocking left some identity unrecoverable (coverage={cov})"


def test_blocking_reduces_pair_count(scn_0001, ontology):
    n = len(scn_0001.observations)
    full = n * (n - 1) // 2
    pairs = generate_candidates(scn_0001.observations, ontology)
    assert len(pairs) < full     # genuine pruning vs O(M^2)


def test_hard_positive_id_0003_survives_blocking(scn_0001, ontology):
    """The UNKNOWN-typed KG_B side of id_0003 must still produce a cross-KG
    candidate against its KG_A T-72 observations (text gate bypassed)."""
    pairs = generate_candidates(scn_0001.observations, ontology, cross_kg_only=True)
    obs = sorted(scn_0001.observations, key=lambda o: (o.t, o.obs_id))
    a3 = {o.obs_id for o in obs if o.local_entity_id == "ent_A_003"}
    b3 = {o.obs_id for o in obs if o.local_entity_id == "ent_B_003"}
    found = False
    for p in pairs:
        ids = {obs[p.i].obs_id, obs[p.j].obs_id}
        if ids & a3 and ids & b3:
            found = True
            break
    assert found, "hard positive id_0003 cross-KG pair was filtered by blocking"


def test_feasibility_towed_emplaced_cannot_move(scn_0001, ontology):
    scn = scn_0001
    obs = scn.obs_index
    zpu = [o for o in scn.observations if o.local_entity_id == "ent_A_006"]
    zpu.sort(key=lambda o: o.t)
    # ZPU-4 stays put -> moving it any meaningful distance is infeasible.
    far = zpu[0]
    fake = scn.observations[0]
    # construct: same towed obs vs a displaced copy
    import copy
    moved = copy.deepcopy(far)
    moved.easting += 300.0
    moved.t = far.t + 30.0
    f = feasibility(far, moved, ontology)
    assert not f.feasible
    assert f.violation_mps > 0


def test_feasibility_tank_normal_move_is_ok(scn_0001, ontology):
    scn = scn_0001
    t72 = [o for o in scn.observations if o.local_entity_id == "ent_A_001"]
    t72.sort(key=lambda o: o.t)
    f = feasibility(t72[0], t72[1], ontology)
    assert f.feasible
