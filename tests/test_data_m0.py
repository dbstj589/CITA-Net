"""M0 acceptance tests: JSON-LD parsing, STKG<->cache consistency, generator
determinism + multi-scenario, ontology kinematics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from citanet.data.loader import load_scenario, validate_consistency

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "battlefield_stkg_dataset"
ONTOLOGY_DIR = DATA_ROOT / "ontology"


def test_scn_0001_spec_numbers(scn_0001):
    scn = scn_0001
    assert len(scn.observations) == 29
    assert scn.n_entities("A") == 6
    assert scn.n_entities("B") == 6
    gold = json.loads((DATA_ROOT / "scenarios" / "scn_0001" / "labels" /
                       "gold_identities.json").read_text(encoding="utf-8"))
    assert len(gold["identities"]) == 5
    assert len(gold["dangling_local_entities"]) == 2


def test_two_t72_are_distinct_identities(scn_0001):
    gold = json.loads((DATA_ROOT / "scenarios" / "scn_0001" / "labels" /
                       "gold_identities.json").read_text(encoding="utf-8"))
    t72 = [i for i in gold["identities"] if i["true_type"] == "T-72"]
    assert {i["gold_identity_id"] for i in t72} == {"id_0001", "id_0003"}


def test_id_0003_is_hard_positive_unknown_on_B(scn_0001):
    scn = scn_0001
    ent_b = scn.entities["B:ent_B_003"]
    assert ent_b.object_type == "UNKNOWN"      # KG_B classifier abstained
    ent_a = scn.entities["A:ent_A_003"]
    assert ent_a.object_type == "T-72"


def test_dangling_are_towed_weapons(scn_0001):
    scn = scn_0001
    assert scn.entities["A:ent_A_006"].object_type == "ZPU-4"
    assert scn.entities["B:ent_B_006"].object_type == "MO-120-RT"


def test_stkg_matches_observations_cache(scn_0001):
    # Re-run the assertion explicitly (load_scenario doesn't call it).
    validate_consistency(DATA_ROOT / "scenarios" / "scn_0001", scn_0001)


def test_relations_and_events_parsed(scn_0001):
    scn = scn_0001
    # id_0003 follows id_0001 in BOTH KGs -> relational evidence for the match.
    a3 = scn.entities["A:ent_A_003"]
    b3 = scn.entities["B:ent_B_003"]
    assert any(r.predicate == "follows" for r in a3.relations)
    assert any(r.predicate == "follows" for r in b3.relations)
    assert any(e.event_id == "evt_advance" for e in a3.events)
    assert any(e.event_id == "evt_advance" for e in b3.events)


def test_landmarks_and_events_from_commons(scn_0001):
    scn = scn_0001
    assert "lm_hill205" in scn.landmarks
    assert "evt_advance" in scn.events


@pytest.mark.parametrize("scenario_id", ["scn_t001", "scn_d002", "scn_e001"])
def test_multi_scenarios_load_and_validate(scenario_id):
    scn = load_scenario(DATA_ROOT / "scenarios" / scenario_id, ONTOLOGY_DIR)
    validate_consistency(DATA_ROOT / "scenarios" / scenario_id, scn)
    assert len(scn.observations) > 0
    assert scn.n_entities("A") >= 1 and scn.n_entities("B") >= 1


def test_splits_exist_and_disjoint():
    splits = {}
    for name in ("train", "dev", "test"):
        ids = (DATA_ROOT / "splits" / f"{name}.txt").read_text(encoding="utf-8").split()
        splits[name] = set(ids)
    # train must not leak into test (§A.9.3)
    assert splits["train"].isdisjoint(splits["test"])
    assert "scn_0001" in splits["dev"]


def test_generator_is_deterministic(tmp_path):
    """Two regenerations with the same seed produce byte-identical observations."""
    import importlib.util

    gen_path = REPO_ROOT / "scripts" / "gen_dataset.py"
    out = subprocess.run(
        [sys.executable, str(gen_path), "--scenario-id", "scn_0001"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert out.returncode == 0, out.stderr
    obs1 = (DATA_ROOT / "scenarios" / "scn_0001" / "observations.jsonl").read_text(encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(gen_path), "--scenario-id", "scn_0001"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert out.returncode == 0, out.stderr
    obs2 = (DATA_ROOT / "scenarios" / "scn_0001" / "observations.jsonl").read_text(encoding="utf-8")
    assert obs1 == obs2


def test_ontology_kinematics(ontology):
    # towed weapons cannot self-propel
    assert ontology.v_max("ZPU-4", "Emplaced") == 0.0
    assert ontology.v_max("MO-120-RT", "Firing") == 0.0
    # tanks move
    assert ontology.v_max("T-72", "Moving") > 10.0
    # UNKNOWN is type-compatible with anything
    assert ontology.types_compatible("UNKNOWN", "T-72")
    assert ontology.types_compatible("T-72", "T-72")
    assert not ontology.types_compatible("T-72", "BMP-2")
