"""Large-suite tests (isolated from the frozen small suite).

Generates a tiny large suite into a temp dir and checks: exact triple counting
+ target growth, grid blocking recall >= 0.98 with candidates << all-pairs and
grid ⊆ brute-force, streaming loader peak memory < whole-suite, label derivation
from gold (no O(M^2) pair lists), and an end-to-end featurize→model→decode→Part-B
smoke.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN = REPO_ROOT / "scripts" / "gen_dataset_large.py"
LARGE_ROOT = REPO_ROOT / "data" / "battlefield_stkg_large"
ONT = LARGE_ROOT / "ontology"


def _gen_module():
    spec = importlib.util.spec_from_file_location("gen_large", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def small_large_suite(tmp_path_factory):
    """Generate a few sectors per split into a temp data root (fast)."""
    mod = _gen_module()
    root = tmp_path_factory.mktemp("large")
    mod.DATA_ROOT = root
    mod.ONTOLOGY_DIR = root / "ontology"
    # copy ontology
    (root / "ontology").mkdir(parents=True)
    for f in ONT.iterdir():
        (root / "ontology" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    knobs = mod.Knobs(identities_per_sector=24, obs_per_track=8)
    # small per-split sector counts via single-split builds
    for split, n in (("train", 4), ("dev", 2), ("test", 2)):
        ids, total = mod.build_split(split, knobs, n_sectors=n, target_triples=None)
        (root / "splits").mkdir(exist_ok=True)
        (root / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return root


def _ont(root):
    from citanet.data.ontology import load_ontology
    return load_ontology(root / "ontology")


def test_triple_count_exact_and_recorded(small_large_suite):
    root = small_large_suite
    sd = root / "sectors" / "sec_dev_0000"
    n_lines = sum(1 for _ in open(sd / "stkg.nt", encoding="utf-8"))
    m = json.loads((sd / "manifest.json").read_text(encoding="utf-8"))
    assert n_lines == m["n_triples"] > 0


def test_target_triples_growth(tmp_path):
    """--target-triples grows sectors until the target is met; manifest records
    the actual total."""
    mod = _gen_module()
    mod.DATA_ROOT = tmp_path
    mod.ONTOLOGY_DIR = tmp_path / "ontology"
    (tmp_path / "ontology").mkdir(parents=True)
    for f in ONT.iterdir():
        (tmp_path / "ontology" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    knobs = mod.Knobs(identities_per_sector=20, obs_per_track=6)
    ids, total = mod.build_split("dev", knobs, n_sectors=None, target_triples=15000)
    assert total >= 15000
    assert len(ids) >= 2                      # needed more than one sector


def test_grid_blocking_recall_and_reduction(small_large_suite):
    from citanet.data.blocking import generate_candidates
    from citanet.data.blocking_grid import (
        blocking_stats,
        generate_candidates_grid,
        gold_cross_kg_same_pairs,
    )
    from citanet.data.stream import load_sector_observations
    root = small_large_suite
    ont = _ont(root)
    recalls, reductions = [], []
    for sid in ("sec_train_0000", "sec_dev_0000", "sec_test_0001"):
        sd = root / "sectors" / sid
        obs = load_sector_observations(sd)
        gold = json.loads((sd / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
        gp = gold_cross_kg_same_pairs(obs, gold, 180.0)
        grid = generate_candidates_grid(obs, ont, dt_max_s=180, r_err_floor_m=80,
                                        cell_size_m=400, use_text_gate=False,
                                        type_by_category=True)
        bf = generate_candidates(obs, ont, dt_max_s=180, r_err_floor_m=80,
                                 use_text_gate=False, type_by_category=True)
        so = sorted(obs, key=lambda o: (o.t, o.obs_id))
        gk = {frozenset((so[p.i].obs_id, so[p.j].obs_id)) for p in grid}
        bk = {frozenset((so[p.i].obs_id, so[p.j].obs_id)) for p in bf}
        assert gk <= bk, "grid produced a pair brute force did not (criteria bug)"
        st = blocking_stats(grid, obs, gp)
        recalls.append(st.recall); reductions.append(st.reduction)
        assert st.n_candidates < st.n_all_pairs    # genuine pruning
    assert min(recalls) >= 0.98, f"blocking recall below 0.98: {recalls}"
    assert min(reductions) > 2.0, f"insufficient candidate reduction: {reductions}"


def test_no_exhaustive_pair_labels_stored(small_large_suite):
    """Gold labels must NOT contain an O(M^2) cross-KG pair list -- only the
    assignment + dangling."""
    root = small_large_suite
    gold = json.loads((root / "sectors" / "sec_dev_0000" / "labels" /
                       "gold_identities.json").read_text(encoding="utf-8"))
    assert "assignment" in gold and "identities" in gold
    assert "cross_kg_pairs" not in json.dumps(gold)


def test_streaming_loader_peak_memory(small_large_suite):
    """Iterating sectors one at a time keeps peak memory near a single sector,
    well below loading all sectors at once."""
    from citanet.data.stream import load_sector_observations, read_split
    root = small_large_suite
    ids = read_split(root, "train")

    tracemalloc.start()
    for sid in ids:                            # streaming
        obs = load_sector_observations(root / "sectors" / sid)
        _ = len(obs)
        del obs
    _, peak_stream = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    allobs = [load_sector_observations(root / "sectors" / sid) for sid in ids]  # all at once
    _, peak_all = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del allobs
    assert peak_stream < peak_all, f"streaming peak {peak_stream} !< all {peak_all}"


def test_end_to_end_large_smoke(small_large_suite):
    from citanet.config import load_config
    from citanet.decode import decode_full
    from citanet.engine import build_model
    from citanet.data.stream import featurize_sector, read_split
    from citanet.model.featurize import FeatureSpace
    from citanet.model.text import Vocab
    from citanet.output_schema import validate_output
    from citanet.serialize import build_part_b
    root = small_large_suite
    ont = _ont(root)
    cfg = load_config(REPO_ROOT / "configs" / "cita_full_large.yaml")
    cfg.data_root = str(root)
    cfg.ontology_dir = str(root / "ontology")
    cfg.decoder.num_slots = 32

    sid = read_split(root, "dev")[0]
    sd = root / "sectors" / sid
    labs = [json.loads(l)["label_text"] for l in open(sd / "observations.jsonl", encoding="utf-8")]
    fs = FeatureSpace(vocab=Vocab.build(labs), types=ont.type_names,
                      states=ont.state_names, sources=ont.source_names, max_tokens=16)
    feats = featurize_sector(sd, fs, ont, cfg)
    model = build_model(cfg, fs, ont)
    with torch.no_grad():
        out = model(feats)
    assert out.assign is not None
    assert torch.allclose(out.assign.sum(dim=1), torch.ones(out.assign.shape[0]), atol=1e-4)
    doc = build_part_b(sid, cfg, decode_full(out, feats, ont), feats)
    validate_output(doc)            # Part-B schema OK on a large sector
