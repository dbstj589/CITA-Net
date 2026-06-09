"""Streaming sector loader for the large suite.

Sectors are independent shards. The loader yields ONE sector's features at a
time so peak memory stays at a single sector (~hundreds of observations), never
the whole suite. Observations are read from the flattened ``observations.jsonl``
(fast path); the canonical ``stkg.nt`` is the provenance/counting artifact and
is consistency-checked separately. Pair labels are derived on the fly from the
gold assignment + blocking (no stored O(M^2) pair lists), with the candidate
negatives serving as hard negatives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch

from ..model.featurize import FeatureSpace, ScenarioFeatures, assemble_features
from .blocking_grid import generate_candidates_grid
from .ontology import Ontology, load_ontology
from .schema import EventRef, Observation, Relation, parse_iso


def read_split(data_root: str | Path, split: str) -> list[str]:
    p = Path(data_root) / "splits" / f"{split}.txt"
    return [s for s in p.read_text(encoding="utf-8").split() if s]


def load_sector_observations(sector_dir: str | Path) -> list[Observation]:
    sector_dir = Path(sector_dir)
    manifest = json.loads((sector_dir / "manifest.json").read_text(encoding="utf-8"))
    epoch = parse_iso(manifest["epoch"])
    obs: list[Observation] = []
    with open(sector_dir / "observations.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            o = json.loads(line)
            t = (parse_iso(o["time"]) - epoch).total_seconds()
            loc = o["location"]; vel = o.get("velocity", {})
            obs.append(Observation(
                obs_id=o["obs_id"], kg_id=o["kg_id"], source=o["source"],
                local_entity_id=o["local_entity_id"], label_text=o["label_text"],
                type=o["type"], type_confidence=float(o["type_confidence"]),
                state=o["state"], state_confidence=float(o["state_confidence"]),
                time_iso=o["time"], t=t,
                easting=float(loc["easting"]), northing=float(loc["northing"]),
                elevation=float(loc.get("elevation", 0.0)), crs=loc.get("crs", "UTM52N"),
                mgrs=loc.get("mgrs", ""), cep_m=float(o["cep_m"]),
                speed_mps=vel.get("speed_mps"), heading_deg=vel.get("heading_deg"),
                relations=[Relation(r["predicate"], r["target_ref"], float(r.get("confidence", 1.0)))
                           for r in o.get("relations", [])],
                events=[EventRef(e["event_id"], e.get("role", "participant"))
                        for e in o.get("events", [])],
                provenance=o.get("provenance", {}),
            ))
    obs.sort(key=lambda x: (x.t, x.obs_id))
    return obs


def featurize_sector(sector_dir: str | Path, fs: FeatureSpace, ontology: Ontology,
                     cfg, max_event_group: int = 16) -> ScenarioFeatures:
    sector_dir = Path(sector_dir)
    obs = load_sector_observations(sector_dir)

    pairs = generate_candidates_grid(
        obs, ontology,
        dt_max_s=cfg.blocking.dt_max_s, theta_text=cfg.blocking.theta_text,
        r_err_floor_m=cfg.blocking.r_err_floor_m,
        reach_vmax_mult=cfg.blocking.reach_vmax_mult,
        reach_extra_m=cfg.blocking.reach_extra_m,
        cell_size_m=getattr(cfg.blocking, "grid_cell_m", 400.0),
        use_text_gate=getattr(cfg.blocking, "use_text_gate", True),
        type_by_category=getattr(cfg.blocking, "type_by_category", False))

    gold = json.loads((sector_dir / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
    obs_to_gold = {a["obs_id"]: a["gold_identity_id"] for a in gold.get("assignment", [])}
    dang_obs = set(gold.get("dangling_observations", []))

    pair_label = torch.zeros(len(pairs), dtype=torch.float32)
    for q, p in enumerate(pairs):
        if not p.cross_kg:
            continue
        a, b = obs[p.i].obs_id, obs[p.j].obs_id
        ga, gb = obs_to_gold.get(a), obs_to_gold.get(b)
        if ga is not None and ga == gb and a not in dang_obs and b not in dang_obs:
            pair_label[q] = 1.0

    dangling_label = torch.tensor([1.0 if o.obs_id in dang_obs else 0.0 for o in obs],
                                  dtype=torch.float32)

    return assemble_features(sector_dir.name, obs, pairs, ontology, fs,
                             pair_label, dangling_label, max_event_group=max_event_group)


def iter_sector_features(cfg, fs: FeatureSpace, ontology: Ontology, split: str,
                         sector_ids: list[str] | None = None) -> Iterator[ScenarioFeatures]:
    """Yield one sector's features at a time (streaming -- one sector in memory)."""
    root = Path(cfg.data_root)
    ids = sector_ids if sector_ids is not None else read_split(root, split)
    for sid in ids:
        yield featurize_sector(root / "sectors" / sid, fs, ontology, cfg)


def build_feature_space_large(cfg, max_label_sectors: int = 8) -> tuple[FeatureSpace, Ontology]:
    """Build the vocab from a few train sectors (enough to cover the paraphrase
    pools) without reading the whole suite."""
    ontology = load_ontology(cfg.ontology_dir)
    root = Path(cfg.data_root)
    train_ids = read_split(root, "train")[:max_label_sectors]
    labels: list[str] = []
    for sid in train_ids:
        jl = root / "sectors" / sid / "observations.jsonl"
        with open(jl, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    labels.append(json.loads(line)["label_text"])
    from ..model.text import Vocab
    fs = FeatureSpace(vocab=Vocab.build(labels), types=ontology.type_names,
                      states=ontology.state_names, sources=ontology.source_names,
                      max_tokens=cfg.encoder.text_max_tokens)
    return fs, ontology
