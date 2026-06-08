"""Serialise a full decode into the spec Part-B output JSON."""
from __future__ import annotations

import dataclasses
from typing import Any

from .config import Config, config_to_dict
from .decode import FullDecodeResult
from .model.featurize import ScenarioFeatures
from .output_schema import SCHEMA_VERSION


def build_part_b(scenario_id: str, cfg: Config, result: FullDecodeResult,
                 feats: ScenarioFeatures) -> dict[str, Any]:
    identities = []
    for idn in result.identities:
        gid = f"G-{scenario_id}-{idn.slot:02d}"
        identities.append({
            "global_id": gid,
            "type": idn.type,
            "type_confidence": idn.type_confidence,
            "match_confidence": idn.match_confidence,
            "local_entity_ids": idn.local_entity_ids,        # {"A":..,"B":..} 1:1
            "members": [{"obs_id": oid, "assign_prob": round(p, 4)}
                        for oid, p in idn.member_obs],
            "trajectory": idn.trajectory,
            "transitions": [dataclasses.asdict(t) for t in idn.transitions],
            "rationale": idn.rationale,
            "fused_representative": idn.fused_representative,
            "conflicts": idn.conflicts,
        })

    n_impossible = sum(1 for idn in result.identities for t in idn.transitions
                       if not t.feasible_motion)
    n_transitions = sum(len(idn.transitions) for idn in result.identities)
    stats = {
        "n_identities": len(result.identities),
        "n_dangling": len(result.dangling),
        "n_observations": len(feats.obs_ids),
        "n_transitions": n_transitions,
        "n_impossible_transitions": n_impossible,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "model": {"name": "CITA-Net", "stage": cfg.stage},
        "config": config_to_dict(cfg),
        "identities": identities,
        "dangling": result.dangling,
        "stats": stats,
    }
