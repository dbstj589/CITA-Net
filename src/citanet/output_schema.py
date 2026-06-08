"""CITA-Net output JSON (spec Part B) schema + a dependency-free validator.

The validator checks structure/types/enums without requiring the ``jsonschema``
package, so validation works fully offline. ``OUTPUT_JSON_SCHEMA`` is also
exported as a Draft-07 schema for external tooling.
"""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "cita-net-output/1.0"

OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CITA-Net entity-alignment output (Part B)",
    "type": "object",
    "required": ["schema_version", "scenario_id", "model", "config",
                 "identities", "dangling", "stats"],
    "properties": {
        "schema_version": {"type": "string"},
        "scenario_id": {"type": "string"},
        "model": {"type": "object"},
        "config": {"type": "object"},
        "identities": {"type": "array", "items": {"$ref": "#/$defs/identity"}},
        "dangling": {"type": "array", "items": {"$ref": "#/$defs/dangling"}},
        "pairwise": {"type": "array"},
        "stats": {"type": "object"},
    },
    "$defs": {
        "identity": {
            "type": "object",
            "required": ["global_id", "type", "type_confidence", "match_confidence",
                         "local_entity_ids", "members", "trajectory", "transitions",
                         "rationale"],
            "properties": {
                "global_id": {"type": "string"},
                "type": {"type": "string"},
                "type_confidence": {"type": "number"},
                "match_confidence": {"type": "number"},
                "local_entity_ids": {"type": "object"},
                "members": {"type": "array", "items": {
                    "type": "object", "required": ["obs_id", "assign_prob"]}},
                "trajectory": {"type": "array"},
                "transitions": {"type": "array", "items": {
                    "type": "object",
                    "required": ["from_obs", "to_obs", "prob", "feasible_motion",
                                 "required_speed_mps"]}},
                "rationale": {"type": "object",
                              "required": ["sim_sem", "b_time", "b_motion",
                                           "b_state", "b_rel", "b_src"]},
                "fused_representative": {"type": "object"},
                "conflicts": {"type": "array"},
            },
        },
        "dangling": {
            "type": "object",
            "required": ["kg_id", "local_entity_id", "obs_ids", "dangling_prob",
                         "decision", "reason"],
            "properties": {
                "kg_id": {"type": "string"},
                "local_entity_id": {"type": "string"},
                "obs_ids": {"type": "array"},
                "dangling_prob": {"type": "number"},
                "nearest_candidate": {"type": ["object", "null"]},
                "decision": {"type": "string", "enum": ["abstain"]},
                "reason": {"type": "string"},
            },
        },
    },
}


class SchemaError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_output(doc: dict[str, Any]) -> None:
    """Validate a Part-B output document; raise :class:`SchemaError` on failure."""
    _check(isinstance(doc, dict), "top-level must be an object")
    for key in ("schema_version", "scenario_id", "model", "config",
                "identities", "dangling", "stats"):
        _check(key in doc, f"missing top-level key '{key}'")
    _check(isinstance(doc["schema_version"], str), "schema_version must be str")
    _check(isinstance(doc["scenario_id"], str), "scenario_id must be str")
    _check(isinstance(doc["identities"], list), "identities must be a list")
    _check(isinstance(doc["dangling"], list), "dangling must be a list")
    _check(isinstance(doc["stats"], dict), "stats must be an object")

    for i, idn in enumerate(doc["identities"]):
        ctx = f"identities[{i}]"
        for key in ("global_id", "type", "type_confidence", "match_confidence",
                    "local_entity_ids", "members", "trajectory", "transitions",
                    "rationale"):
            _check(key in idn, f"{ctx}: missing '{key}'")
        _check(isinstance(idn["global_id"], str), f"{ctx}.global_id must be str")
        _check(_is_num(idn["type_confidence"]), f"{ctx}.type_confidence must be number")
        _check(_is_num(idn["match_confidence"]), f"{ctx}.match_confidence must be number")
        _check(isinstance(idn["local_entity_ids"], dict), f"{ctx}.local_entity_ids must be object")
        # two-level invariant: identity<->identity is one-to-one (<=1 per KG)
        for kg in idn["local_entity_ids"]:
            _check(kg in ("A", "B"), f"{ctx}.local_entity_ids key must be 'A'/'B'")
        for j, mem in enumerate(idn["members"]):
            _check("obs_id" in mem and "assign_prob" in mem,
                   f"{ctx}.members[{j}] needs obs_id+assign_prob")
        for key in ("sim_sem", "b_time", "b_motion", "b_state", "b_rel", "b_src"):
            _check(key in idn["rationale"], f"{ctx}.rationale missing '{key}'")
        for j, tr in enumerate(idn["transitions"]):
            for key in ("from_obs", "to_obs", "prob", "feasible_motion",
                        "required_speed_mps"):
                _check(key in tr, f"{ctx}.transitions[{j}] missing '{key}'")
            _check(isinstance(tr["feasible_motion"], bool),
                   f"{ctx}.transitions[{j}].feasible_motion must be bool")

    for i, d in enumerate(doc["dangling"]):
        ctx = f"dangling[{i}]"
        for key in ("kg_id", "local_entity_id", "obs_ids", "dangling_prob",
                    "decision", "reason"):
            _check(key in d, f"{ctx}: missing '{key}'")
        _check(d["decision"] == "abstain", f"{ctx}.decision must be 'abstain'")
        _check(isinstance(d["obs_ids"], list), f"{ctx}.obs_ids must be a list")
