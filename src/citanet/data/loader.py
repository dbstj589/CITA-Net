"""Canonical scenario loader.

The *canonical* input path parses ``kg_A.jsonld`` + ``kg_B.jsonld`` +
``commons.jsonld`` into a :class:`~citanet.data.schema.Scenario`. The flattened
``observations.jsonl`` is treated as a derived cache: after parsing the STKG we
**assert** that the two agree (``validate_consistency``), per §3.2.

JSON-LD here is authored in compact form (context terms are short keys), so we
read the short keys directly with the standard ``json`` module -- no rdflib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ontology import load_ontology
from .schema import (
    Entity,
    Event,
    EventRef,
    Landmark,
    Observation,
    Relation,
    Scenario,
    parse_iso,
)

# Relation predicates we recognise as entity edges in the JSON-LD nodes.
_BASE_RELATIONS = [
    "partOf", "follows", "near", "firesAt", "engagedWith",
    "emplacedAt", "movesToward", "supports", "screens",
]


def _strip_prefix(iri: str) -> tuple[str, str]:
    """Split a compact IRI 'kgA:ent_A_001' -> ('kgA', 'ent_A_001')."""
    if ":" in iri:
        pre, local = iri.split(":", 1)
        return pre, local
    return "", iri


def _kg_of_prefix(pre: str) -> str:
    # 'kgA' -> 'A', 'kgB' -> 'B'
    if pre.startswith("kg") and len(pre) == 3:
        return pre[2]
    return ""


def _load_graph(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("@graph", [])


def load_scenario(scenario_dir: str | Path, ontology_dir: str | Path) -> Scenario:
    scenario_dir = Path(scenario_dir)
    scenario_id = scenario_dir.name
    manifest = json.loads((scenario_dir / "manifest.json").read_text(encoding="utf-8"))
    epoch_iso = manifest["epoch"]
    ont = load_ontology(ontology_dir)
    # Event-style predicates (participatesIn) are handled separately as events,
    # not as entity relations.
    event_preds = {p for p in ont.relation_names
                   if ont.relations.get(p, {}).get("event")}
    rel_predicates = (set(_BASE_RELATIONS) | set(ont.relation_names)) - event_preds - {"participatesIn"}

    # --- commons: landmarks + events -----------------------------------
    landmarks: dict[str, Landmark] = {}
    events: dict[str, Event] = {}
    for node in _load_graph(scenario_dir / "commons.jsonld"):
        _, local = _strip_prefix(node["@id"])
        ntype = node.get("@type", "")
        if ntype == "Landmark":
            landmarks[local] = Landmark(local, node.get("rdfs:label", local),
                                        float(node.get("easting", 0.0)),
                                        float(node.get("northing", 0.0)))
        elif ntype == "Event":
            events[local] = Event(local, node.get("rdfs:label", local),
                                  node.get("stkg:eventKind", "event"))

    entities: dict[str, Entity] = {}
    observations: list[Observation] = []

    def epoch_seconds(iso: str) -> float:
        return (parse_iso(iso) - parse_iso(epoch_iso)).total_seconds()

    for kg_file in ("kg_A.jsonld", "kg_B.jsonld"):
        for node in _load_graph(scenario_dir / kg_file):
            pre, local = _strip_prefix(node["@id"])
            kg = _kg_of_prefix(pre)
            ntype = node.get("@type", "")
            if ntype == "Entity":
                rels: list[Relation] = []
                for pred in rel_predicates:
                    if pred in node:
                        for tgt in _as_list(node[pred]):
                            _, tlocal = _strip_prefix(tgt)
                            rels.append(Relation(pred, tlocal, 1.0))
                evs = [EventRef(_strip_prefix(e)[1]) for e in _as_list(node.get("participatesIn", []))]
                ent = Entity(local_entity_id=local, kg_id=kg,
                             object_type=node.get("objectType", "UNKNOWN"),
                             relations=rels, events=evs)
                entities[f"{kg}:{local}"] = ent
            elif ntype == "Observation":
                _, ent_local = _strip_prefix(node["observationOf"])
                iso = node["time"]
                obs = Observation(
                    obs_id=local, kg_id=kg, source=node["source"],
                    local_entity_id=ent_local, label_text=node.get("labelText", ""),
                    type=node.get("objectType", "UNKNOWN"),
                    type_confidence=float(node.get("typeConfidence", 0.5)),
                    state=node.get("state", "Unknown"),
                    state_confidence=float(node.get("stateConfidence", 0.5)),
                    time_iso=iso, t=epoch_seconds(iso),
                    easting=float(node["easting"]), northing=float(node["northing"]),
                    elevation=float(node.get("elevation", 0.0)),
                    crs=node.get("crs", "UTM52N"), mgrs=node.get("mgrs", ""),
                    cep_m=float(node.get("cepM", 30.0)),
                    speed_mps=_opt_float(node.get("speedMps")),
                    heading_deg=_opt_float(node.get("headingDeg")),
                    provenance={"wasAttributedTo": node.get("wasAttributedTo"),
                                "generatedAtTime": node.get("generatedAtTime")},
                )
                observations.append(obs)

    # Attach entity relations/events onto each observation (flattened view)
    # and register obs ids on their entities.
    for obs in observations:
        ekey = f"{obs.kg_id}:{obs.local_entity_id}"
        ent = entities.get(ekey)
        if ent is not None:
            ent.obs_ids.append(obs.obs_id)
            obs.relations = list(ent.relations)
            obs.events = list(ent.events)

    observations.sort(key=lambda o: (o.time_iso, o.obs_id))
    return Scenario(scenario_id=scenario_id, epoch_iso=epoch_iso,
                    observations=observations, entities=entities,
                    landmarks=landmarks, events=events, manifest=manifest)


def _as_list(v: Any) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def validate_consistency(scenario_dir: str | Path, scn: Scenario,
                         tol: float = 1e-6) -> None:
    """Assert the parsed STKG matches the flattened observations.jsonl cache."""
    scenario_dir = Path(scenario_dir)
    cache = [json.loads(line) for line in
             (scenario_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    cache_by_id = {o["obs_id"]: o for o in cache}
    parsed_by_id = scn.obs_index

    assert len(cache) == len(parsed_by_id), (
        f"obs count mismatch: jsonl={len(cache)} stkg={len(parsed_by_id)}")
    assert set(cache_by_id) == set(parsed_by_id), "obs id set mismatch between jsonl and STKG"

    for oid, c in cache_by_id.items():
        p = parsed_by_id[oid]
        assert p.kg_id == c["kg_id"], f"{oid}: kg mismatch"
        assert p.source == c["source"], f"{oid}: source mismatch"
        assert p.local_entity_id == c["local_entity_id"], f"{oid}: entity mismatch"
        assert p.type == c["type"], f"{oid}: type mismatch {p.type} vs {c['type']}"
        assert p.state == c["state"], f"{oid}: state mismatch"
        assert p.label_text == c["label_text"], f"{oid}: label mismatch"
        assert p.time_iso == c["time"], f"{oid}: time mismatch"
        assert abs(p.easting - c["location"]["easting"]) < 1e-3, f"{oid}: easting mismatch"
        assert abs(p.northing - c["location"]["northing"]) < 1e-3, f"{oid}: northing mismatch"
        assert abs(p.cep_m - c["cep_m"]) < 1e-3, f"{oid}: cep mismatch"
        # relations (entity-level, flattened) -- compare predicate+target sets
        p_rel = {(r.predicate, r.target_ref) for r in p.relations}
        c_rel = {(r["predicate"], r["target_ref"]) for r in c.get("relations", [])}
        assert p_rel == c_rel, f"{oid}: relation set mismatch {p_rel} vs {c_rel}"
