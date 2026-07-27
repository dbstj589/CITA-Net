"""P3 — serialisation of the true world (no observation fields anywhere).

unified_stkg.nt keeps the "1 line = 1 triple" convention (countable) and holds
the core graph (units, landmarks, events, entities, relation edges,
participatesIn). Time intervals for events/relations live in the JSONL sidecars;
the dense per-second state lives in the Parquet file; terrain in terrain/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import GEO, RDF, RDFS, XSD, GP, P, iri, lit


def nt_lines(entities, units, landmarks, events, relations) -> list[str]:
    """Triples for one batch (a sector or the whole world). Reused by the
    single-shot writer and the streaming writer so the format is identical."""
    lines: list[str] = []

    def T(s, p, o):
        lines.append(f"{s} {p} {o} .")

    for u in units.values():
        s = iri("unit", u.uid)
        T(s, f"<{RDF}type>", P("Unit"))
        T(s, f"<{RDFS}label>", lit(u.label))
        T(s, P("echelon"), lit(u.echelon))
        T(s, P("affiliation"), lit(u.affiliation))
        if u.parent:
            T(s, P("partOf"), iri("unit", u.parent))

    for lm in landmarks.values():
        s = iri("lm", lm.lmid)
        T(s, f"<{RDF}type>", P("Landmark"))
        T(s, f"<{RDFS}label>", lit(lm.name))
        T(s, GP("easting"), lit(round(lm.easting, 2), XSD + "double"))
        T(s, GP("northing"), lit(round(lm.northing, 2), XSD + "double"))
        T(s, GP("elevation"), lit(round(lm.elevation, 2), XSD + "double"))

    for e in events:
        s = iri("event", e.evid)
        T(s, f"<{RDF}type>", P("Event"))
        T(s, P("eventKind"), lit(e.kind))
        T(s, P("startTime"), lit(round(e.start, 3), XSD + "double"))
        T(s, P("endTime"), lit(round(e.end, 3), XSD + "double"))
        if e.landmark:
            T(s, P("atLandmark"), iri("lm", e.landmark))

    tgt_iri = {"entity": lambda r: iri("entity", r),
               "unit": lambda r: iri("unit", r),
               "landmark": lambda r: iri("lm", r)}
    for ent in entities:
        s = iri("entity", ent.eid)
        T(s, f"<{RDF}type>", P("Entity"))
        T(s, P("objectType"), lit(ent.true_type))
        T(s, P("affiliation"), lit(ent.affiliation))
        T(s, P("size"), lit(ent.size))
        if ent.parent_unit:
            T(s, P("partOf"), iri("unit", ent.parent_unit))
        for ev_id, _role in ent.events:
            T(s, P("participatesIn"), iri("event", ev_id))

    for r in relations:
        T(iri("entity", r.subject), P(r.predicate), tgt_iri[r.obj_kind](r.obj))

    return lines


def _write_nt(path: Path, entities, units, landmarks, events, relations) -> int:
    lines = nt_lines(entities, units, landmarks, events, relations)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _write_jsonl(path: Path, records) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_parquet(path: Path, rows) -> None:
    cols = ["entity_id", "t", "easting", "northing", "elevation", "state", "speed", "heading"]
    table = pa.table({c: [row[c] for row in rows] for c in cols})
    pq.write_table(table, path)


def _write_terrain(tdir: Path, terrain, terrain_meta, landmarks) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    np.save(tdir / "elevation.npy", terrain.elevation)
    np.save(tdir / "concealment.npy", terrain.concealment)
    (tdir / "terrain_meta.json").write_text(
        json.dumps(terrain_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    lm_out = [{"id": lm.lmid, "name": lm.name, "easting": round(lm.easting, 2),
               "northing": round(lm.northing, 2), "elevation": round(lm.elevation, 2)}
              for lm in landmarks.values()]
    (tdir / "landmarks.json").write_text(
        json.dumps(lm_out, ensure_ascii=False, indent=2), encoding="utf-8")


def serialize_world(outdir: Path, entities, units, landmarks, events, relations,
                    traj_rows, terrain, terrain_meta) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    n_triples = _write_nt(outdir / "unified_stkg.nt", entities, units, landmarks, events, relations)

    _write_jsonl(outdir / "entities.jsonl", [
        {"id": e.eid, "true_type": e.true_type, "affiliation": e.affiliation,
         "size": e.size, "parent_unit": e.parent_unit} for e in entities])

    _write_jsonl(outdir / "events.jsonl", [
        {"id": e.evid, "type": e.kind, "interval": [round(e.start, 3), round(e.end, 3)],
         "participants": e.participants, "landmark": e.landmark} for e in events])

    _write_jsonl(outdir / "relations.jsonl", [
        {"subject": r.subject, "predicate": r.predicate, "object": r.obj,
         "object_kind": r.obj_kind, "interval": [round(r.start, 3), round(r.end, 3)]}
        for r in relations])

    _write_parquet(outdir / "trajectories.parquet", traj_rows)
    _write_terrain(outdir / "terrain", terrain, terrain_meta, landmarks)

    return {"n_triples": n_triples, "n_entities": len(entities), "n_units": len(units),
            "n_landmarks": len(landmarks), "n_events": len(events),
            "n_relations": len(relations), "n_traj_rows": len(traj_rows)}
