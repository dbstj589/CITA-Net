"""P3 (real) — streaming serialisation with per-category triples, reification and
duplicate elimination.

Triple categories (reported separately so no single one dominates):
  entity_identity  Entity type/objectType/affiliation/size
  unit_echelon     Unit type/label/echelon/affiliation/partOf (the org tree)
  landmark_other   Landmark type/label/geo
  event            Event type/kind/interval/atLandmark
  relation         plain (subject,predicate,object) edges — DISTINCT only
  trajectory_state reified StateSample nodes (time-density; moving-dense/static-sparse)
  reification      Statement nodes reifying events + time-varying tactical relations

Time-varying tactical relations are carried by reified statements (each interval
= a distinct statement), so the plain-edge layer stays duplicate-free. Every
sector's triples are de-duplicated before writing; global uniqueness also holds
because IDs are global. relations.jsonl keeps the full interval-stamped edges.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .common import GEO, RDF, RDFS, XSD, GP, P, iri, lit

_TRAJ_SCHEMA = pa.schema([("entity_id", pa.string()), ("t", pa.float64()),
                          ("easting", pa.float64()), ("northing", pa.float64()),
                          ("elevation", pa.float64()), ("state", pa.string()),
                          ("speed", pa.float64()), ("heading", pa.float64())])

CATEGORIES = ["entity_identity", "unit_echelon", "landmark_other", "event",
              "relation", "trajectory_state", "reification"]


def _cat_lines(entities, units, landmarks, events, relations, traj_rows, reify_preds, reify_cap, stmt_ctr):
    """Return {category: [lines]} for one sector (pre-dedup)."""
    cats = {c: [] for c in CATEGORIES}

    def add(cat, s, p, o):
        cats[cat].append(f"{s} {p} {o} .")

    for e in entities:
        s = iri("entity", e.eid)
        add("entity_identity", s, f"<{RDF}type>", P("Entity"))
        add("entity_identity", s, P("objectType"), lit(e.true_type))
        add("entity_identity", s, P("affiliation"), lit(e.affiliation))
        add("entity_identity", s, P("size"), lit(e.size))
        for evid, _role in e.events:
            add("relation", s, P("participatesIn"), iri("event", evid))

    for u in units.values():
        s = iri("unit", u.uid)
        add("unit_echelon", s, f"<{RDF}type>", P("Unit"))
        add("unit_echelon", s, f"<{RDFS}label>", lit(u.label))
        add("unit_echelon", s, P("echelon"), lit(u.echelon))
        add("unit_echelon", s, P("affiliation"), lit(u.affiliation))
        if u.parent:
            add("unit_echelon", s, P("partOf"), iri("unit", u.parent))

    for lm in landmarks.values():
        s = iri("lm", lm.lmid)
        add("landmark_other", s, f"<{RDF}type>", P("Landmark"))
        add("landmark_other", s, f"<{RDFS}label>", lit(lm.name))
        add("landmark_other", s, GP("easting"), lit(round(lm.easting, 2), XSD + "double"))
        add("landmark_other", s, GP("northing"), lit(round(lm.northing, 2), XSD + "double"))
        add("landmark_other", s, GP("elevation"), lit(round(lm.elevation, 2), XSD + "double"))

    for ev in events:
        s = iri("event", ev.evid)
        add("event", s, f"<{RDF}type>", P("Event"))
        add("event", s, P("eventKind"), lit(ev.kind))
        add("event", s, P("startTime"), lit(round(ev.start, 3), XSD + "double"))
        add("event", s, P("endTime"), lit(round(ev.end, 3), XSD + "double"))
        if ev.landmark:
            add("event", s, P("atLandmark"), iri("lm", ev.landmark))

    tgt = {"entity": lambda r: iri("entity", r), "unit": lambda r: iri("unit", r),
           "landmark": lambda r: iri("lm", r)}
    for r in relations:
        add("relation", iri("entity", r.subject), P(r.predicate), tgt[r.obj_kind](r.obj))

    # --- reification (events + time-varying tactical relations) ---
    for ev in events:
        stmt_ctr[0] += 1; s = iri("stmt", f"ev{stmt_ctr[0]}")
        add("reification", s, f"<{RDF}type>", P("Statement"))
        add("reification", s, P("aboutEvent"), iri("event", ev.evid))
        add("reification", s, P("eventKind"), lit(ev.kind))
        add("reification", s, P("validFrom"), lit(round(ev.start, 3), XSD + "double"))
        add("reification", s, P("validTo"), lit(round(ev.end, 3), XSD + "double"))
        add("reification", s, P("participantCount"), lit(len(ev.participants), XSD + "integer"))
    n_reif = 0
    for r in relations:
        if n_reif >= reify_cap:
            break
        if r.predicate not in reify_preds:
            continue
        n_reif += 1; stmt_ctr[0] += 1; s = iri("stmt", f"r{stmt_ctr[0]}")
        add("reification", s, f"<{RDF}type>", P("Statement"))
        add("reification", s, f"<{RDF}subject>", iri("entity", r.subject))
        add("reification", s, f"<{RDF}predicate>", P(r.predicate))
        add("reification", s, f"<{RDF}object>", tgt[r.obj_kind](r.obj))
        add("reification", s, P("validFrom"), lit(round(r.start, 3), XSD + "double"))
        add("reification", s, P("validTo"), lit(round(r.end, 3), XSD + "double"))

    # --- trajectory state samples (reified true-state observations over time) ---
    for row in traj_rows:
        stmt_ctr[0] += 1
        s = iri("sample", row["entity_id"], f"t{int(round(row['t']))}")
        add("trajectory_state", s, f"<{RDF}type>", P("StateSample"))
        add("trajectory_state", s, P("sampleOf"), iri("entity", row["entity_id"]))
        add("trajectory_state", s, P("atTime"), lit(round(row["t"], 3), XSD + "double"))
        add("trajectory_state", s, GP("easting"), lit(row["easting"], XSD + "double"))
        add("trajectory_state", s, GP("northing"), lit(row["northing"], XSD + "double"))
        add("trajectory_state", s, GP("elevation"), lit(row["elevation"], XSD + "double"))
        add("trajectory_state", s, P("hasState"), lit(row["state"]))
        add("trajectory_state", s, P("speedMps"), lit(row["speed"], XSD + "double"))
    return cats


class StreamWriterReal:
    def __init__(self, outdir: Path, reify_preds, reify_cap):
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        self.reify_preds = set(reify_preds); self.reify_cap = int(reify_cap)
        self.nt = open(outdir / "unified_stkg.nt", "w", encoding="utf-8")
        self.entf = open(outdir / "entities.jsonl", "w", encoding="utf-8")
        self.evf = open(outdir / "events.jsonl", "w", encoding="utf-8")
        self.relf = open(outdir / "relations.jsonl", "w", encoding="utf-8")
        self.pqw = pq.ParquetWriter(outdir / "trajectories.parquet", _TRAJ_SCHEMA)
        self.stmt_ctr = [0]
        self.cat_counts = {c: 0 for c in CATEGORIES}
        self.totals = {"n_triples": 0, "n_entities": 0, "n_units": 0, "n_landmarks": 0,
                       "n_events": 0, "n_relations": 0, "n_traj_rows": 0, "n_sectors": 0,
                       "n_dupes_removed": 0}

    def write_sector(self, entities, units, landmarks, events, relations, traj_rows):
        cats = _cat_lines(entities, units, landmarks, events, relations, traj_rows,
                          self.reify_preds, self.reify_cap, self.stmt_ctr)
        # per-category dedup, then a global (per-sector) dedup safety net
        seen = set(); out_lines = []
        for c in CATEGORIES:
            kept = 0
            for ln in cats[c]:
                if ln in seen:
                    self.totals["n_dupes_removed"] += 1; continue
                seen.add(ln); out_lines.append(ln); kept += 1
            self.cat_counts[c] += kept
        self.nt.write("\n".join(out_lines) + ("\n" if out_lines else ""))

        for e in entities:
            self.entf.write(json.dumps({"id": e.eid, "true_type": e.true_type,
                "affiliation": e.affiliation, "size": e.size, "parent_unit": e.parent_unit},
                ensure_ascii=False) + "\n")
        for ev in events:
            self.evf.write(json.dumps({"id": ev.evid, "type": ev.kind,
                "interval": [round(ev.start, 3), round(ev.end, 3)],
                "participants": ev.participants, "landmark": ev.landmark}, ensure_ascii=False) + "\n")
        for r in relations:
            self.relf.write(json.dumps({"subject": r.subject, "predicate": r.predicate,
                "object": r.obj, "object_kind": r.obj_kind,
                "interval": [round(r.start, 3), round(r.end, 3)]}, ensure_ascii=False) + "\n")
        if traj_rows:
            cols = ["entity_id", "t", "easting", "northing", "elevation", "state", "speed", "heading"]
            self.pqw.write_table(pa.table({c: [row[c] for row in traj_rows] for c in cols}, schema=_TRAJ_SCHEMA))

        self.totals["n_triples"] += len(out_lines)
        self.totals["n_entities"] += len(entities); self.totals["n_units"] += len(units)
        self.totals["n_landmarks"] += len(landmarks); self.totals["n_events"] += len(events)
        self.totals["n_relations"] += len(relations); self.totals["n_traj_rows"] += len(traj_rows)
        self.totals["n_sectors"] += 1

    def close(self):
        self.nt.close(); self.entf.close(); self.evf.close(); self.relf.close(); self.pqw.close()
        return self.totals, self.cat_counts
