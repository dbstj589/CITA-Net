"""P3 (streaming) — write the GT sector-by-sector so a 1M-triple world never
sits fully in memory. unified_stkg.nt / *.jsonl are appended incrementally;
trajectories go through a single pyarrow ParquetWriter (one row-group per
sector). Terrain is written once by the caller (it spans the whole map).
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .serialize import nt_lines

_TRAJ_SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("t", pa.float64()),
    ("easting", pa.float64()), ("northing", pa.float64()), ("elevation", pa.float64()),
    ("state", pa.string()), ("speed", pa.float64()), ("heading", pa.float64()),
])


class StreamWriter:
    def __init__(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        self.nt = open(outdir / "unified_stkg.nt", "w", encoding="utf-8")
        self.ent = open(outdir / "entities.jsonl", "w", encoding="utf-8")
        self.ev = open(outdir / "events.jsonl", "w", encoding="utf-8")
        self.rel = open(outdir / "relations.jsonl", "w", encoding="utf-8")
        self.pqw = pq.ParquetWriter(outdir / "trajectories.parquet", _TRAJ_SCHEMA)
        self.totals = {"n_triples": 0, "n_entities": 0, "n_units": 0, "n_landmarks": 0,
                       "n_events": 0, "n_relations": 0, "n_traj_rows": 0, "n_sectors": 0}

    def write_sector(self, entities, units, landmarks, events, relations, traj_rows) -> dict:
        lines = nt_lines(entities, units, landmarks, events, relations)
        self.nt.write("\n".join(lines) + ("\n" if lines else ""))
        for e in entities:
            self.ent.write(json.dumps(
                {"id": e.eid, "true_type": e.true_type, "affiliation": e.affiliation,
                 "size": e.size, "parent_unit": e.parent_unit}, ensure_ascii=False) + "\n")
        for e in events:
            self.ev.write(json.dumps(
                {"id": e.evid, "type": e.kind, "interval": [round(e.start, 3), round(e.end, 3)],
                 "participants": e.participants, "landmark": e.landmark}, ensure_ascii=False) + "\n")
        for r in relations:
            self.rel.write(json.dumps(
                {"subject": r.subject, "predicate": r.predicate, "object": r.obj,
                 "object_kind": r.obj_kind, "interval": [round(r.start, 3), round(r.end, 3)]},
                ensure_ascii=False) + "\n")
        if traj_rows:
            cols = ["entity_id", "t", "easting", "northing", "elevation", "state", "speed", "heading"]
            self.pqw.write_table(pa.table({c: [row[c] for row in traj_rows] for c in cols},
                                          schema=_TRAJ_SCHEMA))
        sc = {"n_triples": len(lines), "n_entities": len(entities), "n_units": len(units),
              "n_landmarks": len(landmarks), "n_events": len(events),
              "n_relations": len(relations), "n_traj_rows": len(traj_rows)}
        for k, v in sc.items():
            self.totals[k] += v
        self.totals["n_sectors"] += 1
        return sc

    def close(self) -> dict:
        self.nt.close(); self.ent.close(); self.ev.close(); self.rel.close()
        self.pqw.close()
        return self.totals
