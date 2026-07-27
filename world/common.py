"""Shared constants, dataclasses and IRI helpers for the world-GT generator.

Conventions are taken verbatim from the existing Hill-395 generator
(``scripts/gen_dataset_hill395.py``): UTM-52S frame with BASE ≈ (300000,
4238000), 8-wide sector grid at SECTOR_PITCH, the same landmark set and roster
ratios, and the same N-Triples namespaces. The vocabulary (types/states/
relations/events) is NOT hard-coded as truth here — it is validated against the
loaded ontology (see ``world.validate``) so drift is impossible.

Difference from the sensor-suite generator: coordinates, entity IDs and time are
GLOBAL (a single frame + a single timeline). Sectors are only spatial tiles used
to place entities and control density; nothing is reset per tile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- coordinate frame (shared with the frozen suites) -----------------------
CRS = "UTM52S"
BASE_E, BASE_N = 300000.0, 4238000.0
SECTOR_PITCH = 6000.0          # metres between sector tile origins
GRID_COLS = 8                  # sectors per row (idx % 8, idx // 8)

# --- IRI namespaces (mirror the suite; distinct path segment so the GT graph is
#     never confused with a sensor suite) -----------------------------------
NS = "https://example.org/stkg/"
GEO = "https://example.org/geo/"
PROV = "http://www.w3.org/ns/prov#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
PATH_SEG = "hill395_gt"

# --- states that pin an entity in place (speed recorded as exactly 0) --------
#     matches the spec: Emplaced/Halted/Destroyed/Holding/Occupying/Firing.
STATIC_STATES = {"Emplaced", "Halted", "Destroyed", "Holding", "Occupying", "Firing"}

# --- event vocabulary (5 kinds; there is no events.yaml, the set is fixed
#     here and validated by world.validate) -----------------------------------
EVENT_KINDS = ["Attack", "Counterattack", "FireSupport", "Occupy", "Illuminate"]

# --- affiliation is a GT entity attribute (faction is already baked into the
#     type; this maps the few faction-neutral types to a plausible side). This
#     is a synthetic approximation and is documented as such. ------------------
AFFILIATION = {
    "ROK_Infantry": "ROK", "CCF_Infantry": "CCF",
    "US_Tank": "US", "ROK_Tank": "ROK",
    "ROK_Artillery": "ROK", "US_Artillery": "US", "CCF_Artillery": "CCF",
    "Mortar": "ROK", "CCF_AA": "CCF", "QuadFifty": "US",
    "Engineer": "ROK", "VehicleColumn": "CCF", "Searchlight": "US",
    "Unit": "NA", "UNKNOWN": "UNK",
}

# echelon/size label by ontology category (approximation, documented).
SIZE_BY_CATEGORY = {
    "infantry": "squad", "tank": "vehicle", "artillery": "section",
    "mortar": "section", "aa": "section", "engineer": "squad",
    "vehicle": "column", "support": "section", "formation": "regiment",
    "unknown": "unknown",
}

# landmark offsets (metres) from a sector origin (ox, oy). Same set/positions as
# the suite generator's landmark block.
LANDMARK_OFFSETS = {
    "crest": (1500, 600), "northtip": (1500, 1700), "northslope": (1500, 1100),
    "mlr": (1500, -400), "objA": (1500, 1000), "objB": (1500, 1400),
    "valley": (200, 200), "yokkok": (-600, 300),
}


def roster_counts(n: int) -> dict[str, int]:
    """Roster composition for ``n`` entities in a sector (same ratios as the
    suite generator). Returns a mapping of builder-group -> count."""
    return {
        "ccf_inf": max(6, round(n * 0.30)),
        "rok_inf": max(6, round(n * 0.28)),
        "tank": max(2, round(n * 0.10)),
        "arty": max(1, round(n * 0.08)),
        "mortar": max(1, round(n * 0.05)),
        "aa": max(1, round(n * 0.05)),
        "engr": max(1, round(n * 0.05)),
        "veh": max(1, round(n * 0.04)),
        "wreck": max(1, round(n * 0.05)),
        "searchlight": 1,
    }


def sector_origin(sector_idx: int, pitch: float = SECTOR_PITCH) -> tuple[float, float]:
    sx = (sector_idx % GRID_COLS) * pitch
    sy = (sector_idx // GRID_COLS) * pitch
    return BASE_E + sx, BASE_N + sy


# --- IRI helpers ------------------------------------------------------------
def iri(*parts) -> str:
    return "<" + NS + PATH_SEG + "/" + "/".join(str(p) for p in parts) + ">"


def P(name: str) -> str:
    return f"<{NS}{name}>"


def GP(name: str) -> str:
    return f"<{GEO}{name}>"


def lit(v, dtype: Optional[str] = None) -> str:
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"^^<{dtype}>' if dtype else f'"{s}"'


# ---------------------------------------------------------------------------
# Data structures (the omniscient world state)
# ---------------------------------------------------------------------------
@dataclass
class WorldEntity:
    eid: str
    true_type: str
    affiliation: str
    size: str
    parent_unit: Optional[str]
    sector: int
    waypoints: list[tuple[float, float, float]]     # (t, easting, northing)
    states: list[tuple[float, str]]                 # (t_start, state) step schedule
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # (predicate, target_id, target_kind)
    events: list[tuple[str, str]] = field(default_factory=list)          # (event_id, role)
    # lifecycle (multi-phase / long windows): the entity is only observable in
    # [t_enter, t_exit]. Defaults span the whole window (PoC behaviour).
    t_enter: float = 0.0
    t_exit: Optional[float] = None


@dataclass
class Unit:
    uid: str
    echelon: str
    affiliation: str
    label: str
    parent: Optional[str] = None


@dataclass
class Landmark:
    lmid: str
    name: str
    easting: float
    northing: float
    elevation: float = 0.0


@dataclass
class WorldEvent:
    evid: str
    kind: str
    sector: int
    start: float
    end: float
    landmark: Optional[str]
    participants: list[str] = field(default_factory=list)  # entity ids


@dataclass
class RelationEdge:
    subject: str
    predicate: str
    obj: str
    obj_kind: str            # entity | unit | landmark
    start: float
    end: float
