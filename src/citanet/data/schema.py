"""In-memory schema for a loaded scenario.

A :class:`Scenario` is the parsed, validated result of reading a scenario's
``kg_A.jsonld`` + ``kg_B.jsonld`` + ``commons.jsonld`` (the canonical path).
Downstream code (blocking, encoder, decoder, eval) consumes these objects.
Times are stored both as ISO strings and as float seconds relative to the
scenario epoch for cheap arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def parse_iso(ts: str) -> datetime:
    # Tolerate trailing 'Z' and fractional seconds.
    s = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


@dataclass
class Relation:
    predicate: str
    target_ref: str          # local entity id or commons id within scope
    confidence: float = 1.0


@dataclass
class EventRef:
    event_id: str
    role: str = "participant"


@dataclass
class Observation:
    obs_id: str
    kg_id: str               # "A" or "B"
    source: str
    local_entity_id: str
    label_text: str
    type: str
    type_confidence: float
    state: str
    state_confidence: float
    time_iso: str
    t: float                 # seconds since scenario epoch
    easting: float
    northing: float
    elevation: float = 0.0
    crs: str = "UTM52N"
    mgrs: str = ""
    cep_m: float = 30.0
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None
    relations: list[Relation] = field(default_factory=list)
    events: list[EventRef] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.easting, self.northing)


@dataclass
class Entity:
    """A local entity (track) within one KG -- groups its observations."""

    local_entity_id: str
    kg_id: str
    object_type: str
    relations: list[Relation] = field(default_factory=list)
    events: list[EventRef] = field(default_factory=list)
    obs_ids: list[str] = field(default_factory=list)


@dataclass
class Landmark:
    landmark_id: str
    name: str
    easting: float
    northing: float


@dataclass
class Event:
    event_id: str
    name: str
    kind: str = "engagement"


@dataclass
class Scenario:
    scenario_id: str
    epoch_iso: str
    observations: list[Observation]
    entities: dict[str, Entity]          # keyed by "<kg>:<local_id>"
    landmarks: dict[str, Landmark]
    events: dict[str, Event]
    manifest: dict[str, Any] = field(default_factory=dict)

    # ---- convenience views ---------------------------------------------
    def obs_by_kg(self, kg_id: str) -> list[Observation]:
        return [o for o in self.observations if o.kg_id == kg_id]

    @property
    def obs_index(self) -> dict[str, Observation]:
        return {o.obs_id: o for o in self.observations}

    def entity_key(self, kg_id: str, local_id: str) -> str:
        return f"{kg_id}:{local_id}"

    def n_entities(self, kg_id: str) -> int:
        return sum(1 for e in self.entities.values() if e.kg_id == kg_id)

    def epoch_seconds(self, iso: str) -> float:
        return (parse_iso(iso) - parse_iso(self.epoch_iso)).total_seconds()
