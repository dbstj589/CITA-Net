"""P1 (events/relations) — first-class relation edges with time intervals.

Events are already produced as first-class nodes by ``entities.build_world``
(kind, [start,end], participants, landmark). This module turns each entity's
relations into interval-stamped edges: structural relations hold for the whole
window; dynamic ones (occupies/firesAt/withdrawsFrom/engagedWith) are stamped
from the entity's state schedule.
"""
from __future__ import annotations

from .common import RelationEdge

_WHOLE_WINDOW = {"partOf", "emplacedAt", "supports", "reinforces", "screens",
                 "near", "movesToward", "follows"}
_STATE_FOR_PRED = {
    "occupies": {"Occupying"},
    "firesAt": {"Firing"},
    "withdrawsFrom": {"Withdrawing"},
    "engagedWith": {"Engaging"},
}


def _first_state_time(states, wanted):
    for tf, s in states:
        if s in wanted:
            return float(tf)
    return None


def build_relations(entities, cfg) -> list[RelationEdge]:
    W = float(cfg["time"]["window_seconds"])
    edges: list[RelationEdge] = []
    for ent in entities:
        for pred, target, kind in ent.relations:
            if pred in _WHOLE_WINDOW:
                start, end = 0.0, W
            elif pred in _STATE_FOR_PRED:
                t0 = _first_state_time(ent.states, _STATE_FOR_PRED[pred])
                start, end = (0.0 if t0 is None else t0), W
            else:
                start, end = 0.0, W
            edges.append(RelationEdge(ent.eid, pred, target, kind, start, end))
    return edges
