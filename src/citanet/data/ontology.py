"""Ontology access: classes (kinematics), states (compatibility), sources,
relations. Loaded once from the dataset's ``ontology/`` directory and shared
by the generator, blocking, the encoder gate prior, and the CTA constraints.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

UNKNOWN_TYPE = "UNKNOWN"


@dataclass
class Ontology:
    classes: dict[str, Any]
    states: dict[str, Any]
    sources: dict[str, Any]
    relations: dict[str, Any]

    # ---- classes / kinematics -------------------------------------------
    @property
    def type_names(self) -> list[str]:
        return list(self.classes.keys())

    def v_max(self, obj_type: str, state: str) -> float:
        """Max plausible ground speed (m/s) for a type in a given state."""
        cls = self.classes.get(obj_type) or self.classes.get(UNKNOWN_TYPE, {})
        table = cls.get("v_max_mps", {})
        if state in table:
            return float(table[state])
        return float(cls.get("default_v_max_mps", 25.0))

    def types_compatible(self, t1: str, t2: str) -> bool:
        """Type compatibility for blocking: equal, or one side UNKNOWN."""
        if t1 == t2:
            return True
        return UNKNOWN_TYPE in (t1, t2)

    # ---- states ---------------------------------------------------------
    @property
    def state_names(self) -> list[str]:
        return list(self.states.get("states", []))

    def state_compat(self, s1: str, s2: str, obj_type: str | None = None) -> float:
        """Co-reference plausibility C[s1, s2] (earlier s1 -> later s2)."""
        base = self.states.get("compatibility", {})
        val = base.get(s1, {}).get(s2)
        if val is None:
            val = 0.5
        overrides = self.states.get("overrides", {})
        if obj_type and obj_type in overrides:
            ov = overrides[obj_type].get(s1, {})
            if s2 in ov:
                val = ov[s2]
        return float(val)

    @property
    def state_eps(self) -> float:
        return float(self.states.get("eps", 0.01))

    def state_compat_matrix(self, obj_type: str | None = None) -> list[list[float]]:
        names = self.state_names
        return [
            [self.state_compat(a, b, obj_type) for b in names] for a in names
        ]

    # ---- sources --------------------------------------------------------
    @property
    def source_names(self) -> list[str]:
        return list(self.sources.keys())

    def source_reliability(self, src: str) -> float:
        return float(self.sources.get(src, {}).get("reliability", 0.5))

    def source_cep(self, src: str) -> float:
        return float(self.sources.get(src, {}).get("cep_m", 50.0))

    def source_field(self, src: str, field: str, default: float = 0.5) -> float:
        return float(self.sources.get(src, {}).get(field, default))

    # ---- relations ------------------------------------------------------
    @property
    def relation_names(self) -> list[str]:
        return list(self.relations.keys())

    def is_symmetric(self, predicate: str) -> bool:
        return bool(self.relations.get(predicate, {}).get("symmetric", False))


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=8)
def load_ontology(ontology_dir: str | Path) -> Ontology:
    d = Path(ontology_dir)
    return Ontology(
        classes=_load_yaml(d / "classes.yaml"),
        states=_load_yaml(d / "states.yaml"),
        sources=_load_yaml(d / "sources.yaml"),
        relations=_load_yaml(d / "relations.yaml"),
    )
