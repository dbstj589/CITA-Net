"""Kinematic feasibility between two observations (§4.4 b_motion, §4.6 L_traj).

required speed  = dist / dt
feasible speed  = v_max(type, state) + (cep_i + cep_j) / dt   (position slack)

A transition is *feasible* when required <= feasible. The CTA penalises the
violation softly; the trajectory loss penalises infeasible same-identity links.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ontology import Ontology
from .schema import Observation


@dataclass
class Feasibility:
    dt_s: float
    dist_m: float
    required_speed_mps: float
    feasible_speed_mps: float
    feasible: bool
    violation_mps: float          # max(0, required - feasible)


def feasibility(oi: Observation, oj: Observation, ontology: Ontology,
                eps_dt: float = 1e-3) -> Feasibility:
    dt = abs(oj.t - oi.t)
    dist = math.hypot(oi.easting - oj.easting, oi.northing - oj.northing)
    dt_eff = max(dt, eps_dt)
    required = dist / dt_eff
    # Use the more permissive ceiling of the two endpoints (objects may change
    # state between observations).
    vmax = max(ontology.v_max(oi.type, oi.state),
               ontology.v_max(oj.type, oj.state))
    feasible_speed = vmax + (oi.cep_m + oj.cep_m) / dt_eff
    violation = max(0.0, required - feasible_speed)
    return Feasibility(dt_s=dt, dist_m=dist, required_speed_mps=required,
                       feasible_speed_mps=feasible_speed,
                       feasible=(required <= feasible_speed),
                       violation_mps=violation)
