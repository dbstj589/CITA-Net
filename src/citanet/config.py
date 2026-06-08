"""Configuration dataclasses for CITA-Net.

Configs are authored as YAML (see ``configs/``) and loaded into nested
dataclasses. Every hyper-parameter referenced in the methodology (§6) is
exposed here so experiments are fully reproducible from a single file.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BlockingConfig:
    """Candidate-generation / blocking thresholds (§4.1)."""

    dt_max_s: float = 180.0          # max forward time gap for a candidate pair
    theta_text: float = 0.3         # min cosine text similarity
    r_err_floor_m: float = 5.0      # additive slack on top of cep_i + cep_j
    allow_unknown_type: bool = True  # UNKNOWN is type-compatible with anything
    # Reach-radius inflation. Defaults keep the reach gate == the kinematic
    # bound. Raising these RELAXES blocking so kinematically-impossible pairs
    # survive to the CTA, letting the soft b_motion term (not the hard blocking
    # gate) do the rejecting -- used by the M3 b_motion ablation.
    reach_vmax_mult: float = 1.0
    reach_extra_m: float = 0.0


@dataclass
class EncoderConfig:
    """Source-aware multimodal encoder (§4.2)."""

    d_model: int = 128
    text_encoder: str = "learned"    # "learned" (offline) | "sbert"
    sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    text_max_tokens: int = 16
    use_source_gate: bool = True
    spatiotemporal_freqs: int = 8    # Fourier feature bands for (x, y, t)


@dataclass
class GraphConfig:
    """Relation-context GNN (§4.3)."""

    enabled: bool = False            # M2+ turns this on
    gnn_layers: int = 2
    n_heads: int = 4
    hops: int = 2


@dataclass
class CTAConfig:
    """Constraint-aware transition attention (§4.4)."""

    time_eps_s: float = 5.0          # clock-skew tolerance
    big_penalty: float = 50.0        # -BIG for backward-in-time pairs
    motion_beta: float = 2.0         # softplus temperature on speed violation
    enabled_terms: list[str] = field(
        default_factory=lambda: ["sem", "time", "motion", "state", "rel", "src"]
    )


@dataclass
class DecoderConfig:
    """Latent identity-trajectory decoder (§4.5)."""

    enabled: bool = False            # M3 turns this on
    num_slots: int = 16              # K identity slots (+1 implicit null slot)
    sinkhorn_iters: int = 30
    sinkhorn_temp: float = 0.5
    # Relative column-balance strength in the row-stochastic Sinkhorn. Kept low:
    # strong (uniform) column marginals fragment identities of unequal size.
    sinkhorn_col_strength: float = 0.1


@dataclass
class LossConfig:
    """Loss weights (§4.6)."""

    lambda_pair: float = 1.0
    lambda_transition: float = 1.0
    lambda_trajectory: float = 0.5
    lambda_dangling: float = 1.0
    lambda_assign: float = 1.0
    pair_pos_weight: float = 3.0     # class weight for the rare positive pairs


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 200
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    log_every: int = 20


@dataclass
class Config:
    """Top-level config aggregating every sub-module."""

    stage: str = "cita_lite"         # cita_lite | cita_g | cita_full
    seed: int = 20250606
    device: str = "auto"             # auto | cpu | cuda
    data_root: str = "data/battlefield_stkg_dataset"
    ontology_dir: str = "data/battlefield_stkg_dataset/ontology"

    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    cta: CTAConfig = field(default_factory=CTAConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def _merge(dc: Any, overrides: dict[str, Any]) -> Any:
    """Recursively apply a (possibly partial) dict onto a dataclass instance."""
    if not dataclasses.is_dataclass(dc) or not isinstance(overrides, dict):
        return overrides
    fields = {f.name: f for f in dataclasses.fields(dc)}
    for key, val in overrides.items():
        if key not in fields:
            raise KeyError(f"Unknown config key '{key}' for {type(dc).__name__}")
        cur = getattr(dc, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            setattr(dc, key, _merge(cur, val))
        else:
            setattr(dc, key, val)
    return dc


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a :class:`Config`, applying it over defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _merge(Config(), raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
