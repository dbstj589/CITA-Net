"""Sector-streamed training / evaluation for the large suite.

Sectors are the batch unit. Each epoch optionally **subsamples** a few train
sectors (CPU-bounded) or, with ``full_sectors=True``, uses all of them. Features
are built per sector on the fly and discarded after the step, so peak memory is
one sector's tensors -- the whole suite is never resident.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .decode import decode_entities, decode_full
from .engine import build_model
from .eval import aggregate, evaluate_decode, evaluate_full
from .losses import total_loss
from .model.citanet import CITANet
from .model.featurize import FeatureSpace
from .data.stream import (
    build_feature_space_large,
    featurize_sector,
    read_split,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sector_dir(cfg: Config, sid: str) -> Path:
    return Path(cfg.data_root) / "sectors" / sid


@dataclass
class LargeBundle:
    cfg: Config
    fs: FeatureSpace
    ontology: object
    model: CITANet


@torch.no_grad()
def evaluate_large(cfg: Config, model: CITANet, fs, ontology, split: str,
                   max_sectors: int | None = None) -> tuple[dict, list[dict]]:
    model.eval()
    ids = read_split(cfg.data_root, split)
    if max_sectors is not None:
        ids = ids[:max_sectors]
    per = []
    for sid in ids:                       # streaming: one sector at a time
        feats = featurize_sector(_sector_dir(cfg, sid), fs, ontology, cfg)
        out = model(feats)
        sd = _sector_dir(cfg, sid)
        # Decoder-aware: the Sinkhorn full decode requires the M3 decoder; the
        # M1/M2 ablations (decoder off) fall back to the greedy entity decode.
        # When the decoder is on this is byte-identical to the prior path.
        if out.assign is not None:
            m = evaluate_full(decode_full(out, feats, ontology), feats, sd)
        else:
            m = evaluate_decode(decode_entities(out, feats), sd)
        m["scenario_id"] = sid
        per.append(m)
        del feats, out
    return aggregate(per), per


def train_large(cfg: Config, sectors_per_epoch: int = 12, full_sectors: bool = False,
                dev_eval_sectors: int = 6, verbose: bool = True) -> LargeBundle:
    set_seed(cfg.seed)
    fs, ontology = build_feature_space_large(cfg)
    model = build_model(cfg, fs, ontology)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    train_ids = read_split(cfg.data_root, "train")
    rng = random.Random(cfg.seed)

    best_f1, best_state = -1.0, copy.deepcopy(model.state_dict())
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        if full_sectors:
            batch = list(train_ids); rng.shuffle(batch)
        else:
            batch = rng.sample(train_ids, min(sectors_per_epoch, len(train_ids)))
        tot, parts_acc = 0.0, {}
        for sid in batch:                 # streaming
            feats = featurize_sector(_sector_dir(cfg, sid), fs, ontology, cfg)
            out = model(feats)
            loss, parts = total_loss(out, feats, cfg, _sector_dir(cfg, sid))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            tot += float(loss.detach())
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v
            del feats, out
        if epoch % max(1, cfg.train.log_every // 2) == 0 or epoch == cfg.train.epochs:
            agg, _ = evaluate_large(cfg, model, fs, ontology, "dev", max_sectors=dev_eval_sectors)
            if agg.get("f1", 0.0) >= best_f1:
                best_f1 = agg["f1"]; best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch % cfg.train.log_every == 0 or epoch == 1):
            ps = " ".join(f"{k}={v / len(batch):.4f}" for k, v in parts_acc.items())
            print(f"epoch {epoch:3d}/{cfg.train.epochs}  loss={tot / len(batch):.4f}  "
                  f"best_dev_f1={best_f1:.4f}  {ps}")

    model.load_state_dict(best_state)
    return LargeBundle(cfg=cfg, fs=fs, ontology=ontology, model=model)


def save_large_bundle(bundle: LargeBundle, out_dir: str | Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.model.state_dict(), out_dir / "model.pt")
    bundle.fs.vocab.save(out_dir / "vocab.json")
    (out_dir / "feature_space.json").write_text(json.dumps(
        {"types": bundle.fs.types, "states": bundle.fs.states,
         "sources": bundle.fs.sources, "max_tokens": bundle.fs.max_tokens,
         "stage": bundle.cfg.stage}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_large_bundle(cfg: Config, run_dir: str | Path):
    from .data.ontology import load_ontology
    from .model.text import Vocab
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "feature_space.json").read_text(encoding="utf-8"))
    fs = FeatureSpace(vocab=Vocab.load(run_dir / "vocab.json"), types=meta["types"],
                      states=meta["states"], sources=meta["sources"],
                      max_tokens=meta["max_tokens"])
    ontology = load_ontology(cfg.ontology_dir)
    model = build_model(cfg, fs, ontology)
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    return model, fs, ontology
