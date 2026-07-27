"""Sector-streamed training / evaluation for the large suite.

Sectors are the batch unit. Each epoch optionally **subsamples** a few train
sectors (CPU-bounded) or, with ``full_sectors=True``, uses all of them. Features
are built per sector on the fly and discarded after the step, so peak memory is
one sector's tensors -- the whole suite is never resident.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sector_dir(cfg: Config, sid: str) -> Path:
    return Path(cfg.data_root) / "sectors" / sid


# ---------------------------------------------------------------------------
# Featurization disk cache (OPT-IN via env CITANET_FEAT_CACHE=<dir>).
# featurize_sector is deterministic in (data, blocking cfg, vocab/ontology) and
# INDEPENDENT of seed, cta.enabled_terms, and model weights -- so its output can
# be computed once and reused across every epoch AND every ablation run. This
# turns a ~58s/sector recompute into a ~5s/sector load with byte-identical
# tensors. When the env var is unset the code path is exactly the original
# (no caching), so other experiments are unaffected.
# ---------------------------------------------------------------------------
def _feat_sig(cfg: Config, fs) -> str:
    b = cfg.blocking
    payload = {
        "data_root": str(cfg.data_root),
        "ontology_dir": str(cfg.ontology_dir),
        "blocking": {k: getattr(b, k) for k in
                     ("dt_max_s", "theta_text", "r_err_floor_m", "reach_vmax_mult",
                      "reach_extra_m", "grid_cell_m", "use_text_gate", "type_by_category")},
        "text_max_tokens": cfg.encoder.text_max_tokens,
        "vocab": list(fs.vocab.itos), "types": list(fs.types),
        "states": list(fs.states), "sources": list(fs.sources),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


_PAIR_TENSOR_FIELDS = ("pair_i", "pair_j", "pair_cross", "p_dt", "p_dist",
                       "p_req_speed", "p_feas_speed", "p_state_compat", "p_src_i",
                       "p_src_j", "p_rel_jaccard", "p_text_cos", "pair_label")


def _subsample_pairs(feats, max_pairs: int, seed: int):
    """TRAIN-ONLY (OPT-IN via env CITANET_TRAIN_MAX_PAIRS). Keep ALL positive
    cross-KG pairs + a random sample of the remaining candidates up to `max_pairs`,
    so the per-sector pair tensors (the memory/compute bottleneck at ~1.47M pairs)
    shrink ~10x and fit an 8GB GPU. Evaluation is untouched (full candidate set).
    Obs-level tensors, edges, and the decoder path are unchanged."""
    P = int(feats.pair_i.shape[0])
    if not max_pairs or P <= max_pairs:
        return feats
    keep = torch.zeros(P, dtype=torch.bool)
    keep[(feats.pair_label > 0.5) & feats.pair_cross] = True     # all positives
    budget = max_pairs - int(keep.sum())
    if budget > 0:
        rest = (~keep).nonzero(as_tuple=True)[0].numpy()
        g = np.random.default_rng(seed)
        take = g.choice(rest, size=min(budget, rest.size), replace=False)
        keep[torch.from_numpy(take)] = True
    idx = keep.nonzero(as_tuple=True)[0]
    import copy as _copy
    sub = _copy.copy(feats)                       # shallow: obs-level fields shared
    for name in _PAIR_TENSOR_FIELDS:
        setattr(sub, name, getattr(feats, name)[idx])
    sub.pairs = [feats.pairs[k] for k in idx.tolist()]
    return sub


def _featurize(cfg: Config, sid: str, fs, ontology, cache_dir, sig):
    if not cache_dir:
        return featurize_sector(_sector_dir(cfg, sid), fs, ontology, cfg)
    path = Path(cache_dir) / f"{sid}__{sig}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    feats = featurize_sector(_sector_dir(cfg, sid), fs, ontology, cfg)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(feats, tmp)
    os.replace(tmp, path)                    # atomic publish (safe for sequential reuse)
    return feats


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
    # Evaluation scores the FULL candidate set (no subsampling), so on a small GPU
    # (8GB) its ~1.47M-pair tensors can OOM even under no_grad. Setting
    # CITANET_EVAL_ON_CPU=1 runs eval on CPU (never OOMs) while training stays on
    # the GPU. Unset (default) => eval on the model's own device, so a big-VRAM GPU
    # evaluates on-GPU and the CPU-only path stays byte-identical.
    orig_dev = next(model.parameters()).device
    eval_dev = torch.device("cpu") if os.environ.get("CITANET_EVAL_ON_CPU") else orig_dev
    if orig_dev != eval_dev:
        model.to(eval_dev)
    ids = read_split(cfg.data_root, split)
    if max_sectors is not None:
        ids = ids[:max_sectors]
    cache_dir = os.environ.get("CITANET_FEAT_CACHE")
    sig = _feat_sig(cfg, fs) if cache_dir else None
    per = []
    for sid in ids:                       # streaming: one sector at a time
        feats = _featurize(cfg, sid, fs, ontology, cache_dir, sig).to(eval_dev)
        sd = _sector_dir(cfg, sid)
        with torch.device(eval_dev):
            out = model(feats)
            # Decoder-aware: the Sinkhorn full decode requires the M3 decoder; the
            # M1/M2 ablations (decoder off) fall back to the greedy entity decode.
            if out.assign is not None:
                m = evaluate_full(decode_full(out, feats, ontology), feats, sd)
            else:
                m = evaluate_decode(decode_entities(out, feats), sd)
        m["scenario_id"] = sid
        per.append(m)
        del feats, out
    if orig_dev != eval_dev:
        model.to(orig_dev)                # restore for continued GPU training
    return aggregate(per), per


def train_large(cfg: Config, sectors_per_epoch: int = 12, full_sectors: bool = False,
                dev_eval_sectors: int = 6, verbose: bool = True) -> LargeBundle:
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    fs, ontology = build_feature_space_large(cfg)
    with torch.device(device):            # build params/buffers on the target device
        model = build_model(cfg, fs, ontology)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    train_ids = read_split(cfg.data_root, "train")
    rng = random.Random(cfg.seed)
    cache_dir = os.environ.get("CITANET_FEAT_CACHE")
    sig = _feat_sig(cfg, fs) if cache_dir else None
    max_pairs = int(os.environ["CITANET_TRAIN_MAX_PAIRS"]) if os.environ.get("CITANET_TRAIN_MAX_PAIRS") else None
    if verbose:
        print(f"== device={device} train_max_pairs={max_pairs} ==")

    best_f1, best_state = -1.0, copy.deepcopy(model.state_dict())
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        if full_sectors:
            batch = list(train_ids); rng.shuffle(batch)
        else:
            batch = rng.sample(train_ids, min(sectors_per_epoch, len(train_ids)))
        tot, parts_acc = 0.0, {}
        for sid in batch:                 # streaming
            feats = _featurize(cfg, sid, fs, ontology, cache_dir, sig)
            if max_pairs:                 # train-only negative subsampling (keeps all positives)
                ssig = (int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) + cfg.seed * 1009 + epoch) & 0xFFFFFFFF
                feats = _subsample_pairs(feats, max_pairs, ssig)
            feats = feats.to(device)
            with torch.device(device):    # no-op on CPU; places factory tensors on GPU
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
    # move to CPU so the checkpoint is loadable from a CPU-only venv (analysis)
    cpu_state = {k: v.detach().cpu() for k, v in bundle.model.state_dict().items()}
    torch.save(cpu_state, out_dir / "model.pt")
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
    device = cfg.resolved_device()
    with torch.device(device):
        model = build_model(cfg, fs, ontology)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return model, fs, ontology
