"""Training / evaluation engine shared by the CLI scripts.

Scenarios are small, so each is a single forward/backward (no minibatching) and
we iterate scenarios per epoch. Everything runs on CPU in minutes. Seeds are
fixed for reproducibility.
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
from .data.ontology import load_ontology
from .decode import decode_entities, decode_full
from .eval import aggregate, evaluate_decode, evaluate_full
from .losses import total_loss
from .model.citanet import CITANet
from .model.featurize import FeatureSpace, featurize_scenario


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _apply_determinism()


def _apply_determinism() -> None:
    """OPT-IN via env CITANET_DETERMINISTIC=1: force bit-reproducible GPU execution.

    Unset (the default) leaves every call below untouched, so the existing CPU and GPU
    paths stay byte-identical -- frozen / amb160 / study-1 results are unaffected.

    Why this exists: CUDA scatter/atomic accumulation order varies between runs, and over
    40 epochs that amplifies enough that two runs of the SAME seed diverge. Study 1
    measured this directly -- `no_time` is a null manipulation on this data (b_time is
    structurally 0, see docs/PREREG_realistic_v1_study2.md appendix A) yet its dF1 spread
    was +-0.078, i.e. as large as every real effect being tested.

    CUBLAS_WORKSPACE_CONFIG must be set before the cuBLAS handle is created. The runner
    exports it too; setting it here is the fallback, and we warn when CUDA is already
    initialised because at that point it can no longer take effect.
    """
    import os
    if os.environ.get("CITANET_DETERMINISTIC") != "1":
        return
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            print("!! CITANET_DETERMINISTIC=1 but CUBLAS_WORKSPACE_CONFIG was unset and CUDA "
                  "is already initialised -- export it before starting python")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_split(data_root: str | Path, split: str) -> list[str]:
    p = Path(data_root) / "splits" / f"{split}.txt"
    return [s for s in p.read_text(encoding="utf-8").split() if s]


@dataclass
class Bundle:
    cfg: Config
    fs: FeatureSpace
    ontology: object
    model: CITANet
    train_feats: list
    dev_feats: list


def scenario_dir(cfg: Config, scenario_id: str) -> Path:
    return Path(cfg.data_root) / "scenarios" / scenario_id


def build_feature_space(cfg: Config) -> tuple[FeatureSpace, object]:
    ontology = load_ontology(cfg.ontology_dir)
    train_ids = read_split(cfg.data_root, "train")
    fs = FeatureSpace.build(cfg.data_root, ontology, train_ids,
                            max_tokens=cfg.encoder.text_max_tokens)
    return fs, ontology


def featurize_split(cfg: Config, fs: FeatureSpace, ontology, split: str) -> list:
    feats = []
    for sid in read_split(cfg.data_root, split):
        feats.append(featurize_scenario(scenario_dir(cfg, sid), fs, ontology, cfg))
    return feats


def build_model(cfg: Config, fs: FeatureSpace, ontology) -> CITANet:
    set_seed(cfg.seed)
    model = CITANet(fs, ontology, cfg)
    return model


def train(cfg: Config, verbose: bool = True) -> Bundle:
    set_seed(cfg.seed)
    fs, ontology = build_feature_space(cfg)
    train_feats = featurize_split(cfg, fs, ontology, "train")
    dev_feats = featurize_split(cfg, fs, ontology, "dev")
    model = build_model(cfg, fs, ontology)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    dev_ids = read_split(cfg.data_root, "dev")

    def dev_f1() -> float:
        if model.decoder is not None:
            agg, _ = evaluate_full_split(cfg, model, ontology, dev_feats, dev_ids)
        else:
            agg, _ = evaluate(cfg, model, dev_feats, dev_ids)
        return agg.get("f1", 0.0)

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        order = list(range(len(train_feats)))
        random.shuffle(order)
        epoch_parts: dict[str, float] = {}
        epoch_total = 0.0
        for k in order:
            feats = train_feats[k]
            out = model(feats)
            sdir = scenario_dir(cfg, feats.scenario_id)
            loss, parts = total_loss(out, feats, cfg, sdir)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            epoch_total += float(loss.detach())
            for kk, vv in parts.items():
                epoch_parts[kk] = epoch_parts.get(kk, 0.0) + vv
        # dev-based checkpoint selection: L_assign's permutation-sensitive target
        # makes late epochs unstable, so we keep the best-on-dev model, not the
        # last. Cheap given the tiny suite.
        if epoch % max(1, cfg.train.log_every // 4) == 0 or epoch == cfg.train.epochs:
            f1 = dev_f1()
            if f1 >= best_f1:
                best_f1 = f1
                best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch % cfg.train.log_every == 0 or epoch == 1):
            parts_str = " ".join(f"{k}={v / len(train_feats):.4f}" for k, v in epoch_parts.items())
            print(f"epoch {epoch:3d}/{cfg.train.epochs}  loss={epoch_total / len(train_feats):.4f}  "
                  f"best_dev_f1={best_f1:.4f}  {parts_str}")

    model.load_state_dict(best_state)
    return Bundle(cfg=cfg, fs=fs, ontology=ontology, model=model,
                  train_feats=train_feats, dev_feats=dev_feats)


@torch.no_grad()
def evaluate(cfg: Config, model: CITANet, cfg_feats: list, split_ids: list[str],
             match_threshold: float = 0.5, dangling_threshold: float = 0.5
             ) -> tuple[dict, list[dict]]:
    model.eval()
    per_scenario = []
    for feats, sid in zip(cfg_feats, split_ids):
        out = model(feats)
        res = decode_entities(out, feats, match_threshold, dangling_threshold)
        m = evaluate_decode(res, scenario_dir(cfg, sid))
        m["scenario_id"] = sid
        per_scenario.append(m)
    return aggregate(per_scenario), per_scenario


@torch.no_grad()
def evaluate_full_split(cfg: Config, model: CITANet, ontology, feats_list: list,
                        split_ids: list[str]) -> tuple[dict, list[dict]]:
    """M3 evaluation via the Sinkhorn decoder (decode_full)."""
    model.eval()
    per = []
    for feats, sid in zip(feats_list, split_ids):
        out = model(feats)
        res = decode_full(out, feats, ontology)
        m = evaluate_full(res, feats, scenario_dir(cfg, sid))
        m["scenario_id"] = sid
        per.append(m)
    return aggregate(per), per


def save_bundle(bundle: Bundle, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.model.state_dict(), out_dir / "model.pt")
    bundle.fs.vocab.save(out_dir / "vocab.json")
    meta = {"types": bundle.fs.types, "states": bundle.fs.states,
            "sources": bundle.fs.sources, "max_tokens": bundle.fs.max_tokens,
            "stage": bundle.cfg.stage}
    (out_dir / "feature_space.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                encoding="utf-8")


def load_bundle(cfg: Config, run_dir: str | Path) -> tuple[CITANet, FeatureSpace, object]:
    from .model.text import Vocab
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "feature_space.json").read_text(encoding="utf-8"))
    vocab = Vocab.load(run_dir / "vocab.json")
    fs = FeatureSpace(vocab=vocab, types=meta["types"], states=meta["states"],
                      sources=meta["sources"], max_tokens=meta["max_tokens"])
    ontology = load_ontology(cfg.ontology_dir)
    model = CITANet(fs, ontology, cfg)
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    return model, fs, ontology
