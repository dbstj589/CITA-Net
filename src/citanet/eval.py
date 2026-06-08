"""Evaluation metrics (§7).

Entity alignment : Precision / Recall / F1, Hits@1, MRR.
Dangling         : Precision / Recall (+ residual alignment accuracy).
Merge quality    : Wrong-Merge Rate, Identity Fragmentation Rate.
Trajectory       : Trajectory Consistency Rate, Impossible Transition Rate
                   (computed in M3 from decoded trajectories).

Predictions are matched to gold by exact (A-local, B-local) entity pairs in the
lite/g decode; M3 adds slot<->gold overlap matching for the full decoder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .decode import DecodeResult


def _gold(scenario_dir: str | Path) -> dict[str, Any]:
    p = Path(scenario_dir) / "labels" / "gold_identities.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate_decode(res: DecodeResult, scenario_dir: str | Path) -> dict[str, Any]:
    gold = _gold(scenario_dir)
    gold_pairs = {(i["member_local_entities"]["A"], i["member_local_entities"]["B"])
                  for i in gold["identities"]
                  if "A" in i["member_local_entities"] and "B" in i["member_local_entities"]}
    gold_a_to_b = {a: b for (a, b) in gold_pairs}
    gold_dangling = {(d["kg_id"], d["local_entity_id"]) for d in gold["dangling_local_entities"]}

    pred_pairs = {(idn.a_entity, idn.b_entity) for idn in res.identities}

    tp = len(pred_pairs & gold_pairs)
    precision = _safe_div(tp, len(pred_pairs))
    recall = _safe_div(tp, len(gold_pairs))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # Wrong-merge: predicted match whose two sides are NOT the same gold identity.
    wrong = sum(1 for (a, b) in pred_pairs if gold_a_to_b.get(a) != b)
    wrong_merge_rate = _safe_div(wrong, len(pred_pairs))

    # Fragmentation: a gold A-entity split across >1 predicted match (impossible
    # in one-to-one decode, but kept for parity with M3). Here: gold matches with
    # no prediction count as fragmented-out (unrecovered).
    recovered_a = {a for (a, _) in pred_pairs}
    fragmentation_rate = _safe_div(
        sum(1 for (a, _) in gold_pairs if a not in recovered_a), len(gold_pairs))

    # Hits@1 / MRR over gold A-entities, ranking B by affinity.
    hits, rr, ranked_n = 0, 0.0, 0
    for a, b_gold in gold_a_to_b.items():
        cands = sorted(((bb, s) for (aa, bb), s in res.affinity.items() if aa == a),
                       key=lambda kv: kv[1], reverse=True)
        if not cands:
            ranked_n += 1
            continue
        ranked_n += 1
        order = [bb for bb, _ in cands]
        if b_gold in order:
            rank = order.index(b_gold) + 1
            rr += 1.0 / rank
            if rank == 1:
                hits += 1
    hits1 = _safe_div(hits, ranked_n)
    mrr = _safe_div(rr, ranked_n)

    # Dangling P/R
    pred_dangling = {(e.kg_id, e.local_entity_id) for e in res.dangling}
    d_tp = len(pred_dangling & gold_dangling)
    d_precision = _safe_div(d_tp, len(pred_dangling))
    d_recall = _safe_div(d_tp, len(gold_dangling))

    return {
        "n_gold_matches": len(gold_pairs),
        "n_pred_matches": len(pred_pairs),
        "precision": precision, "recall": recall, "f1": f1,
        "hits@1": hits1, "mrr": mrr,
        "wrong_merge_rate": wrong_merge_rate,
        "fragmentation_rate": fragmentation_rate,
        "dangling_precision": d_precision, "dangling_recall": d_recall,
        "n_gold_dangling": len(gold_dangling), "n_pred_dangling": len(pred_dangling),
    }


def evaluate_full(result, feats, scenario_dir: str | Path) -> dict[str, Any]:
    """M3 metrics from the full (Sinkhorn) decode: entity P/R/F1 plus
    Wrong-Merge Rate, Identity Fragmentation Rate, Trajectory Consistency Rate,
    and Impossible Transition Rate (§7)."""
    gold = _gold(scenario_dir)
    gold_pairs = {(i["member_local_entities"]["A"], i["member_local_entities"]["B"])
                  for i in gold["identities"]
                  if "A" in i["member_local_entities"] and "B" in i["member_local_entities"]}
    gold_a_to_b = {a: b for (a, b) in gold_pairs}
    gold_dangling = {(d["kg_id"], d["local_entity_id"]) for d in gold["dangling_local_entities"]}

    # entity (kg, local) -> gold id (or None if dangling)
    ent_gold: dict[tuple[str, str], str] = {}
    for i in gold["identities"]:
        for kg, lid in i["member_local_entities"].items():
            ent_gold[(kg, lid)] = i["gold_identity_id"]

    pred_pairs = set()
    for idn in result.identities:
        le = idn.local_entity_ids
        if "A" in le and "B" in le:
            pred_pairs.add((le["A"], le["B"]))

    tp = len(pred_pairs & gold_pairs)
    precision = _safe_div(tp, len(pred_pairs))
    recall = _safe_div(tp, len(gold_pairs))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # member entities per predicted identity (incl. conflicts) -> gold ids
    wrong = 0
    gold_to_preds: dict[str, set[int]] = {}
    for k, idn in enumerate(result.identities):
        member_entities = set()
        for oid, _ in idn.member_obs:
            member_entities.add(feats.entity_of[oid])
        gids = set()
        for ekey in member_entities:
            kg, lid = ekey.split(":", 1)
            g = ent_gold.get((kg, lid))
            if g is not None:
                gids.add(g)
                gold_to_preds.setdefault(g, set()).add(k)
        if len(gids) > 1:
            wrong += 1
    wrong_merge_rate = _safe_div(wrong, len(result.identities))

    fragmented = sum(1 for g, preds in gold_to_preds.items() if len(preds) > 1)
    # gold identities not recovered at all also count against fragmentation/recall
    unrecovered = sum(1 for (a, b) in gold_pairs
                      if gold_a_to_b.get(a) not in {p[1] for p in pred_pairs if p[0] == a})
    fragmentation_rate = _safe_div(fragmented + unrecovered, len(gold_pairs))

    n_trans = sum(len(idn.transitions) for idn in result.identities)
    n_impossible = sum(1 for idn in result.identities for t in idn.transitions
                       if not t.feasible_motion)
    impossible_transition_rate = _safe_div(n_impossible, n_trans)
    trajectory_consistency_rate = 1.0 - impossible_transition_rate if n_trans else 1.0

    pred_dangling = {(d["kg_id"], d["local_entity_id"]) for d in result.dangling}
    d_tp = len(pred_dangling & gold_dangling)
    d_precision = _safe_div(d_tp, len(pred_dangling))
    d_recall = _safe_div(d_tp, len(gold_dangling))

    return {
        "n_gold_matches": len(gold_pairs), "n_pred_matches": len(pred_pairs),
        "precision": precision, "recall": recall, "f1": f1,
        "wrong_merge_rate": wrong_merge_rate,
        "fragmentation_rate": fragmentation_rate,
        "trajectory_consistency_rate": trajectory_consistency_rate,
        "impossible_transition_rate": impossible_transition_rate,
        "n_transitions": n_trans, "n_impossible_transitions": n_impossible,
        "dangling_precision": d_precision, "dangling_recall": d_recall,
        "n_gold_dangling": len(gold_dangling), "n_pred_dangling": len(pred_dangling),
    }


def aggregate(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    """Macro-average numeric metrics across scenarios."""
    if not metrics_list:
        return {}
    keys = [k for k, v in metrics_list[0].items() if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        agg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return agg
