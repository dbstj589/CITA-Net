#!/usr/bin/env python
"""Extract quantitative + qualitative results for the Hill 395 experiment.

Reloads the trained model from runs/cita_full_large/ (no retrain), runs streaming
decode over dev+test sectors, and writes a self-contained results folder:

    hill395_experiment_results/
      README.md                  - human-readable index of everything
      metrics_summary.md/.json   - dev/test aggregate metrics + baseline compare
      per_sector_metrics.csv     - per-sector metrics (dev+test)
      training_curve.csv         - loss / dev-F1 per logged epoch
      dataset_stats.json         - triple/obs counts, type/state/source dist, blocking
      qualitative_cases.md/.json - correct merges / wrong-merges / fragments /
                                   dangling / impossible-transitions, with Korean labels
      type_confusion.csv         - which object types get wrongly merged
      samples/output_*.json      - full Part-B decoded sectors (examples)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.decode import decode_full
from citanet.engine_large import load_large_bundle
from citanet.eval import aggregate, evaluate_full
from citanet.data.stream import featurize_sector, read_split
from citanet.serialize import build_part_b

REPO = Path(__file__).resolve().parents[1]
# Defaults reproduce the original single-run behaviour; override via CLI to point
# the extractor at any trained run directory (e.g. an ablation variant).
DEFAULT_CFG = REPO / "configs" / "cita_full_hill395.yaml"
DEFAULT_RUN = REPO / "runs" / "cita_full_large"  # train_large.py default run dir
DEFAULT_LOG = REPO / "runs" / "hill395_train.log"
DEFAULT_OUT = REPO / "hill395_experiment_results"
DATA = REPO / "data" / "battlefield_hill395_large"

# original large-suite baseline (README-quoted) for context
BASELINE = {"precision": 0.80, "recall": 0.40, "f1": 0.52}
TABLE = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
         "trajectory_consistency_rate", "impossible_transition_rate",
         "dangling_precision", "dangling_recall"]


def load_obs(sd: Path) -> dict:
    obs = {}
    for line in open(sd / "observations.jsonl", encoding="utf-8"):
        o = json.loads(line)
        obs[o["obs_id"]] = o
    return obs


def entity_summary(obs: dict, kg: str, lid: str):
    """Majority (type, label) and source set for a KG-local entity."""
    recs = [o for o in obs.values() if o["kg_id"] == kg and o["local_entity_id"] == lid]
    if not recs:
        return None
    types = Counter(o["type"] for o in recs)
    labels = Counter(o["label_text"] for o in recs)
    srcs = sorted({o["source"] for o in recs})
    return {"kg": kg, "local_entity_id": lid, "type": types.most_common(1)[0][0],
            "label": labels.most_common(1)[0][0], "sources": srcs, "n_obs": len(recs)}


def member_oids(idn):
    out = []
    for m in idn.member_obs:
        out.append(m[0] if isinstance(m, (tuple, list)) else m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN),
                    help="trained bundle directory to reload (train_large.py --run-dir)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help="where to write the results (metrics_summary etc.)")
    ap.add_argument("--log", default=str(DEFAULT_LOG),
                    help="training log to parse for the loss/dev-F1 curve")
    ap.add_argument("--metrics-only", action="store_true",
                    help="write only metrics_summary + per_sector_metrics; skip the "
                         "qualitative cases, samples, dataset_stats, type_confusion, "
                         "README and training curve (for ablation sweeps)")
    args = ap.parse_args()

    CFG, RUN, OUT, LOG = Path(args.config), Path(args.run_dir), Path(args.out_dir), Path(args.log)
    metrics_only = args.metrics_only

    OUT.mkdir(parents=True, exist_ok=True)
    if not metrics_only:
        (OUT / "samples").mkdir(exist_ok=True)
    cfg = load_config(CFG)
    model, fs, ont = load_large_bundle(cfg, RUN)

    # ---------------- training curve ----------------
    curve = []
    if not metrics_only and LOG.exists():
        for ln in LOG.read_text(encoding="utf-8").splitlines():
            m = re.match(r"epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+best_dev_f1=([-\d.]+)", ln)
            if m:
                curve.append({"epoch": int(m.group(1)), "loss": float(m.group(3)),
                              "best_dev_f1": float(m.group(4))})
        ps = {k: [] for k in ("pair", "dangling", "transition", "trajectory", "assign")}
    if not metrics_only:
        with open(OUT / "training_curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss", "best_dev_f1"])
            w.writeheader(); w.writerows(curve)

    # ---------------- evaluate + collect qualitative ----------------
    per_all, splits_agg = [], {}
    cases = {"correct_merges": [], "wrong_merges": [], "fragments": [],
             "dangling_correct": [], "dangling_missed": [], "impossible_transitions": []}
    confusion = Counter()
    sample_saved = 0

    for split in ("dev", "test"):
        per = []
        for sid in read_split(cfg.data_root, split):
            sd = Path(cfg.data_root) / "sectors" / sid
            obs = load_obs(sd)
            feats = featurize_sector(sd, fs, ont, cfg)
            with torch.no_grad():
                out = model(feats)
            res = decode_full(out, feats, ont)
            m = evaluate_full(res, feats, sd); m["scenario_id"] = sid; m["split"] = split
            per.append(m); per_all.append(m)

            if metrics_only:
                del feats, out
                continue

            gold = json.loads((sd / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
            gold_pairs = {(i["member_local_entities"]["A"], i["member_local_entities"]["B"])
                          for i in gold["identities"]
                          if "A" in i["member_local_entities"] and "B" in i["member_local_entities"]}
            gold_a2b = dict(gold_pairs)
            gold_true = {i["member_local_entities"].get("A"): i["true_type"] for i in gold["identities"]}
            ent_gold = {}
            for i in gold["identities"]:
                for kg, lid in i["member_local_entities"].items():
                    ent_gold[(kg, lid)] = i["gold_identity_id"]
            gold_dangling = {(d["kg_id"], d["local_entity_id"]) for d in gold["dangling_local_entities"]}
            recovered_a = set()

            for idn in res.identities:
                le = idn.local_entity_ids
                if "A" in le and "B" in le:
                    a, b = le["A"], le["B"]
                    recovered_a.add(a)
                    ea, eb = entity_summary(obs, "A", a), entity_summary(obs, "B", b)
                    correct = gold_a2b.get(a) == b
                    rec = {"sector": sid, "A": ea, "B": eb,
                           "match_confidence": round(float(idn.match_confidence), 3),
                           "pred_type": idn.type}
                    if correct and len(cases["correct_merges"]) < 25:
                        cases["correct_merges"].append(rec)
                    if not correct:
                        confusion[tuple(sorted([ea["type"] if ea else "?", eb["type"] if eb else "?"]))] += 1
                        if len(cases["wrong_merges"]) < 25:
                            rec["gold_A_true_type"] = gold_true.get(a, "?")
                            cases["wrong_merges"].append(rec)
                # impossible transitions (state/kinematic violations)
                for t in idn.transitions:
                    if not t.feasible_motion and len(cases["impossible_transitions"]) < 20:
                        fo, to = obs.get(t.from_obs), obs.get(t.to_obs)
                        if fo and to:
                            cases["impossible_transitions"].append({
                                "sector": sid, "from": {"state": fo["state"], "type": fo["type"], "label": fo["label_text"]},
                                "to": {"state": to["state"], "type": to["type"], "label": to["label_text"]},
                                "required_speed_mps": round(float(t.required_speed_mps), 2)})

            # fragments: gold matches not recovered
            for (a, b) in gold_pairs:
                if a not in recovered_a and len(cases["fragments"]) < 20:
                    ea, eb = entity_summary(obs, "A", a), entity_summary(obs, "B", b)
                    cases["fragments"].append({"sector": sid, "true_type": gold_true.get(a, "?"),
                                               "A": ea, "B": eb})
            # dangling correctness
            pred_dang = {(d["kg_id"], d["local_entity_id"]) for d in res.dangling}
            for (kg, lid) in (pred_dang & gold_dangling):
                if len(cases["dangling_correct"]) < 15:
                    es = entity_summary(obs, kg, lid)
                    if es: cases["dangling_correct"].append({"sector": sid, **es})
            for (kg, lid) in (gold_dangling - pred_dang):
                if len(cases["dangling_missed"]) < 15:
                    es = entity_summary(obs, kg, lid)
                    if es: cases["dangling_missed"].append({"sector": sid, **es})

            # save a few full Part-B decoded sectors
            if sample_saved < 3:
                doc = build_part_b(sid, cfg, res, feats)
                (OUT / "samples" / f"output_{sid}.json").write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
                sample_saved += 1
            del feats, out
        splits_agg[split] = aggregate(per)

    # ---------------- per-sector csv ----------------
    cols = ["scenario_id", "split", "n_gold_matches", "n_pred_matches", "precision",
            "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
            "trajectory_consistency_rate", "impossible_transition_rate",
            "dangling_precision", "dangling_recall"]
    with open(OUT / "per_sector_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m in per_all:
            w.writerow({k: (round(m[k], 4) if isinstance(m.get(k), float) else m.get(k)) for k in cols})

    # ---------------- metrics summary ----------------
    summary = {"dev": splits_agg["dev"], "test": splits_agg["test"], "baseline_large": BASELINE}
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 백마고지(Hill 395) 실험 — 정량 결과\n",
          f"학습 설정: `{cfg.stage}`  (run_dir=`{RUN.name}`).\n",
          "## dev / test 집계 지표\n",
          "| 지표 | dev | test | 기존 large suite(참고) |", "|---|---|---|---|"]
    for k in TABLE:
        base = BASELINE.get(k)
        md.append(f"| {k} | {splits_agg['dev'][k]:.4f} | {splits_agg['test'][k]:.4f} | "
                  f"{base if base is not None else '-'} |")
    md.append("\n*기존 large suite 참고치는 README 인용(F1~0.52 / P~0.80 / R~0.40).*\n")
    (OUT / "metrics_summary.md").write_text("\n".join(md), encoding="utf-8")

    if metrics_only:
        print("metrics-only ->", OUT)
        print("dev :", {k: round(splits_agg['dev'][k], 4) for k in ("precision", "recall", "f1")})
        print("test:", {k: round(splits_agg['test'][k], 4) for k in ("precision", "recall", "f1")})
        return

    # ---------------- type confusion ----------------
    with open(OUT / "type_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["type_pair", "wrong_merge_count"])
        for pair, c in confusion.most_common():
            w.writerow([" + ".join(pair), c])

    # ---------------- qualitative json + md ----------------
    (OUT / "qualitative_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt_ent(e):
        return f"{e['label']}({e['type']}, src={'/'.join(e['sources'])}, {e['n_obs']}obs)" if e else "?"

    q = ["# 백마고지(Hill 395) 실험 — 정성 결과 사례\n",
         "두 정보망(KG-A 아군 직접관측 / KG-B 항공·감청)이 같은 전장을 관측한 결과를, 모델이 개체정합한 사례.\n"]
    q.append(f"## ✅ 올바른 병합 (correct merges) — 예시 {len(cases['correct_merges'])}건\n")
    q.append("동일 실체를 A·B 양쪽에서 정확히 같은 개체로 합친 경우.\n")
    for c in cases["correct_merges"][:12]:
        q.append(f"- [{c['sector']}] A={fmt_ent(c['A'])} ↔ B={fmt_ent(c['B'])}  "
                 f"(예측타입 {c['pred_type']}, 확신도 {c['match_confidence']})")
    q.append(f"\n## ❌ 잘못된 병합 (wrong merges) — 예시 {len(cases['wrong_merges'])}건\n")
    q.append("서로 다른 실체를 같은 개체로 합친 오류. 동일 type 대량 hard negative에서 주로 발생.\n")
    for c in cases["wrong_merges"][:12]:
        q.append(f"- [{c['sector']}] A={fmt_ent(c['A'])} ↔ B={fmt_ent(c['B'])}  "
                 f"(A 정답타입 {c.get('gold_A_true_type')}, 확신도 {c['match_confidence']})")
    q.append(f"\n## ✂️ 단편화 (fragments / 미회수) — 예시 {len(cases['fragments'])}건\n")
    q.append("정답상 같은 실체인데 모델이 합치지 못한 경우(보수적 abstain). recall 손실 원인.\n")
    for c in cases["fragments"][:12]:
        q.append(f"- [{c['sector']}] 정답타입 {c['true_type']}: A={fmt_ent(c['A'])} / B={fmt_ent(c['B'])}")
    q.append(f"\n## 🚫 매칭불가 정탐 (dangling 정확) — 예시 {len(cases['dangling_correct'])}건\n")
    q.append("한쪽 KG에만 보인 부대를 올바르게 '매칭불가'로 판정.\n")
    for c in cases["dangling_correct"][:10]:
        q.append(f"- [{c['sector']}] {fmt_ent(c)} (KG-{c['kg']})")
    q.append(f"\n## ⚠️ 매칭불가 미탐 (dangling 놓침) — 예시 {len(cases['dangling_missed'])}건\n")
    for c in cases["dangling_missed"][:10]:
        q.append(f"- [{c['sector']}] {fmt_ent(c)} (KG-{c['kg']})")
    q.append(f"\n## 🛰️ 물리적 불가능 전이 (impossible transitions) — 예시 {len(cases['impossible_transitions'])}건\n")
    q.append("같은 개체로 묶였으나 운동학/상태상 불가능한 전이(예: Destroyed→이동, 과속).\n")
    for c in cases["impossible_transitions"][:12]:
        q.append(f"- [{c['sector']}] {c['from']['state']}({c['from']['type']}) → "
                 f"{c['to']['state']}({c['to']['type']}), 필요속도 {c['required_speed_mps']} m/s")
    q.append(f"\n## 🔀 타입 혼동 (wrong-merge 타입쌍 상위)\n")
    for pair, cnt in confusion.most_common(10):
        q.append(f"- {' + '.join(pair)}: {cnt}회")
    (OUT / "qualitative_cases.md").write_text("\n".join(q), encoding="utf-8")

    # ---------------- dataset stats ----------------
    gm = json.loads((DATA / "manifest_global.json").read_text(encoding="utf-8"))
    tc, sc, srcc = Counter(), Counter(), Counter()
    n_obs = 0
    for sid in (read_split(cfg.data_root, "train") + read_split(cfg.data_root, "dev")
                + read_split(cfg.data_root, "test")):
        for line in open(DATA / "sectors" / sid / "observations.jsonl", encoding="utf-8"):
            o = json.loads(line); n_obs += 1
            tc[o["type"]] += 1; sc[o["state"]] += 1; srcc[o["source"]] += 1
    stats = {"total_triples": gm["total_triples"], "sectors": gm["sectors"],
             "triples_per_split": gm["triples_per_split"], "knobs": gm["knobs"],
             "total_observations": n_obs,
             "type_distribution": dict(tc.most_common()),
             "state_distribution": dict(sc.most_common()),
             "source_distribution": dict(srcc.most_common())}
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- README index ----------------
    counts = {k: len(v) for k, v in cases.items()}
    readme = [
        "# 백마고지(Hill 395) STKG 실험 결과\n",
        f"- 데이터셋: `data/battlefield_hill395_large/` ({gm['total_triples']:,} 트리플, "
        f"{sum(gm['sectors'].values())} 섹터, {n_obs:,} 관측)",
        f"- 모델/설정: Full CITA-Net, `configs/cita_full_hill395.yaml`, 50 train 섹터 × 40 epoch",
        f"- **test F1 {splits_agg['test']['f1']:.3f} / precision {splits_agg['test']['precision']:.3f} / "
        f"recall {splits_agg['test']['recall']:.3f}**\n",
        "## 파일 목록",
        "| 파일 | 내용 |", "|---|---|",
        "| `metrics_summary.md` / `.json` | dev/test 집계 지표 + 기존 baseline 비교 |",
        "| `per_sector_metrics.csv` | 섹터별 지표(dev+test 25개 섹터) |",
        "| `training_curve.csv` | epoch별 loss / best dev-F1 |",
        "| `dataset_stats.json` | 트리플·관측 수, 타입/상태/소스 분포 |",
        "| `qualitative_cases.md` / `.json` | 정성 사례(병합/오병합/단편화/dangling/불가능전이) |",
        "| `type_confusion.csv` | 오병합이 잦은 타입쌍 |",
        "| `samples/output_*.json` | 디코드된 Part-B 섹터 전문(예시 3개) |\n",
        "## 정성 사례 수집 건수",
        f"- 올바른 병합 {counts['correct_merges']} / 잘못된 병합 {counts['wrong_merges']} / "
        f"단편화 {counts['fragments']}",
        f"- dangling 정탐 {counts['dangling_correct']} / 미탐 {counts['dangling_missed']} / "
        f"불가능 전이 {counts['impossible_transitions']}",
    ]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("results ->", OUT)
    print("test:", {k: round(splits_agg['test'][k], 4) for k in ("precision", "recall", "f1")})
    print("qualitative case counts:", counts)


if __name__ == "__main__":
    main()
