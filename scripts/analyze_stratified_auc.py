"""Analysis B -- stratified discriminative power. EXPLORATORY. NO TRAINING.

Question: is a term's signal specific to the uncertainty it was designed for? For each
term we split the candidate pairs by a ground-truth condition that term targets and
compare its incremental dAUC (over w0*sem) inside vs outside that condition.

PREDICTIONS, recorded here before the numbers were computed (they are asserted by the
constant below and printed verbatim into the report):

    time   -- b_time discriminates better when the pair's reported times are further
              from the true times.
    motion -- b_motion discriminates better when at least one endpoint comes from a
              high-position-error source.
    state  -- b_state discriminates better when the entity's TRUE state actually changes
              inside the pair's time interval.
    rel    -- b_rel discriminates better when the entity's predicate combination is
              distinctive (few other entities share it).
    src    -- b_src discriminates better when the two sources are of DIFFERENT
              reliability classes (heterogeneous) than when both are high-reliability.

This analysis is EXPLORATORY and stays so regardless of outcome: the strata were chosen
after seeing study 1, the five contrasts are not corrected for multiplicity, and a stratum
also shifts the positive rate, which moves AUC on its own.

Note on the source strata: the dumped `both_low_cep` (motion stratum) and a naive
"both high reliability" flag turned out to select the IDENTICAL set of source pairs on
this ontology -- median CEP and median reliability happen to partition the seven sources
the same way. The src stratum is therefore defined here as homogeneous-vs-heterogeneous
reliability class, which is genuinely distinct from the motion stratum.

    python scripts/analyze_stratified_auc.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[1]
TERMS = ["b_time", "b_motion", "b_state", "b_rel", "b_src"]
B_BOOT = 400

PREDICTION = {
    "b_time":   "보고시각-참시각 오차가 큰 층에서 증분 판별력이 더 크다",
    "b_motion": "고오차 출처가 포함된 층에서 증분 판별력이 더 크다",
    "b_state":  "참 상태 전이가 구간 내에 있는 층에서 증분 판별력이 더 크다",
    "b_rel":    "술어조합이 희소(고유)한 층에서 증분 판별력이 더 크다",
    "b_src":    "출처 신뢰도 등급이 이질적인 층에서 증분 판별력이 더 크다",
}


def exact_auc(score, label):
    n_pos = int(label.sum()); n_neg = label.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable"); s = score[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks); r[order] = ranks
    return float((r[label == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=str(REPO / "results/realistic_v1_score_analysis/dump"))
    ap.add_argument("--out-dir", default=str(REPO / "results/realistic_v1_score_analysis"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    src_meta = yaml.safe_load((REPO / "data/realistic_v1/ontology/sources.yaml")
                              .read_text(encoding="utf-8"))
    print("== 분석 B (탐색적) — 사전 예측 ==")
    for k, v in PREDICTION.items():
        print(f"   {k:9} {v}")

    rows = []
    for path in sorted(Path(args.dump_dir).glob(f"m3_full_seed*_{args.split}.parquet")):
        seed = int(path.stem.split("_seed")[1].split("_")[0])
        t = pq.read_table(path)
        label = t["label"].to_numpy().astype(np.int8)
        sem = t["sem_w0"].to_numpy()
        si, sj = t["src_i"].to_numpy(), t["src_j"].to_numpy()
        # source reliability class (high/low by median) -> homogeneous vs heterogeneous
        names = list(src_meta)
        # dump stored fs.sources indices; rebuild the same order from the ontology file
        rel = np.array([float(src_meta[n]["reliability"]) for n in names], dtype=np.float32)
        hi = rel >= float(np.median(rel))
        het = (hi[si] != hi[sj]).astype(np.int8)

        strata = {
            "b_time":   ("time_err_sum > median", (t["time_err_sum"].to_numpy()
                                                   > np.median(t["time_err_sum"].to_numpy())).astype(np.int8)),
            "b_motion": ("high-error source present", (1 - t["both_low_cep"].to_numpy()).astype(np.int8)),
            "b_state":  ("true state transition in interval", t["gt_state_trans"].to_numpy()),
            "b_rel":    ("distinctive predicate combo", (t["rel_share_min"].to_numpy()
                                                         < np.median(t["rel_share_min"].to_numpy())).astype(np.int8)),
            "b_src":    ("heterogeneous source reliability", het),
        }
        for tm, (desc, flag) in strata.items():
            v = t[tm].to_numpy()
            for lvl, name in ((1, "in-condition"), (0, "out-of-condition")):
                m = flag == lvl
                if m.sum() == 0 or label[m].sum() == 0 or (label[m] == 0).sum() == 0:
                    continue
                d = exact_auc(sem[m] + v[m], label[m]) - exact_auc(sem[m], label[m])
                rows.append({"seed": seed, "split": args.split, "term": tm, "stratum": desc,
                             "level": name, "n": int(m.sum()),
                             "pos_rate": float(label[m].mean()), "d_auc_vs_sem": d})
        del t
        print(f"  [seed {seed}] done")

    with open(out / "stratified_auc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{'term':10} {'in-cond ΔAUC':>13} {'out ΔAUC':>11} {'차이':>10} {'예측 방향':>10} {'일치?':>7}")
    summary = []
    for tm in TERMS:
        ins = [r["d_auc_vs_sem"] for r in rows if r["term"] == tm and r["level"] == "in-condition"]
        outs = [r["d_auc_vs_sem"] for r in rows if r["term"] == tm and r["level"] == "out-of-condition"]
        if not ins or not outs:
            continue
        mi, mo = float(np.mean(ins)), float(np.mean(outs))
        # "더 크다" = the term's incremental contribution is more favourable in-condition;
        # sign-aware: compare |dAUC| only when the term helps, else compare raw.
        agree = (mi > mo) if mi > 0 or mo > 0 else (abs(mi) < abs(mo))
        n_agree = sum(1 for a, b in zip(ins, outs) if (a > b))
        summary.append((tm, mi, mo, mi - mo, agree, n_agree))
        print(f"{tm:10} {mi:+13.5f} {mo:+11.5f} {mi-mo:+10.5f} {'in>out':>10} "
              f"{'예' if agree else '아니오':>7} ({n_agree}/5 시드)")
    print(f"\nWROTE {out/'stratified_auc.csv'}")


if __name__ == "__main__":
    main()
