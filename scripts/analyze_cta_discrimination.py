"""Analysis A -- per-term discriminative power of the CTA components. NO TRAINING.

Question: does each constraint term carry signal that separates gold-positive from
gold-negative cross-KG candidate pairs? This is the *content* evidence answering the
objection "removing a term lowers F1 only because score mass was removed" -- a term with
no discriminative power cannot be doing alignment work whatever removing it does to F1.

Two quantities per term, over the pairs dumped by scripts/dump_cta_scores.py:
  (i)  marginal AUC   -- the term's own contribution used alone as a score.
  (ii) incremental dAUC = AUC(w0*sem + b_term) - AUC(w0*sem). PRIMARY, because the
       question is contribution BEYOND semantics.
Reference rows: the full score, and w0*sem alone.

Uncertainty: pairs inside a sector share observations and are strongly dependent, so a
pair-level bootstrap would understate the CI by orders of magnitude. We resample SECTORS
with replacement (cluster bootstrap, B=1000) and take percentile CIs. The incremental
dAUC uses a PAIRED bootstrap -- both AUCs are recomputed on the SAME resampled sectors
and differenced there -- because the decision rule asks whether the dAUC CI excludes 0.

Cost: an exact pooled AUC needs a sort over ~8.5M rows, so 1000 resamples x 12 variants
x 10 cells is intractable directly. Each sector's positives/negatives are binned once
onto a common global quantile grid and a resample becomes a histogram sum, exact up to
the bin width. Point estimates are always computed exactly (rank-based, unbinned); the
histogram estimator is checked against them and the worst discrepancy is reported, so
the approximation is measured rather than assumed.

    python scripts/analyze_cta_discrimination.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
TERMS = ["b_time", "b_motion", "b_state", "b_rel", "b_src"]
NBINS = 1 << 14
B_BOOT = 1000


def exact_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC with average ranks for ties (Mann-Whitney U)."""
    n_pos = int(label.sum()); n_neg = label.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
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


def digitize(vals: np.ndarray):
    """(codes, n_bins) such that codes preserve the ordering of `vals`.

    Several CTA terms are low-cardinality (b_time has 1 distinct value, b_state 7,
    b_src 24, b_rel 30). Quantile bins collapse on those and lump genuinely different
    values together, which breaks the tie correction (0.5*neg assumes a bin holds only
    tied values) -- an earlier quantile-only version reached |err|=0.17 against the exact
    AUC. So: when the variable has at most NBINS distinct values, bin on the distinct
    values themselves, which makes the histogram estimator EXACT. Only genuinely
    high-cardinality variables (sem, score, b_motion: ~200k) fall back to quantile bins.
    """
    u = np.unique(vals)
    if u.size <= NBINS:
        return np.searchsorted(u, vals), u.size
    sub = vals[np.random.default_rng(0).choice(vals.size, 2_000_000, replace=False)]
    edges = np.unique(np.quantile(sub, np.linspace(0, 1, NBINS + 1)))
    nb = edges.size - 1
    return np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, nb - 1), nb


def auc_from_hist(pos: np.ndarray, neg: np.ndarray) -> float:
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    below = np.concatenate(([0.0], np.cumsum(neg)[:-1]))
    return float((pos * (below + 0.5 * neg)).sum() / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=str(REPO / "results/realistic_v1_score_analysis/dump"))
    ap.add_argument("--out-dir", default=str(REPO / "results/realistic_v1_score_analysis"))
    ap.add_argument("--boot", type=int, default=B_BOOT)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows, max_hist_err = [], 0.0
    for path in sorted(Path(args.dump_dir).glob("m3_full_seed*_*.parquet")):
        seed = int(path.stem.split("_seed")[1].split("_")[0])
        split = path.stem.rsplit("_", 1)[1]
        t = pq.read_table(path, columns=["sector", "label", "sem_w0", "score"] + TERMS)
        sector = np.asarray(t["sector"]); label = t["label"].to_numpy().astype(np.int8)
        sem = t["sem_w0"].to_numpy()
        variants = {"score_total": t["score"].to_numpy(), "sem_only": sem}
        for tm in TERMS:
            v = t[tm].to_numpy()
            variants[f"marginal_{tm}"] = v
            variants[f"incr_{tm}"] = sem + v
        del t

        sids = np.unique(sector)
        idx = {s: np.flatnonzero(sector == s) for s in sids}
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(sids), (args.boot, len(sids)))   # SHARED across variants

        exact, boot = {}, {}
        for name, val in variants.items():
            exact[name] = exact_auc(val, label)
            codes, nb = digitize(val)
            P = np.zeros((len(sids), nb)); N = np.zeros((len(sids), nb))
            for k, s in enumerate(sids):
                ii = idx[s]
                b = codes[ii]
                lab = label[ii]
                P[k] = np.bincount(b[lab == 1], minlength=nb)
                N[k] = np.bincount(b[lab == 0], minlength=nb)
            max_hist_err = max(max_hist_err, abs(auc_from_hist(P.sum(0), N.sum(0)) - exact[name]))
            boot[name] = np.array([auc_from_hist(P[p].sum(0), N[p].sum(0)) for p in picks])

        for name in variants:
            lo, hi = np.percentile(boot[name], [2.5, 97.5])
            r = {"seed": seed, "split": split, "variant": name, "auc": exact[name],
                 "ci_lo": float(lo), "ci_hi": float(hi)}
            if name.startswith("incr_"):
                d = boot[name] - boot["sem_only"]          # PAIRED on the same resample
                dlo, dhi = np.percentile(d, [2.5, 97.5])
                r["d_auc_vs_sem"] = exact[name] - exact["sem_only"]
                r["d_ci_lo"], r["d_ci_hi"] = float(dlo), float(dhi)
                r["d_excludes_0"] = bool(dlo > 0 or dhi < 0)
            elif name.startswith("marginal_"):
                r["excludes_0.5"] = bool(lo > 0.5 or hi < 0.5)
            rows.append(r)
        print(f"  [{seed} {split}] sem_only={exact['sem_only']:.4f}  total={exact['score_total']:.4f}  "
              + "  ".join(f"{t.replace('b_','')}Δ={exact[f'incr_{t}']-exact['sem_only']:+.4f}"
                          for t in TERMS))

    cols = ["seed", "split", "variant", "auc", "ci_lo", "ci_hi", "d_auc_vs_sem",
            "d_ci_lo", "d_ci_hi", "d_excludes_0", "excludes_0.5"]
    with open(out_dir / "auc_by_term.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nWROTE {out_dir/'auc_by_term.csv'}")
    print(f"histogram-vs-exact max |err| = {max_hist_err:.2e}  (bins={NBINS}, B={args.boot})")


if __name__ == "__main__":
    main()
