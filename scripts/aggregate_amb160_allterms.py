"""amb160 conditional-validity aggregation for ALL FIVE constraint terms.

Compares, on the hard data (data/hard_ambiguity/amb160, num_slots=160, epochs=40,
--full-sectors, decode_full scoring, seeds 19521006/07/08/09/10):

    m3_full (baseline)  vs  no_motion (existing)  and  no_time / no_state / no_rel / no_src (new)

m3_full & no_motion are read from the EXISTING main run directory (not re-trained);
the four new variants from runs/hard_ambiguity/amb160_allterms/.

Reports, for dev and test separately, mean+-std of the full metric panel, and for
each removal variant the DeltaF1 vs m3_full with per-seed sign consistency and a
paired t-test (df=4). n=5 is low power -- p is reported but never used alone.

Writes CSV + markdown + a bar chart into results/ablation_hard_allterms/.
"""
import json, csv, math, statistics
from pathlib import Path

RUN_MAIN = Path("runs/hard_ambiguity/amb160_slots160")      # m3_full, no_motion (existing)
RUN_NEW = Path("runs/hard_ambiguity/amb160_allterms")       # no_time/state/rel/src (new)
OUT = Path("results/ablation_hard_allterms")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "m3_full"
VARIANTS = ["no_motion", "no_time", "no_state", "no_rel", "no_src"]
MODELS = [BASE] + VARIANTS
RUNDIR = {m: (RUN_MAIN if m in (BASE, "no_motion") else RUN_NEW) for m in MODELS}
SEEDS = [19521006, 19521007, 19521008, 19521009, 19521010]
SPLITS = ["dev", "test"]
METRICS = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
           "impossible_transition_rate", "trajectory_consistency_rate",
           "dangling_precision", "dangling_recall"]
# term -> the diagnostic metric it is supposed to target, and the direction that
# would count as "removing it specifically hurts what it was designed to guard".
TARGETED = {
    "no_state": [("wrong_merge_rate", "up")],
    "no_rel":   [("recall", "down"), ("fragmentation_rate", "up")],
    "no_time":  [("wrong_merge_rate", "up"), ("impossible_transition_rate", "up")],
    "no_motion": [("recall", "down")],
    "no_src":   [("precision", "?")],   # per-source-pair precision is NOT in metrics.json
}

_cache = {}


def agg(model, seed, split):
    key = (model, seed)
    if key not in _cache:
        _cache[key] = json.load(open(RUNDIR[model] / f"{model}_seed{seed}" / "metrics.json"))
    return _cache[key][split]["aggregate"]


# ---------- stats (closed form; scipy not required) ----------
def _betacf(a, b, x):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d; h = d
    for m in range(1, 200):
        aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
        d = 1.0 + aa * d; d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c; c = tiny if abs(c) < tiny else c
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d = 1.0 + aa * d; d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c; c = tiny if abs(c) < tiny else c
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_t(diffs):
    n = len(diffs)
    md = statistics.mean(diffs)
    if all(abs(d - diffs[0]) < 1e-12 for d in diffs):
        return (md, 0.0, 1.0) if abs(md) < 1e-12 else (md, float("inf"), 0.0)
    se = statistics.stdev(diffs) / math.sqrt(n)
    t = md / se
    return md, t, max(0.0, min(1.0, _betai((n - 1) / 2.0, 0.5, (n - 1) / ((n - 1) + t * t))))


def contrast(variant, split, metric):
    """variant MINUS baseline, per seed. Negative DeltaF1 => removing the term hurt."""
    dd = [agg(variant, s, split)[metric] - agg(BASE, s, split)[metric] for s in SEEDS]
    md, t, p = paired_t(dd)
    signs = {("+" if d > 0 else "-" if d < 0 else "0") for d in dd}
    return {"per_seed": dd, "mean": md, "t": t, "p": p,
            "sc": len(signs) == 1, "sign": "+" if md > 0 else ("-" if md < 0 else "0")}


# ---------- collect ----------
summary = {}
rows = []
for m in MODELS:
    for sp in SPLITS:
        for k in METRICS:
            vals = [agg(m, s, sp)[k] for s in SEEDS]
            summary[(m, sp, k)] = (statistics.mean(vals), statistics.stdev(vals), vals)
    for s in SEEDS:
        for sp in SPLITS:
            a = agg(m, s, sp)
            rows.append({"model": m, "seed": s, "split": sp, **{k: a.get(k) for k in METRICS}})

with open(OUT / "metrics_raw.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["model", "seed", "split"] + METRICS)
    w.writeheader()
    w.writerows(rows)

with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "split", "metric", "mean", "std"] + [f"seed{s}" for s in SEEDS])
    for (m, sp, k), (mu, sd, vals) in summary.items():
        w.writerow([m, sp, k, f"{mu:.4f}", f"{sd:.4f}"] + [f"{v:.4f}" for v in vals])

with open(OUT / "deltas.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["variant", "split", "metric", "mean_delta_vs_m3_full", "sign",
                "sign_consistent", "t_df4", "p_two_tailed"] + [f"seed{s}" for s in SEEDS])
    for v in VARIANTS:
        for sp in SPLITS:
            for k in METRICS:
                c = contrast(v, sp, k)
                w.writerow([v, sp, k, f"{c['mean']:+.4f}", c["sign"],
                            "YES" if c["sc"] else "no",
                            "inf" if math.isinf(c["t"]) else f"{c['t']:.2f}",
                            f"{c['p']:.4f}"] + [f"{d:+.4f}" for d in c["per_seed"]])


def ms(m, sp, k):
    mu, sd, _ = summary[(m, sp, k)]
    return f"{mu:.4f}±{sd:.4f}"


# ---------- markdown ----------
md = []
md.append("# amb160 (hard) — conditional validity of ALL FIVE constraint terms, n=5\n")
md.append("Data `data/hard_ambiguity/amb160`, decoder `num_slots=160`, epochs=40, `--full-sectors`, "
          "decode_full scoring, seeds 19521006/07/08/09/10. `m3_full` and `no_motion` are the "
          "**existing** amb160 main-run results (re-used, not re-trained); `no_time`/`no_state`/"
          "`no_rel`/`no_src` are new runs with identical settings, differing only in which single "
          "term is dropped from `cta.enabled_terms`.\n")
md.append("**n=5 → low statistical power. The paired t-test (df=4) is reported for completeness; "
          "interpretation leans on per-seed sign consistency (SC) and dev/test agreement, not p alone.**\n")
md.append("Convention: Δ = variant − m3_full. **Negative ΔF1 ⇒ removing the term HURT ⇒ the term helps "
          "on hard data (reversal vs the easy-data finding).**\n")

md.append("## 1. Full metric panel — dev (mean±std, n=5)\n")
hdr = "| model | " + " | ".join(METRICS) + " |"
md.append(hdr); md.append("|" + "---|" * (len(METRICS) + 1))
for m in MODELS:
    md.append("| " + m + " | " + " | ".join(ms(m, "dev", k) for k in METRICS) + " |")
md.append("")
md.append("## 2. Full metric panel — test (mean±std, n=5)\n")
md.append(hdr); md.append("|" + "---|" * (len(METRICS) + 1))
for m in MODELS:
    md.append("| " + m + " | " + " | ".join(ms(m, "test", k) for k in METRICS) + " |")
md.append("")

md.append("## 3. (a)+(b) ΔF1 vs m3_full — direction, sign consistency, paired t-test\n")
for sp in SPLITS:
    md.append(f"### {sp}\n")
    md.append("| variant | per-seed ΔF1 (06/07/08/09/10) | mean ΔF1 | direction | SC | t(df=4) | p | reading |")
    md.append("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        c = contrast(v, sp, "f1")
        ps = ", ".join(f"{d:+.4f}" for d in c["per_seed"])
        tv = "inf" if math.isinf(c["t"]) else f"{c['t']:.2f}"
        reading = ("term HELPS (removal hurts) → reversal" if c["mean"] < 0
                   else "term does NOT help (removal ≥ full)" if c["mean"] > 0 else "tie")
        md.append(f"| {v} | {ps} | {c['mean']:+.4f} | {'removal hurts' if c['mean']<0 else 'removal helps/ties'} "
                  f"| {'YES' if c['sc'] else 'no'} | {tv} | {c['p']:.3f} | {reading} |")
    md.append("")

md.append("## 4. (c) Is motion special, or do other terms reverse too?\n")
md.append("| variant | dev ΔF1 | dev SC | test ΔF1 | test SC | reversal (both splits, ΔF1<0)? |")
md.append("|---|---|---|---|---|---|")
reversal = {}
for v in VARIANTS:
    cd, ct = contrast(v, "dev", "f1"), contrast(v, "test", "f1")
    rev = cd["mean"] < 0 and ct["mean"] < 0
    reversal[v] = {"dev": cd, "test": ct, "reversal": rev,
                   "consistent": rev and cd["sc"] and ct["sc"]}
    tag = ("YES — both splits" + (", sign-consistent on both" if cd["sc"] and ct["sc"] else ", SC partial")
           ) if rev else "no"
    md.append(f"| {v} | {cd['mean']:+.4f} | {'YES' if cd['sc'] else 'no'} | "
              f"{ct['mean']:+.4f} | {'YES' if ct['sc'] else 'no'} | {tag} |")
md.append("")
rev_terms = [v for v in VARIANTS if reversal[v]["reversal"]]
md.append(f"- Terms whose removal lowers F1 on BOTH dev and test (= term helps on hard data): "
          f"**{', '.join(rev_terms) if rev_terms else 'none'}**.")
md.append(f"- Motion-only? **{'YES — only no_motion reverses' if rev_terms == ['no_motion'] else 'NO'}** "
          f"({len(rev_terms)}/5 terms reverse).\n")

md.append("## 5. (d) Does each term's targeted diagnostic degrade specifically?\n")
md.append("| variant | targeted metric | expected on removal | dev Δ | test Δ | dev SC | test SC | matches expectation? |")
md.append("|---|---|---|---|---|---|---|---|")
for v in VARIANTS:
    for k, direction in TARGETED[v]:
        cd, ct = contrast(v, "dev", k), contrast(v, "test", k)
        if direction == "?":
            match = "n/a (no directional prediction)"
        else:
            want_pos = (direction == "up")
            ok = [(c["mean"] > 0) == want_pos for c in (cd, ct)]
            match = "YES (both splits)" if all(ok) else ("partial (one split)" if any(ok) else "NO")
        md.append(f"| {v} | {k} | {direction} | {cd['mean']:+.4f} | {ct['mean']:+.4f} | "
                  f"{'YES' if cd['sc'] else 'no'} | {'YES' if ct['sc'] else 'no'} | {match} |")
md.append("")
md.append("- **per-source-pair precision is NOT present in `metrics.json`** (available keys: "
          + ", ".join(METRICS) + "). The `no_src` row therefore uses overall precision only; "
          "a source-pair-resolved precision check could not be performed with the stored metrics.\n")

md.append("## 6. Caveats\n")
md.append("- n=5 per variant → paired t-test df=4 has low power; a non-significant p does NOT establish "
          "absence of an effect, and a significant p at n=5 is fragile.")
md.append("- 5 removal variants are compared against one baseline; **no multiple-comparison correction "
          "is applied** — with 5 tests at α=0.05 the family-wise false-positive risk is ≈23%.")
md.append("- Selection was made on dev; test values are reported as the final outcome. All four new "
          "variants that were trained are reported here regardless of outcome; no variant was dropped.")
md.append("- Absolute performance on amb160 is low for every model (test F1 ≈ "
          f"{summary[(BASE,'test','f1')][0]:.2f} for m3_full); these are RELATIVE contrasts on hard data.")

(OUT / "report.md").write_text("\n".join(md), encoding="utf-8")

# ---------- bar chart ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
labels = [m.replace("m3_full", "m3_full\n(all terms)") for m in MODELS]
colors = ["#2c7fb8"] + ["#d95f02"] + ["#7570b3"] * 4
for ax, sp in zip(axes, SPLITS):
    mus = [summary[(m, sp, "f1")][0] for m in MODELS]
    sds = [summary[(m, sp, "f1")][1] for m in MODELS]
    ax.bar(labels, mus, yerr=sds, capsize=4, color=colors, alpha=0.9)
    ax.axhline(mus[0], ls="--", lw=1.2, color="#2c7fb8", label="m3_full baseline")
    ax.set_title(f"{sp} F1 (amb160, n=5 mean±std)")
    for i, v in enumerate(mus):
        ax.text(i, v + sds[i] + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8)
axes[0].set_ylabel("F1")
fig.suptitle("amb160 (hard): F1 of each single-term removal vs the full model  "
             "(bar below the dashed line ⇒ that term helps under ambiguity)")
fig.tight_layout()
fig.savefig(OUT / "f1_by_term.png", dpi=140)

print("WROTE:", *[str(OUT / p) for p in
                  ["metrics_raw.csv", "summary.csv", "deltas.csv", "report.md", "f1_by_term.png"]], sep="\n  ")
print("\n----- REPORT -----\n")
# Write to a UTF-8 stdout buffer so a cp949 console cannot crash on em-dashes etc.
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print((OUT / "report.md").read_text(encoding="utf-8"))
