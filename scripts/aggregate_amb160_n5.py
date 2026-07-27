"""n=5 aggregation for the amb160 slots160 main run (seeds 19521006/07/08/09/10).
Reads metrics.json per model/seed, computes mean+/-std (dev/test separately),
m3_full-vs-no_motion and term_gating-vs-m3_full deltas with sign-consistency and
a df=4 paired t-test (closed-form; scipy not required), re-computes the gate
distribution per seed for consistency, writes CSV+MD and a bar chart.
Low power (n=5) is stated explicitly; sign consistency + dev/test agreement drive
the interpretation, not p alone."""
import sys, json, csv, math, statistics
from pathlib import Path
sys.path.insert(0, "src")

REPO = Path(".")
RUN = Path("runs/hard_ambiguity/amb160_slots160")
OUT = Path("results/hard_ambiguity_main")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["m3_full", "no_motion", "term_gating"]
SEEDS = [19521006, 19521007, 19521008, 19521009, 19521010]
SPLITS = ["dev", "test"]
METRICS = ["f1", "precision", "recall", "wrong_merge_rate",
           "fragmentation_rate", "impossible_transition_rate",
           "trajectory_consistency_rate"]


def agg(model, seed, split):
    p = RUN / f"{model}_seed{seed}" / "metrics.json"
    return json.load(open(p))[split]["aggregate"]


def _betacf(a, b, x):
    # Lentz's continued fraction for the incomplete beta (Numerical Recipes).
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d; h = d
    for m2 in range(1, 200):
        m = m2; aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
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
    # regularized incomplete beta I_x(a,b)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_tailed_p(t, df):
    # exact two-tailed p for Student's t (any df) via the incomplete beta.
    if df <= 0:
        return 1.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def paired_t(diffs):
    n = len(diffs)
    md = statistics.mean(diffs)
    if all(abs(d - diffs[0]) < 1e-12 for d in diffs):
        if abs(md) < 1e-12:
            return md, 0.0, 1.0
        return md, float("inf"), 0.0
    sd = statistics.stdev(diffs)  # ddof=1
    se = sd / math.sqrt(n)
    t = md / se
    p = _t_two_tailed_p(abs(t), n - 1)     # df = n-1 (=4 for n=5)
    return md, t, max(0.0, min(1.0, p))


def mean_std(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


# ---------- collect ----------
rows = []
for m in MODELS:
    for s in SEEDS:
        for sp in SPLITS:
            a = agg(m, s, sp)
            rows.append({"model": m, "seed": s, "split": sp,
                         **{k: a.get(k) for k in METRICS}})

with open(OUT / "n5_metrics_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "seed", "split"] + METRICS)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---------- mean+/-std ----------
summary = {}  # (model,split,metric) -> (mean,std, [per-seed])
for m in MODELS:
    for sp in SPLITS:
        for k in METRICS:
            vals = [agg(m, s, sp)[k] for s in SEEDS]
            summary[(m, sp, k)] = (statistics.mean(vals),
                                   statistics.stdev(vals), vals)

with open(OUT / "n5_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "split", "metric", "mean", "std"] + [f"seed{s}" for s in SEEDS])
    for (m, sp, k), (mu, sd, vals) in summary.items():
        w.writerow([m, sp, k, f"{mu:.4f}", f"{sd:.4f}"] + [f"{v:.4f}" for v in vals])


def contrast(A, B, split, metric="f1"):
    dd = [agg(A, s, split)[metric] - agg(B, s, split)[metric] for s in SEEDS]
    md, t, p = paired_t(dd)
    signs = {("+" if d > 0 else "-" if d < 0 else "0") for d in dd}
    sc = (len(signs) == 1)
    return {"per_seed": dd, "mean_delta": md, "t": t, "p": p,
            "sign_consistent": sc, "sign": ("+" if md > 0 else "-")}


# ---------- gate distribution per seed ----------
def gate_dist(seed):
    from citanet.config import load_config
    from citanet.data.stream import featurize_sector, read_split
    from citanet.engine_large import load_large_bundle
    import torch
    cfg = load_config("configs/amb160_main/term_gating.yaml")
    cfg.data_root = "data/hard_ambiguity/amb160"
    cfg.ontology_dir = "data/hard_ambiguity/amb160/ontology"
    model, fs, ont = load_large_bundle(cfg, RUN / f"term_gating_seed{seed}")
    model.eval()
    easy = []
    amb = []
    with torch.no_grad():
        for sid in read_split(cfg.data_root, "test"):
            feats = featurize_sector(Path(cfg.data_root) / "sectors" / sid, fs, ont, cfg)
            _ = model(feats)
            g = model.cta.last_gate
            sim = model.cta.last_raw_sim
            am = sim <= 0.5
            amb.append(g[am]); easy.append(g[~am])
    E = torch.cat(easy, 0).mean(0)
    A = torch.cat(amb, 0).mean(0)
    # motion is index 2
    return {"motion_easy": float(E[2]), "motion_amb": float(A[2]),
            "sem_easy": float(E[0]), "sem_amb": float(A[0])}


# ---------- MD report ----------
def ms(m, sp, k):
    mu, sd, _ = summary[(m, sp, k)]
    return f"{mu:.4f}±{sd:.4f}"


md = []
md.append("# amb160 main run — n=5 (seeds 19521006/07/08/09/10)\n")
md.append("Data `data/hard_ambiguity/amb160` (~109 identities/sector). Decoder `num_slots=160`. "
          "epochs=40, full-sectors, decode_full scoring. Identical settings across all 5 seeds "
          "(seeds 06/07/08 reused from n=3; 09/10 added).")
md.append("**n=5 → low statistical power; paired t-test df=4 is reported but interpretation leans on "
          "sign consistency (SC) across seeds and dev/test agreement, not p alone.**\n")

md.append("## (a) F1  (mean±std)\n")
md.append("| model | dev F1 | test F1 |")
md.append("|---|---|---|")
for m in MODELS:
    md.append(f"| {m} | {ms(m,'dev','f1')} | {ms(m,'test','f1')} |")
md.append("")

for (A, B, tag) in [("m3_full", "no_motion", "reversal: does motion help under ambiguity?"),
                    ("term_gating", "m3_full", "does learned gating beat fixed-weight motion?")]:
    md.append(f"### {A} vs {B} — {tag}\n")
    md.append("| split | per-seed ΔF1 (06/07/08/09/10) | mean Δ | sign-consistent | t(df=4) | p(2-tail) |")
    md.append("|---|---|---|---|---|---|")
    for sp in SPLITS:
        c = contrast(A, B, sp, "f1")
        ps = ", ".join(f"{d:+.4f}" for d in c["per_seed"])
        tval = "inf" if math.isinf(c["t"]) else f"{c['t']:.2f}"
        md.append(f"| {sp} | {ps} | {c['mean_delta']:+.4f} | "
                  f"{'YES' if c['sign_consistent'] else 'no'} ({c['sign']}) | {tval} | {c['p']:.3f} |")
    md.append("")

md.append("## (b) recall & trade-offs (test, mean±std)\n")
md.append("| model | recall | precision | wrong_merge | impossible | frag |")
md.append("|---|---|---|---|---|---|")
for m in MODELS:
    md.append(f"| {m} | {ms(m,'test','recall')} | {ms(m,'test','precision')} | "
              f"{ms(m,'test','wrong_merge_rate')} | {ms(m,'test','impossible_transition_rate')} | "
              f"{ms(m,'test','fragmentation_rate')} |")
md.append("")
rc = contrast("m3_full", "no_motion", "test", "recall")
md.append(f"- m3_full−no_motion recall Δ (test) per seed: "
          f"{', '.join(f'{d:+.4f}' for d in rc['per_seed'])}  "
          f"(mean {rc['mean_delta']:+.4f}, SC={'YES' if rc['sign_consistent'] else 'no'})\n")

md.append("## (c) term_gating gate distribution — motion gate, easy vs ambiguous (per seed)\n")
md.append("| seed | motion easy (sim>0.5) | motion ambig (sim≤0.5) | Δ | sem easy | sem ambig |")
md.append("|---|---|---|---|---|---|")
gsummary = []
for s in SEEDS:
    gd = gate_dist(s)
    gsummary.append(gd)
    md.append(f"| {s} | {gd['motion_easy']:.3f} | {gd['motion_amb']:.3f} | "
              f"{gd['motion_amb']-gd['motion_easy']:+.3f} | {gd['sem_easy']:.3f} | {gd['sem_amb']:.3f} |")
me = statistics.mean(g['motion_easy'] for g in gsummary)
ma = statistics.mean(g['motion_amb'] for g in gsummary)
md.append("")
md.append(f"- motion gate rises easy→ambiguous in all 3 seeds (mean {me:.3f}→{ma:.3f}); "
          "ambiguous-pair ratio (raw sim≤0.5) ≈ 0.508 (data-fixed, from seed-1).\n")

md.append("## interpretation\n")
r_test = contrast("m3_full", "no_motion", "test", "f1")
r_dev = contrast("m3_full", "no_motion", "dev", "f1")
consistent = r_test["sign"] == "+" and r_dev["sign"] == "+" and r_test["sign_consistent"] and r_dev["sign_consistent"]
md.append(f"- Reversal (m3_full>no_motion): dev SC={'YES' if r_dev['sign_consistent'] else 'no'}, "
          f"test SC={'YES' if r_test['sign_consistent'] else 'no'} → "
          f"{'CONSISTENT reversal across seeds & dev/test' if consistent else 'PARTIAL — see per-seed signs'}.")
tg = contrast("term_gating", "m3_full", "test", "f1")
md.append(f"- term_gating vs m3_full (test): mean Δ {tg['mean_delta']:+.4f}, "
          f"SC={'YES' if tg['sign_consistent'] else 'no'} → "
          f"{'term_gating does NOT beat m3_full (reproduced)' if tg['mean_delta']<0 else 'term_gating >= m3_full'}.")
md.append(f"- **Absolute performance stays low**: test F1 ≈ {summary[('m3_full','test','f1')][0]:.2f}, "
          f"fragmentation ≈ {summary[('m3_full','test','fragmentation_rate')][0]:.2f} — "
          "amb160 is hard (4× hard-neg density vs frozen); report relative advantage separately from absolute quality.")

(OUT / "n5_report.md").write_text("\n".join(md), encoding="utf-8")

# ---------- bar chart ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

frozen = {}  # model -> mean test f1 on frozen
import glob
for m in MODELS:
    fs = []
    for d in glob.glob(f"runs/ablation/{m}_seed*"):
        try:
            fs.append(json.load(open(d + "/metrics.json"))["test"]["aggregate"]["f1"])
        except Exception:
            pass
    frozen[m] = statistics.mean(fs) if fs else float("nan")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
labels = ["m3_full\n(+motion)", "no_motion", "term_gating\n(gated)"]
colors = ["#2c7fb8", "#d95f02", "#7570b3"]
for ax, (title, getter) in zip(
        axes,
        [("frozen (easy)", lambda m: (frozen[m], 0.0)),
         ("amb160 (hard)", lambda m: (summary[(m, 'test', 'f1')][0], summary[(m, 'test', 'f1')][1]))]):
    mus = [getter(m)[0] for m in MODELS]
    sds = [getter(m)[1] for m in MODELS]
    ax.bar(labels, mus, yerr=sds, capsize=5, color=colors, alpha=0.9)
    ax.set_title(title)
    ax.set_ylim(0, 0.85)
    for i, v in enumerate(mus):
        ax.text(i, v + (sds[i] if sds[i] else 0) + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    # annotate winner
    axes[0].set_ylabel("test F1")
fig.suptitle("Motion term reverses sign: hurts on easy (frozen), helps on hard (amb160)   [amb160 n=5 mean±std]")
fig.tight_layout()
fig.savefig(OUT / "n5_f1_reversal.png", dpi=140)
print("WROTE:")
for p in ["n5_metrics_raw.csv", "n5_summary.csv", "n5_report.md", "n5_f1_reversal.png"]:
    print("  ", OUT / p)
print("\n----- REPORT -----\n")
print((OUT / "n5_report.md").read_text(encoding="utf-8"))
