"""realistic-v1 conditional-validity aggregation for all five constraint terms.

Same statistical machinery as scripts/aggregate_amb160_allterms.py (verified), reading
runs/realistic_v1/ablation/ and contrasting every result against the amb160 control
(results/ablation_hard_allterms/deltas.csv) -- the pre-registered comparison: on amb160
only kinematics reversed cleanly (test dF1 -0.020, 5/5 SC, p=0.013); time/state/rel were
non-contributing and src was weak.

Pre-registered hypothesis: each constraint term is useful when the uncertainty it targets
is actually present. Whatever comes out is reported as-is.

TWO ANALYSES, both reported regardless of outcome (fixed before the 30 runs finished):
  * PRIMARY   = standard decode, straight from each run's metrics.json. This is the
                pre-registered main analysis.
  * SECONDARY = revive margin 0.7 (experiment #7's value, NOT re-tuned), read from
                revive_results_n5.jsonl if scripts/revive_realistic_v1.py has run.

Denominators: every ratio metric is reported next to its denominator, because variants
predict different numbers of links (pilot dev: n_pred_matches 22.9..31.1, a 36% spread)
and a bare ratio would hide that selection effect. wrong_merge_rate's denominator --
len(result.identities), eval.py:136 -- is NOT stored in metrics.json and cannot be
reconstructed from it; it is recovered from the revive pass's margin-0 re-decode, which
is mathematically identical to the standard decode, so the value is exact. Without that
pass the column reads "n/a (revive 미실행)".

Seeds are auto-detected, so this also works mid-sweep. With n<2 the paired t-test is
undefined and reported as n/a rather than invented; sign consistency is likewise vacuous
at n=1 and reported as such.

    python scripts/aggregate_realistic_v1.py

Writes CSV + markdown + an F1 bar chart into results/realistic_v1_ablation/, suffixed
_n{N} so earlier versions are never overwritten.
"""
import csv
import json
import math
import re
import statistics
from pathlib import Path

RUN = Path("runs/realistic_v1/ablation")
AMB160_DELTAS = Path("results/ablation_hard_allterms/deltas.csv")
OUT = Path("results/realistic_v1_ablation")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "m3_full"
VARIANTS = ["no_motion", "no_time", "no_state", "no_rel", "no_src"]
MODELS = [BASE] + VARIANTS
SPLITS = ["dev", "test"]
METRICS = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
           "impossible_transition_rate", "trajectory_consistency_rate",
           "dangling_precision", "dangling_recall"]
# ratio metric -> the count key holding its denominator ("" = not stored anywhere)
DENOM = {
    "precision": "n_pred_matches",
    "recall": "n_gold_matches",
    "f1": "",                                  # harmonic mean of two ratios
    "wrong_merge_rate": "n_pred_identities",   # recovered from the revive margin-0 pass
    "fragmentation_rate": "n_gold_matches",
    "impossible_transition_rate": "n_transitions",
    "trajectory_consistency_rate": "n_transitions",
    "dangling_precision": "n_pred_dangling",
    "dangling_recall": "n_gold_dangling",
}
# term -> the diagnostic metric it targets, and the direction on REMOVAL that would count
# as "removing it specifically hurts what it was designed to guard".
TARGETED = {
    "no_time":   [("wrong_merge_rate", "up"), ("impossible_transition_rate", "up")],
    "no_state":  [("wrong_merge_rate", "up")],
    "no_rel":    [("recall", "down"), ("fragmentation_rate", "up")],
    "no_motion": [("recall", "down")],
    "no_src":    [("precision", "?")],   # per-source-pair precision is NOT in metrics.json
}
REVIVE_MARGIN = 0.70


# ---------- discover what has finished ----------
def discover_seeds():
    seeds = {}
    for p in RUN.glob("*_seed*/metrics.json"):
        m = re.fullmatch(r"(.+)_seed(\d+)", p.parent.name)
        if m and m.group(1) in MODELS:
            seeds.setdefault(int(m.group(2)), set()).add(m.group(1))
    return sorted(s for s, models in seeds.items() if set(MODELS) <= models), seeds


SEEDS, FOUND = discover_seeds()
N = len(SEEDS)
if not SEEDS:
    raise SystemExit(f"No seed has all {len(MODELS)} variants finished yet. "
                     f"Found: { {s: sorted(m) for s, m in sorted(FOUND.items())} }")
SUF = f"_n{N}"


# ---------- load the two analyses ----------
def load_standard():
    """(model, seed, split) -> aggregate dict, from each run's metrics.json."""
    out = {}
    for m in MODELS:
        for s in SEEDS:
            d = json.loads((RUN / f"{m}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
            for sp in SPLITS:
                out[(m, s, sp)] = dict(d[sp]["aggregate"])
    return out


def load_revive(path: Path, margin: float):
    """(model, seed, split) -> aggregate dict for one margin, or {} if not run yet."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if abs(r["margin"] - margin) < 1e-12:
            out[(r["variant"], r["seed"], r["split"])] = r
    return out


STD = load_standard()
REV_PATH = OUT / f"revive_results{SUF}.jsonl"
REV0 = load_revive(REV_PATH, 0.0)              # == standard decode: source of the denominators
REV7 = load_revive(REV_PATH, REVIVE_MARGIN)    # the secondary analysis

# graft the recovered wrong_merge denominator onto the primary analysis
for k, v in REV0.items():
    if k in STD:
        for extra in ("n_pred_identities", "n_wrong_merges"):
            if extra in v:
                STD[k][extra] = v[extra]

ANALYSES = [("primary", "표준 디코드 (사전 등록 주 분석)", STD)]
if REV7:
    ANALYSES.append(("revive", f"revive margin {REVIVE_MARGIN} (2차 분석)", REV7))
# ASCII chart labels: the bundled matplotlib font has no Hangul glyphs, and a figure
# full of tofu boxes is worse than an English title.
CHART_LABEL = {"primary": "primary: standard decode",
               "revive": f"secondary: revive margin {REVIVE_MARGIN}"}


# ---------- stats (closed form; scipy not required) ----------
def _betacf(a, b, x):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
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
    """(mean, t, p). t/p are None when n<2 -- undefined, never fabricated."""
    n = len(diffs)
    md = statistics.mean(diffs)
    if n < 2:
        return md, None, None
    if all(abs(d - diffs[0]) < 1e-12 for d in diffs):
        return (md, 0.0, 1.0) if abs(md) < 1e-12 else (md, float("inf"), 0.0)
    se = statistics.stdev(diffs) / math.sqrt(n)
    t = md / se
    return md, t, max(0.0, min(1.0, _betai((n - 1) / 2.0, 0.5, (n - 1) / ((n - 1) + t * t))))


def holm(pvals: dict):
    """Holm-Bonferroni step-down adjusted p-values. Reference column only -- the primary
    verdict stays uncorrected + sign consistency, as pre-registered."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    if not items:
        return {k: None for k in pvals}
    items.sort(key=lambda kv: kv[1])
    k = len(items)
    adj, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (k - i) * p))   # enforce monotonicity
        adj[key] = running
    for key in pvals:
        adj.setdefault(key, None)
    return adj


def contrast(data, variant, split, metric):
    """variant MINUS baseline, per seed. Negative dF1 => removing the term hurt."""
    dd = [data[(variant, s, split)][metric] - data[(BASE, s, split)][metric] for s in SEEDS]
    md, t, p = paired_t(dd)
    signs = {("+" if d > 0 else "-" if d < 0 else "0") for d in dd}
    return {"per_seed": dd, "mean": md, "t": t, "p": p,
            "sc": len(signs) == 1, "sign": "+" if md > 0 else ("-" if md < 0 else "0")}


def fmt_t(c):
    return "n/a" if c["t"] is None else ("inf" if math.isinf(c["t"]) else f"{c['t']:.2f}")


def fmt_p(c):
    return "n/a" if c["p"] is None else f"{c['p']:.4f}"


def fmt_sc(c):
    """Sign consistency is vacuous with a single seed -- never print a bare YES for n=1."""
    return "n/a (n=1)" if N < 2 else (f"YES ({N}/{N})" if c["sc"] else "no")


def mstd(data, m, sp, k):
    vals = [data[(m, s, sp)].get(k) for s in SEEDS]
    if any(v is None for v in vals):
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def cell(data, m, sp, k):
    mu, sd = mstd(data, m, sp, k)
    if mu is None:
        return "n/a"
    return f"{mu:.4f}±{sd:.4f}" if N > 1 else f"{mu:.4f}"


def denom_cell(data, m, sp, k):
    dk = DENOM.get(k, "")
    if not dk:
        return "—"
    mu, _ = mstd(data, m, sp, dk)
    return "n/a (revive 미실행)" if mu is None else f"{mu:.1f}"


# ---------- amb160 control ----------
def load_amb160():
    out = {}
    if not AMB160_DELTAS.exists():
        return out
    with open(AMB160_DELTAS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["variant"], r["split"], r["metric"])] = (
                float(r["mean_delta_vs_m3_full"]), r["sign_consistent"] == "YES",
                float(r["p_two_tailed"]))
    return out


AMB = load_amb160()

# ---------- CSV output ----------
rows = []
for tag, _label, data in ANALYSES:
    for m in MODELS:
        for s in SEEDS:
            for sp in SPLITS:
                a = data[(m, s, sp)]
                rows.append({"analysis": tag, "model": m, "seed": s, "split": sp,
                             **{k: a.get(k) for k in METRICS},
                             **{v: a.get(v) for v in sorted(set(DENOM.values()) - {""})}})
with open(OUT / f"metrics_raw{SUF}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

with open(OUT / f"deltas{SUF}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["analysis", "variant", "split", "metric", "mean_delta_vs_m3_full", "sign",
                "sign_consistent", f"t_df{max(N-1,0)}", "p_two_tailed", "p_holm",
                "denominator_key", "denominator_mean_variant", "denominator_mean_m3_full",
                "amb160_mean_delta", "amb160_sign_consistent"] + [f"seed{s}" for s in SEEDS])
    for tag, _label, data in ANALYSES:
        for sp in SPLITS:
            for k in METRICS:
                cs = {v: contrast(data, v, sp, k) for v in VARIANTS}
                hp = holm({v: c["p"] for v, c in cs.items()})
                for v in VARIANTS:
                    c = cs[v]
                    a = AMB.get((v, sp, k))
                    dk = DENOM.get(k, "")
                    dv, _ = mstd(data, v, sp, dk) if dk else (None, None)
                    db, _ = mstd(data, BASE, sp, dk) if dk else (None, None)
                    w.writerow([tag, v, sp, k, f"{c['mean']:+.4f}", c["sign"],
                                "YES" if c["sc"] else "no", fmt_t(c), fmt_p(c),
                                "n/a" if hp[v] is None else f"{hp[v]:.4f}",
                                dk or "—",
                                "n/a" if dv is None else f"{dv:.1f}",
                                "n/a" if db is None else f"{db:.1f}",
                                f"{a[0]:+.4f}" if a else "n/a",
                                ("YES" if a[1] else "no") if a else "n/a"]
                               + [f"{d:+.4f}" for d in c["per_seed"]])

# ---------- markdown ----------
md = []
md.append(f"# realistic-v1 — conditional validity of all five constraint terms, n={N}\n")
md.append(f"Data `data/realistic_v1` (161 true identities/sector, 3.99M triples), decoder "
          f"`num_slots=200`, blocking `dt_max_s=450 / r_err_floor_m=300`, epochs=40, "
          f"`--full-sectors`, decode_full scoring. Seeds: {', '.join(map(str, SEEDS))}.\n")
if N < 5:
    md.append(f"> **PARTIAL — n={N} of the planned 5 seeds.** Directional reading only.\n")
md.append(f"**n={N} → low statistical power. The paired t-test (df={max(N-1,0)}) is reported for "
          "completeness; interpretation leans on per-seed sign consistency (SC) and dev/test "
          "agreement, not p alone. Holm-adjusted p is a reference column; the pre-registered "
          "verdict uses the uncorrected p together with SC.**\n")
md.append("Convention: Δ = variant − m3_full. **Negative ΔF1 ⇒ removing the term HURT ⇒ the term "
          "helps.** Where dev and test disagree in sign, the result is marked NOT TRUSTWORTHY.\n")
md.append(f"Analyses reported: **{', '.join(l for _, l, _ in ANALYSES)}**"
          + ("" if REV7 else "  \n> 2차(revive) 분석은 `scripts/revive_realistic_v1.py` 미실행으로 "
                            "아직 없음. `wrong_merge_rate`의 분모도 그 패스에서 복구된다.") + "\n")

sec = 0
for tag, label, data in ANALYSES:
    sec += 1
    md.append(f"## {sec}. [{label}] 전체 지표 패널 (mean{'±std' if N>1 else ''}, 분모 병기)\n")
    for sp in SPLITS:
        md.append(f"### {sp}\n")
        md.append("| model | " + " | ".join(METRICS) + " |")
        md.append("|" + "---|" * (len(METRICS) + 1))
        for m in MODELS:
            md.append("| " + m + " | " + " | ".join(cell(data, m, sp, k) for k in METRICS) + " |")
        md.append("| *분모(평균)* | " + " | ".join(denom_cell(data, BASE, sp, k) for k in METRICS)
                  + " |   ← m3_full 기준")
        md.append("")

sec += 1
md.append(f"## {sec}. (a) ΔF1 vs m3_full — 방향·부호일관성·paired t·Holm\n")
for tag, label, data in ANALYSES:
    md.append(f"### [{label}]\n")
    for sp in SPLITS:
        cs = {v: contrast(data, v, sp, "f1") for v in VARIANTS}
        hp = holm({v: c["p"] for v, c in cs.items()})
        md.append(f"**{sp}**\n")
        md.append(f"| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df={max(N-1,0)}) "
                  "| p | p(Holm) | reading |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            c = cs[v]
            reading = ("term HELPS (removal hurts)" if c["mean"] < 0
                       else "term does NOT help" if c["mean"] > 0 else "tie")
            md.append(f"| {v} | {', '.join(f'{d:+.4f}' for d in c['per_seed'])} | {c['mean']:+.4f} "
                      f"| {'removal hurts' if c['mean']<0 else 'removal helps/ties'} | {fmt_sc(c)} "
                      f"| {fmt_t(c)} | {fmt_p(c)} | "
                      f"{'n/a' if hp[v] is None else f'{hp[v]:.4f}'} | {reading} |")
        md.append("")

sec += 1
md.append(f"## {sec}. (b) amb160 대조 — realistic-v1에서 새로 유효해진 항이 있는가\n")
md.append("amb160은 불확실성이 깨끗한 벤치마크였고 운동학만 역전했다. 사전 등록 가설이 맞다면 "
          "새로 주입한 4종 불확실성(시각 오프셋·보고지연, 상태 동역학, 편제 관계 지문, 출처 오차 "
          "이질성)을 겨냥한 항들이 여기서 음수로 돌아서야 한다.\n")
verdict = {}
for tag, label, data in ANALYSES:
    md.append(f"### [{label}]\n")
    md.append("| variant | realistic-v1 dev ΔF1 | realistic-v1 test ΔF1 | amb160 test ΔF1 "
              "| amb160 SC | 판정 |")
    md.append("|---|---|---|---|---|---|")
    vd = {}
    for v in VARIANTS:
        cd, ct = contrast(data, v, "dev", "f1"), contrast(data, v, "test", "f1")
        a = AMB.get((v, "test", "f1"))
        helps = cd["mean"] < 0 and ct["mean"] < 0
        conflict = (cd["mean"] < 0) != (ct["mean"] < 0)
        helped_amb = a is not None and a[0] < 0 and a[1]
        t_ = ("NOT TRUSTWORTHY (dev/test 부호 불일치)" if conflict else
              "YES — 새로 유효" if helps and not helped_amb else
              "amb160에서도 이미 유효" if helps else "no")
        vd[v] = {"dev": cd, "test": ct, "helps": helps, "conflict": conflict, "tag": t_}
        md.append(f"| {v} | {cd['mean']:+.4f} | {ct['mean']:+.4f} | "
                  f"{f'{a[0]:+.4f}' if a else 'n/a'} | "
                  f"{('YES' if a[1] else 'no') if a else 'n/a'} | {t_} |")
    verdict[tag] = vd
    newly = [v for v in VARIANTS if vd[v]["tag"] == "YES — 새로 유효"]
    md.append("")
    md.append(f"- 새로 유효: **{', '.join(newly) if newly else 'none'}**")
    md.append(f"- 신뢰 불가(dev/test 충돌): "
              f"**{', '.join(v for v in VARIANTS if vd[v]['conflict']) or 'none'}**\n")

sec += 1
md.append(f"## {sec}. (d) 운동학이 realistic-v1에서도 유효한가\n")
amb_m = AMB.get(("no_motion", "test", "f1"))
md.append(f"- amb160(대조): test ΔF1 {f'{amb_m[0]:+.4f}' if amb_m else 'n/a'}, "
          f"SC {('YES' if amb_m[1] else 'no') if amb_m else 'n/a'}, "
          f"p {f'{amb_m[2]:.4f}' if amb_m else 'n/a'} → 운동학이 도움이 됐다.")
for tag, label, data in ANALYSES:
    cd, ct = contrast(data, "no_motion", "dev", "f1"), contrast(data, "no_motion", "test", "f1")
    md.append(f"- [{label}] dev ΔF1 {cd['mean']:+.4f} (SC {fmt_sc(cd)}), "
              f"test ΔF1 {ct['mean']:+.4f} (SC {fmt_sc(ct)}, t {fmt_t(ct)}, p {fmt_p(ct)}) "
              f"→ **{verdict[tag]['no_motion']['tag']}**")
md.append("")

sec += 1
md.append(f"## {sec}. recall·precision 채널 — ΔF1이 어느 경로에서 나오는가\n")
md.append("특히 `no_motion`의 ΔF1이 재현율 경로에서 나오는지 확인한다.\n")
for tag, label, data in ANALYSES:
    md.append(f"### [{label}]\n")
    md.append("| variant | split | ΔF1 | ΔRecall | ΔPrecision | 주 경로 |")
    md.append("|---|---|---|---|---|---|")
    for v in VARIANTS:
        for sp in SPLITS:
            cf = contrast(data, v, sp, "f1")
            cr = contrast(data, v, sp, "recall")
            cp = contrast(data, v, sp, "precision")
            path = ("recall" if abs(cr["mean"]) > 2 * abs(cp["mean"]) else
                    "precision" if abs(cp["mean"]) > 2 * abs(cr["mean"]) else "both")
            md.append(f"| {v} | {sp} | {cf['mean']:+.4f} | {cr['mean']:+.4f} "
                      f"| {cp['mean']:+.4f} | {path} |")
    md.append("")

sec += 1
md.append(f"## {sec}. (c) 각 항의 겨냥 지표가 특이적으로 나빠지는가 (분모 병기)\n")
for tag, label, data in ANALYSES:
    md.append(f"### [{label}]\n")
    md.append("| variant | 겨냥 지표 | 제거 시 기대 | dev Δ | test Δ | dev SC | test SC "
              "| 분모(변형/기준, test) | 일치? |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        for k, direction in TARGETED[v]:
            cd, ct = contrast(data, v, "dev", k), contrast(data, v, "test", k)
            if direction == "?":
                match = "n/a (방향 예측 없음)"
            else:
                want_pos = (direction == "up")
                ok = [(c["mean"] > 0) == want_pos for c in (cd, ct)]
                match = "YES (양쪽)" if all(ok) else ("partial (한쪽)" if any(ok) else "NO")
            md.append(f"| {v} | {k} | {direction} | {cd['mean']:+.4f} | {ct['mean']:+.4f} "
                      f"| {fmt_sc(cd)} | {fmt_sc(ct)} "
                      f"| {denom_cell(data, v, 'test', k)} / {denom_cell(data, BASE, 'test', k)} "
                      f"| {match} |")
    md.append("")
md.append("- **출처쌍별(per-source-pair) 정밀도는 `metrics.json`에 없다** (보유 키: "
          + ", ".join(METRICS) + "). `no_src` 행은 전체 정밀도로만 대리하며, 출처쌍 분해 정밀도 "
          "점검은 **수행 불가**다.\n")

sec += 1
md.append(f"## {sec}. 한계\n")
md.append(f"- n={N} → paired t(df={max(N-1,0)})는 저검정력. 비유의 p가 효과 부재를 뜻하지 않고, "
          "이 n에서의 유의 p도 취약하다.")
md.append("- 5개 변형을 하나의 기준선과 비교하며 **다중비교 미보정**(α=0.05에서 FWER ≈23%). "
          "Holm 보정 p를 참고 열로 병기했으나 주 판정은 사전 등록대로 미보정 + 부호일관성.")
md.append("- dev로 선택하고 test를 보고한다. 학습된 모든 변형을 결과와 무관하게 보고했고 "
          "제외한 변형은 없다.")
md.append(f"- 절대 성능이 모든 모델에서 낮다 (m3_full test F1 ≈ "
          f"{mstd(STD, BASE, 'test', 'f1')[0]:.3f}, recall ≈ {mstd(STD, BASE, 'test', 'recall')[0]:.3f}). "
          "디코더가 대부분의 관측을 ∅ 슬롯으로 보내므로 항 효과가 **저재현율 영역**에서 측정된다. "
          "2차(revive) 분석이 이 영역 의존성을 점검한다.")
md.append("- **`fragmentation_rate`는 비율이 아니다.** 분자가 `fragmented + unrecovered`라 "
          "1.0을 넘을 수 있다(실측 최대 1.13). 0~1로 해석하면 안 된다.")
md.append("- `aggregate`는 섹터별 **비율의 매크로 평균**(카운트 합산 후 재계산이 아님)이라 "
          "분모가 작은 섹터도 동일 가중을 받는다. 그래서 분모 평균을 함께 싣는다.")
md.append("- `wrong_merge_rate`의 분모는 `metrics.json`에 저장되지 않는다(eval.py:136의 "
          "`len(result.identities)`). revive 패스의 margin-0 재디코드에서 복구했으며, margin-0은 "
          "표준 디코드와 수학적으로 동일하므로 근사가 아니라 정확값이다"
          + ("." if REV0 else " — **아직 미실행이라 이 표에서는 n/a**."))
md.append("- 40에폭은 amb160 대조와 맞춘 고정 예산이며 수렴점이 아니다(dev F1이 40에폭에서도 "
          "상승 중). 변형들은 수렴이 아니라 **동일 예산**에서 비교된다. 80에폭 탐색 런이 "
          "이 예산 의존성을 별도로 점검한다(본 집계 제외).")
md.append("- 모든 수치는 통제된 합성 suite에 대한 것이며 실제 전장 데이터 일반화 성능이 아니다.")

(OUT / f"report{SUF}.md").write_text("\n".join(md), encoding="utf-8")

# ---------- bar chart ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nrow = len(ANALYSES)
fig, axes = plt.subplots(nrow, 2, figsize=(13, 5 * nrow), squeeze=False)
labels = [m.replace("m3_full", "m3_full\n(all terms)") for m in MODELS]
colors = ["#2c7fb8", "#d95f02"] + ["#7570b3"] * 4
for r, (tag, label, data) in enumerate(ANALYSES):
    for c, sp in enumerate(SPLITS):
        ax = axes[r][c]
        mus = [mstd(data, m, sp, "f1")[0] for m in MODELS]
        sds = [mstd(data, m, sp, "f1")[1] for m in MODELS]
        ax.bar(labels, mus, yerr=sds if N > 1 else None, capsize=4, color=colors, alpha=0.9)
        ax.axhline(mus[0], ls="--", lw=1.2, color="#2c7fb8", label="m3_full baseline")
        ax.set_title(f"{CHART_LABEL.get(tag, tag)} - {sp} F1 "
                     f"(n={N}{', mean+/-std' if N > 1 else ''})", fontsize=10)
        for i, v in enumerate(mus):
            ax.text(i, v + (sds[i] if N > 1 else 0) + 0.004, f"{v:.3f}", ha="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=8)
    axes[r][0].set_ylabel("F1")
fig.suptitle("realistic-v1: F1 of each single-term removal vs the full model  "
             "(bar BELOW the dashed line ⇒ that term helps)")
fig.tight_layout()
fig.savefig(OUT / f"f1_by_term{SUF}.png", dpi=140)

print("WROTE:", *[str(OUT / f"{p}{SUF}.{e}") for p, e in
                  [("metrics_raw", "csv"), ("deltas", "csv"),
                   ("report", "md"), ("f1_by_term", "png")]], sep="\n  ")
print(f"\nanalyses: {[t for t, _, _ in ANALYSES]}   seeds: {SEEDS}")
print("\n----- REPORT -----\n")
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print((OUT / f"report{SUF}.md").read_text(encoding="utf-8"))
