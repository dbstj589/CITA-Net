"""Study 2 v2 aggregation -- single pre-registered hypothesis on b_rel, with a placebo arm.

Pre-registration: docs/PREREG_realistic_v1_study2_v2.md (committed f69bc5e, before any
study-2 training run).

Primary hypothesis (the ONLY one): no_rel's test dF1 vs m3_full, paired by seed, is < 0.
Confirmed iff all three hold -- paired t (df=14) p < 0.05, sign consistency >= 12/15, and
dev/test direction agreement. No multiplicity correction is applied, because exactly one
hypothesis was registered; the placebo arm is a regime check / null estimate, not a
tested hypothesis, so it does not enter any correction.

The placebo arm (no_time) is a null manipulation: b_time is structurally 0 on this data
because blocking only emits forward-in-time candidates. Its role depends on the regime
that the smoke test selected:
  regime A (deterministic) -- verification gate: dF1 must be exactly 0.
  regime B (non-deterministic) -- empirical null: no_rel's deltas are additionally
    compared against the placebo's deltas, paired by seed, as a REFERENCE test that does
    not replace the primary decision above.

Both decoders are reported unconditionally: standard (primary) and revive margin 0.7
(secondary, value fixed, no re-tuning).

    python scripts/aggregate_study2_v2.py --regime A|B
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARMS = ["m3_full", "no_rel", "no_time"]
SEEDS = [19521011 + i for i in range(15)]
SPLITS = ["dev", "test"]
METRICS = ["f1", "precision", "recall", "wrong_merge_rate", "fragmentation_rate",
           "impossible_transition_rate", "dangling_precision", "dangling_recall"]


def _bcf(a, b, x):
    t = 1e-30; qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = t if abs(d) < t else d; d = 1 / d; h = d
    for m in range(1, 300):
        aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
        d = 1 + aa * d; d = t if abs(d) < t else d
        c = 1 + aa / c; c = t if abs(c) < t else c
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d = 1 + aa * d; d = t if abs(d) < t else d
        c = 1 + aa / c; c = t if abs(c) < t else c
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 3e-12:
            break
    return h


def _bi(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lb + a * math.log(x) + b * math.log(1 - x))
    return bt * _bcf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _bcf(b, a, 1 - x) / b


def paired_t(d):
    n = len(d); m = st.mean(d)
    if n < 2:
        return m, None, None
    if all(abs(x - d[0]) < 1e-15 for x in d):
        return (m, 0.0, 1.0) if abs(m) < 1e-15 else (m, float("inf"), 0.0)
    se = st.stdev(d) / math.sqrt(n); t = m / se
    return m, t, max(0.0, min(1.0, _bi((n - 1) / 2, 0.5, (n - 1) / ((n - 1) + t * t))))


def ci95(d):
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    m, s = st.mean(d), st.stdev(d) / math.sqrt(n)
    return (m - 2.145 * s, m + 2.145 * s)          # t_.975(df=14) = 2.145


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=["A", "B"])
    ap.add_argument("--runs-dir", default=str(REPO / "runs/realistic_v1_study2/runs"))
    ap.add_argument("--revive", default=str(REPO / "results/realistic_v1_study2/revive_results.jsonl"))
    ap.add_argument("--out-dir", default=str(REPO / "results/realistic_v1_study2"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs_dir)

    seeds = [s for s in SEEDS
             if all((runs / f"{a}_seed{s}" / "COMPLETED").exists() for a in ARMS)]
    if not seeds:
        raise SystemExit("no seed has all three arms completed (COMPLETED marker missing)")
    n = len(seeds)

    std = {}
    for a in ARMS:
        for s in seeds:
            d = json.loads((runs / f"{a}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
            for sp in SPLITS:
                std[(a, s, sp)] = d[sp]["aggregate"]
    data = {"standard": std}
    rp = Path(args.revive)
    if rp.exists():
        rev = {}
        for line in rp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if abs(r.get("margin", 0) - 0.7) < 1e-12:
                    rev[(r["variant"], r["seed"], r["split"])] = r
        if rev:
            data["revive"] = rev

    def delta(dec, arm, sp, k="f1"):
        return [data[dec][(arm, s, sp)][k] - data[dec][("m3_full", s, sp)][k] for s in seeds]

    md = [f"# 연구 2 v2 결과 — `b_rel` 단일 가설, 위약 대조 (체제 {args.regime}, n={n})\n",
          f"사전 등록: [`PREREG_realistic_v1_study2_v2.md`](../../docs/PREREG_realistic_v1_study2_v2.md) "
          f"(커밋 `f69bc5e`, 첫 학습 런 이전).\n",
          f"시드 {n}개: {', '.join(map(str, seeds))}. 팔 3개(m3_full / no_rel / no_time 위약). "
          f"40에폭, `configs/realistic_v1` 무수정.\n"]
    if n < 15:
        md.append(f"> **부분 결과 — 계획된 15시드 중 {n}개 완료.**\n")

    # ---- placebo ----
    md.append("## 1. 위약 팔 (`no_time`)\n")
    pl = {sp: delta("standard", "no_time", sp) for sp in SPLITS}
    if args.regime == "A":
        bad = [(sp, s, d) for sp in SPLITS for s, d in zip(seeds, pl[sp]) if d != 0.0]
        md.append("체제 A: 위약은 **검증 게이트**다. ΔF1이 정확히 0이어야 한다.\n")
        md.append(f"- 게이트 결과: **{'통과 (전 시드 dev·test 모두 Δ=0)' if not bad else 'VIOLATION'}**")
        for sp, s, d in bad[:10]:
            md.append(f"  - seed {s} {sp}: Δ={d:+.6f}")
    else:
        md.append("체제 B: 위약은 **경험적 영분포**다(아무것도 바꾸지 않는 조작의 ΔF1 분포).\n")
        md.append("| split | mean ΔF1 | std | 시드별 부호(음수) |")
        md.append("|---|---:|---:|:---:|")
        for sp in SPLITS:
            d = pl[sp]
            md.append(f"| {sp} | {st.mean(d):+.4f} | {st.stdev(d) if n > 1 else 0:.4f} | "
                      f"{sum(1 for x in d if x < 0)}/{n} |")
    md.append("")

    # ---- primary ----
    md.append("## 2. 1차 가설 — `no_rel` test ΔF1 < 0 (표준 디코드)\n")
    md.append(f"| split | mean ΔF1 | 95% CI | 부호일관 | t(df={n-1}) | p |")
    md.append("|---|---:|---|:---:|---:|---:|")
    verdict = {}
    for sp in SPLITS:
        d = delta("standard", "no_rel", sp)
        m, t, p = paired_t(d); lo, hi = ci95(d); neg = sum(1 for x in d if x < 0)
        verdict[sp] = {"mean": m, "p": p, "neg": neg, "d": d}
        md.append(f"| {sp} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | **{neg}/{n}** | "
                  f"{'n/a' if t is None else f'{t:.2f}'} | {'n/a' if p is None else f'{p:.4f}'} |")
    md.append("")
    md.append("시드별 ΔF1 (test): " + ", ".join(f"{x:+.4f}" for x in verdict["test"]["d"]) + "\n")

    c1 = verdict["test"]["p"] is not None and verdict["test"]["p"] < 0.05
    c2 = verdict["test"]["neg"] >= 12
    c3 = (verdict["test"]["mean"] < 0) == (verdict["dev"]["mean"] < 0)
    ok = c1 and c2 and c3
    md.append("**사전 등록 확증 조건 (셋 다 충족해야 함)**\n")
    md.append(f"1. paired t p < 0.05 → `{verdict['test']['p']:.4f}` — {'충족' if c1 else '미충족'}")
    md.append(f"2. 부호일관성 ≥ 12/15 → `{verdict['test']['neg']}/{n}` — {'충족' if c2 else '미충족'}")
    md.append(f"3. dev/test 방향 일치 → {'충족' if c3 else '미충족'}")
    md.append(f"\n### 판정: **{'지지 — rel 항의 학습 시 F1 기여를 40에폭·realistic-v1 조건부로 확립' if ok else ('역전 — 그대로 보고, 원인 탐색은 별도 승인' if verdict['test']['mean'] > 0 else '귀무 — 판별 신호(점수 수준)는 확립, 학습 시 F1 기여는 본 예산·검정력에서 미검출')}**\n")

    if args.regime == "B":
        dr = delta("standard", "no_rel", "test"); dp = pl["test"]
        dd = [a - b for a, b in zip(dr, dp)]
        m, t, p = paired_t(dd)
        md.append("**참고 검정(체제 B)** — `no_rel` Δ vs 위약 Δ, 동일 시드 짝지음: "
                  f"차의 차 평균 {m:+.4f}, p={p:.4f}. 1차 판정을 대체하지 않는다.\n")

    # ---- secondary + channels ----
    if "revive" in data:
        md.append("## 3. 2차 분석 — revive margin 0.7\n")
        md.append(f"| split | mean ΔF1 | 부호일관 | t | p |")
        md.append("|---|---:|:---:|---:|---:|")
        for sp in SPLITS:
            d = delta("revive", "no_rel", sp); m, t, p = paired_t(d)
            md.append(f"| {sp} | {m:+.4f} | {sum(1 for x in d if x<0)}/{n} | "
                      f"{'n/a' if t is None else f'{t:.2f}'} | {'n/a' if p is None else f'{p:.4f}'} |")
        md.append("")
    else:
        md.append("## 3. 2차 분석 — revive margin 0.7\n\n> 아직 미실행.\n")

    md.append("## 4. recall / precision 채널 (표준 디코드, test)\n")
    md.append("| arm | ΔF1 | ΔRecall | ΔPrecision |")
    md.append("|---|---:|---:|---:|")
    for a in ("no_rel", "no_time"):
        md.append(f"| {a} | {st.mean(delta('standard',a,'test')):+.4f} | "
                  f"{st.mean(delta('standard',a,'test','recall')):+.4f} | "
                  f"{st.mean(delta('standard',a,'test','precision')):+.4f} |")
    md.append("")

    md.append("## 5. 연구 1 대비 (기술 통계만 — 검정하지 않음)\n")
    md.append("연구 1은 실행 체제가 다를 수 있어 통계 결합(v1의 df=19 풀링)을 **삭제**했다. "
              "연구 1의 `no_rel` test ΔF1은 −0.0761 (n=5, 부호일관 0/5)이었다.\n")

    md.append("## 6. 한계\n")
    md.append(f"- **실행 체제 {args.regime}**" + (" (결정론 강제)" if args.regime == "A"
              else " (비결정론 — 연구 1과 동일 체제)") + ".")
    md.append("- 40에폭 예산이며 수렴점이 아니다(연구 1 탐색 런에서 80에폭이 +0.05).")
    md.append("- 단일 합성 데이터셋(realistic-v1). 실제 전장 일반화 성능이 아니다.")
    md.append("- 점수 수준 분석의 결론(판별 신호 확립)은 이 결과와 독립적으로 성립한다.")

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    with open(out / "deltas_per_seed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["decode", "arm", "split", "metric"] + [f"seed{s}" for s in seeds] + ["mean"])
        for dec in data:
            for a in ("no_rel", "no_time"):
                for sp in SPLITS:
                    for k in METRICS:
                        d = delta(dec, a, sp, k)
                        w.writerow([dec, a, sp, k] + [f"{x:+.6f}" for x in d] + [f"{st.mean(d):+.6f}"])
    print(f"WROTE {out/'report.md'}, {out/'deltas_per_seed.csv'}  (n={n} seeds)")
    print("\n".join(md[-14:]))


if __name__ == "__main__":
    main()
