# realistic-v1 — conditional validity of all five constraint terms, n=1

Data `data/realistic_v1` (~161 true identities/sector, 3.99M triples), decoder `num_slots=200`, blocking `dt_max_s=450 / r_err_floor_m=300`, epochs=40, `--full-sectors`, decode_full scoring. Seeds with all 6 variants finished: 19521006.

> **PARTIAL RUN — n=1 of the planned 5 seeds.** Directional reading only; the paired t-test is undefined at n=1 and is reported as n/a.

**n=1 → low statistical power. The paired t-test (df=0) is reported for completeness; interpretation leans on per-seed sign consistency (SC) and dev/test agreement, not p alone.**

Convention: Δ = variant − m3_full. **Negative ΔF1 ⇒ removing the term HURT ⇒ the term helps on this data.** Where dev and test disagree in sign, the result is marked NOT TRUSTWORTHY.

## 1. Full metric panel — dev (mean, n=1)

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.7055 | 0.1746 | 0.2792 | 0.1815 | 0.9774 | 0.2694 | 0.7306 | 0.2602 | 0.7849 |
| no_motion | 0.7424 | 0.2104 | 0.3270 | 0.1835 | 0.9855 | 0.2568 | 0.7432 | 0.2806 | 0.7308 |
| no_time | 0.7138 | 0.1615 | 0.2612 | 0.1973 | 0.9693 | 0.2601 | 0.7399 | 0.2513 | 0.7937 |
| no_state | 0.6514 | 0.1338 | 0.2202 | 0.2160 | 1.0262 | 0.2800 | 0.7200 | 0.2446 | 0.7900 |
| no_rel | 0.6805 | 0.1864 | 0.2912 | 0.2308 | 1.0014 | 0.2941 | 0.7059 | 0.2743 | 0.7606 |
| no_src | 0.7209 | 0.1976 | 0.3087 | 0.2134 | 0.9875 | 0.2615 | 0.7385 | 0.2664 | 0.7625 |

## 2. Full metric panel — test (mean, n=1)

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.6994 | 0.1751 | 0.2782 | 0.2038 | 0.9775 | 0.2761 | 0.7239 | 0.2541 | 0.7422 |
| no_motion | 0.7263 | 0.2099 | 0.3231 | 0.1780 | 0.9734 | 0.2473 | 0.7527 | 0.2805 | 0.7182 |
| no_time | 0.7312 | 0.1618 | 0.2629 | 0.1846 | 0.9558 | 0.2513 | 0.7487 | 0.2485 | 0.7852 |
| no_state | 0.6566 | 0.1295 | 0.2149 | 0.2185 | 1.0072 | 0.2825 | 0.7175 | 0.2374 | 0.7799 |
| no_rel | 0.7122 | 0.1920 | 0.2998 | 0.1984 | 0.9701 | 0.2708 | 0.7292 | 0.2639 | 0.7319 |
| no_src | 0.7072 | 0.1786 | 0.2831 | 0.1929 | 0.9828 | 0.2545 | 0.7455 | 0.2489 | 0.7238 |

## 3. (a) ΔF1 vs m3_full — direction, sign consistency, paired t-test

### dev

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=0) | p | reading |
|---|---|---|---|---|---|---|---|
| no_motion | +0.0478 | +0.0478 | removal helps/ties | n/a (n=1) | n/a | n/a | term does NOT help (removal ≥ full) |
| no_time | -0.0180 | -0.0180 | removal hurts | n/a (n=1) | n/a | n/a | term HELPS (removal hurts) |
| no_state | -0.0590 | -0.0590 | removal hurts | n/a (n=1) | n/a | n/a | term HELPS (removal hurts) |
| no_rel | +0.0120 | +0.0120 | removal helps/ties | n/a (n=1) | n/a | n/a | term does NOT help (removal ≥ full) |
| no_src | +0.0295 | +0.0295 | removal helps/ties | n/a (n=1) | n/a | n/a | term does NOT help (removal ≥ full) |

### test

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=0) | p | reading |
|---|---|---|---|---|---|---|---|
| no_motion | +0.0449 | +0.0449 | removal helps/ties | n/a (n=1) | n/a | n/a | term does NOT help (removal ≥ full) |
| no_time | -0.0153 | -0.0153 | removal hurts | n/a (n=1) | n/a | n/a | term HELPS (removal hurts) |
| no_state | -0.0633 | -0.0633 | removal hurts | n/a (n=1) | n/a | n/a | term HELPS (removal hurts) |
| no_rel | +0.0216 | +0.0216 | removal helps/ties | n/a (n=1) | n/a | n/a | term does NOT help (removal ≥ full) |
| no_src | +0.0049 | +0.0049 | removal helps/ties | n/a (n=1) | n/a | n/a | term does NOT help (removal ≥ full) |

## 4. (b) amb160 control — does any term become newly useful on realistic-v1?

amb160 was the clean-uncertainty benchmark where only kinematics reversed. If the pre-registered hypothesis holds, terms targeting the four newly injected uncertainties (time offsets/report lag, state dynamics, formation-relation fingerprints, per-source error heterogeneity) should turn negative here.

| variant | realistic-v1 dev ΔF1 | realistic-v1 test ΔF1 | amb160 test ΔF1 | amb160 SC | newly useful on realistic-v1? |
|---|---|---|---|---|---|
| no_motion | +0.0478 | +0.0449 | -0.0203 | YES | no |
| no_time | -0.0180 | -0.0153 | -0.0001 | no | YES — newly useful |
| no_state | -0.0590 | -0.0633 | -0.0111 | no | YES — newly useful |
| no_rel | +0.0120 | +0.0216 | -0.0014 | no | no |
| no_src | +0.0295 | +0.0049 | -0.0123 | no | no |

- Newly useful on realistic-v1: **no_time, no_state**.
- Not trustworthy (dev/test sign conflict): **none**.

## 5. (d) Does kinematics still hold on realistic-v1?

- amb160 (control): test ΔF1 -0.0203, SC YES, p 0.0133 → kinematics helped.
- realistic-v1: dev ΔF1 +0.0478 (SC n/a (n=1)), test ΔF1 +0.0449 (SC n/a (n=1), t n/a, p n/a).
- **Reading: kinematics does NOT help here (removal ≥ full) — the amb160 result does NOT reproduce**

## 6. (c) Does each term's targeted diagnostic degrade specifically?

| variant | targeted metric | expected on removal | dev Δ | test Δ | dev SC | test SC | matches? |
|---|---|---|---|---|---|---|---|
| no_motion | recall | down | +0.0359 | +0.0348 | n/a (n=1) | n/a (n=1) | NO |
| no_time | wrong_merge_rate | up | +0.0158 | -0.0192 | n/a (n=1) | n/a (n=1) | partial (one split) |
| no_time | impossible_transition_rate | up | -0.0093 | -0.0248 | n/a (n=1) | n/a (n=1) | NO |
| no_state | wrong_merge_rate | up | +0.0345 | +0.0147 | n/a (n=1) | n/a (n=1) | YES (both splits) |
| no_rel | recall | down | +0.0118 | +0.0170 | n/a (n=1) | n/a (n=1) | NO |
| no_rel | fragmentation_rate | up | +0.0239 | -0.0074 | n/a (n=1) | n/a (n=1) | partial (one split) |
| no_src | precision | ? | +0.0154 | +0.0078 | n/a (n=1) | n/a (n=1) | n/a (no directional prediction) |

- **per-source-pair precision is NOT present in `metrics.json`** (available keys: precision, recall, f1, wrong_merge_rate, fragmentation_rate, impossible_transition_rate, trajectory_consistency_rate, dangling_precision, dangling_recall). The `no_src` row therefore uses overall precision only; a source-pair-resolved precision check **could not be performed** with the stored metrics.

## 7. Caveats

- n=1 per variant → paired t-test df=0 has low power; a non-significant p does NOT establish absence of an effect, and a significant p at this n is fragile.
- 5 removal variants are compared against one baseline; **no multiple-comparison correction is applied** — with 5 tests at α=0.05 the family-wise false-positive risk is ≈23%.
- Selection was made on dev; test values are reported as the final outcome. Every trained variant is reported regardless of outcome; none was dropped.
- Absolute performance is low for every model (m3_full test F1 ≈ 0.278, recall ≈ 0.175, fragmentation ≈ 0.978). These are RELATIVE contrasts; the decoder routes most observations to the null slot, so term effects are measured in a low-recall regime.
- 40 epochs matches the amb160 control exactly (required for the §4 comparison), but dev F1 was still rising at epoch 40 on this larger suite — the models are compared at a fixed budget, not at convergence.