# amb160 (hard) — conditional validity of ALL FIVE constraint terms, n=5

Data `data/hard_ambiguity/amb160`, decoder `num_slots=160`, epochs=40, `--full-sectors`, decode_full scoring, seeds 19521006/07/08/09/10. `m3_full` and `no_motion` are the **existing** amb160 main-run results (re-used, not re-trained); `no_time`/`no_state`/`no_rel`/`no_src` are new runs with identical settings, differing only in which single term is dropped from `cta.enabled_terms`.

**n=5 → low statistical power. The paired t-test (df=4) is reported for completeness; interpretation leans on per-seed sign consistency (SC) and dev/test agreement, not p alone.**

Convention: Δ = variant − m3_full. **Negative ΔF1 ⇒ removing the term HURT ⇒ the term helps on hard data (reversal vs the easy-data finding).**

## 1. Full metric panel — dev (mean±std, n=5)

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.7434±0.0163 | 0.4503±0.0080 | 0.5601±0.0100 | 0.2155±0.0130 | 0.8893±0.0304 | 0.2922±0.0093 | 0.7078±0.0093 | 0.6772±0.0513 | 0.5212±0.0156 |
| no_motion | 0.7471±0.0223 | 0.4415±0.0200 | 0.5539±0.0206 | 0.2207±0.0138 | 0.8942±0.0342 | 0.2977±0.0099 | 0.7023±0.0099 | 0.6727±0.1011 | 0.5157±0.0149 |
| no_time | 0.7556±0.0088 | 0.4533±0.0206 | 0.5654±0.0166 | 0.2157±0.0182 | 0.8817±0.0276 | 0.2903±0.0148 | 0.7097±0.0148 | 0.7030±0.1289 | 0.5225±0.0251 |
| no_state | 0.7739±0.0244 | 0.4498±0.0212 | 0.5675±0.0157 | 0.2034±0.0208 | 0.8675±0.0287 | 0.2855±0.0099 | 0.7145±0.0099 | 0.6312±0.1358 | 0.5348±0.0241 |
| no_rel | 0.7601±0.0098 | 0.4470±0.0284 | 0.5615±0.0213 | 0.2175±0.0112 | 0.8753±0.0103 | 0.2904±0.0114 | 0.7096±0.0114 | 0.6638±0.1531 | 0.5321±0.0327 |
| no_src | 0.7581±0.0136 | 0.4429±0.0129 | 0.5583±0.0087 | 0.2055±0.0127 | 0.8819±0.0187 | 0.2827±0.0090 | 0.7173±0.0090 | 0.6371±0.0995 | 0.5331±0.0160 |

## 2. Full metric panel — test (mean±std, n=5)

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.7558±0.0161 | 0.4532±0.0126 | 0.5661±0.0132 | 0.2195±0.0088 | 0.8772±0.0259 | 0.2943±0.0069 | 0.7057±0.0069 | 0.7041±0.0576 | 0.5020±0.0077 |
| no_motion | 0.7482±0.0133 | 0.4305±0.0145 | 0.5457±0.0113 | 0.2219±0.0081 | 0.9010±0.0263 | 0.2935±0.0048 | 0.7065±0.0048 | 0.6653±0.1025 | 0.5100±0.0153 |
| no_time | 0.7589±0.0129 | 0.4524±0.0263 | 0.5660±0.0230 | 0.2108±0.0154 | 0.8851±0.0224 | 0.2936±0.0117 | 0.7064±0.0117 | 0.7217±0.1406 | 0.5050±0.0146 |
| no_state | 0.7583±0.0224 | 0.4389±0.0311 | 0.5549±0.0276 | 0.2169±0.0291 | 0.8727±0.0543 | 0.2953±0.0135 | 0.7047±0.0135 | 0.6430±0.1506 | 0.5156±0.0191 |
| no_rel | 0.7638±0.0220 | 0.4492±0.0251 | 0.5647±0.0230 | 0.2141±0.0161 | 0.8767±0.0341 | 0.2949±0.0128 | 0.7051±0.0128 | 0.6823±0.1589 | 0.5103±0.0261 |
| no_src | 0.7544±0.0105 | 0.4385±0.0204 | 0.5538±0.0188 | 0.2115±0.0138 | 0.8769±0.0366 | 0.2873±0.0068 | 0.7127±0.0068 | 0.6356±0.0976 | 0.5164±0.0144 |

## 3. (a)+(b) ΔF1 vs m3_full — direction, sign consistency, paired t-test

### dev

| variant | per-seed ΔF1 (06/07/08/09/10) | mean ΔF1 | direction | SC | t(df=4) | p | reading |
|---|---|---|---|---|---|---|---|
| no_motion | -0.0198, -0.0363, +0.0261, -0.0147, +0.0135 | -0.0062 | removal hurts | no | -0.55 | 0.613 | term HELPS (removal hurts) → reversal |
| no_time | +0.0059, +0.0055, +0.0091, +0.0267, -0.0205 | +0.0053 | removal helps/ties | no | 0.71 | 0.518 | term does NOT help (removal ≥ full) |
| no_state | -0.0149, +0.0075, +0.0118, +0.0357, -0.0029 | +0.0074 | removal helps/ties | no | 0.88 | 0.427 | term does NOT help (removal ≥ full) |
| no_rel | -0.0056, -0.0150, +0.0280, +0.0300, -0.0302 | +0.0014 | removal helps/ties | no | 0.12 | 0.910 | term does NOT help (removal ≥ full) |
| no_src | -0.0106, -0.0124, +0.0100, +0.0173, -0.0134 | -0.0018 | removal hurts | no | -0.28 | 0.792 | term HELPS (removal hurts) → reversal |

### test

| variant | per-seed ΔF1 (06/07/08/09/10) | mean ΔF1 | direction | SC | t(df=4) | p | reading |
|---|---|---|---|---|---|---|---|
| no_motion | -0.0270, -0.0352, -0.0139, -0.0082, -0.0174 | -0.0203 | removal hurts | YES | -4.24 | 0.013 | term HELPS (removal hurts) → reversal |
| no_time | +0.0193, -0.0038, -0.0026, +0.0340, -0.0474 | -0.0001 | removal hurts | no | -0.01 | 0.995 | term HELPS (removal hurts) → reversal |
| no_state | -0.0287, +0.0111, -0.0145, +0.0202, -0.0437 | -0.0111 | removal hurts | no | -0.93 | 0.404 | term HELPS (removal hurts) → reversal |
| no_rel | +0.0215, +0.0044, +0.0113, +0.0091, -0.0532 | -0.0014 | removal hurts | no | -0.10 | 0.922 | term HELPS (removal hurts) → reversal |
| no_src | -0.0199, -0.0138, -0.0178, +0.0240, -0.0340 | -0.0123 | removal hurts | no | -1.27 | 0.273 | term HELPS (removal hurts) → reversal |

## 4. (c) Is motion special, or do other terms reverse too?

| variant | dev ΔF1 | dev SC | test ΔF1 | test SC | reversal (both splits, ΔF1<0)? |
|---|---|---|---|---|---|
| no_motion | -0.0062 | no | -0.0203 | YES | YES — both splits, SC partial |
| no_time | +0.0053 | no | -0.0001 | no | no |
| no_state | +0.0074 | no | -0.0111 | no | no |
| no_rel | +0.0014 | no | -0.0014 | no | no |
| no_src | -0.0018 | no | -0.0123 | no | YES — both splits, SC partial |

- Terms whose removal lowers F1 on BOTH dev and test (= term helps on hard data): **no_motion, no_src**.
- Motion-only? **NO** (2/5 terms reverse).

## 5. (d) Does each term's targeted diagnostic degrade specifically?

| variant | targeted metric | expected on removal | dev Δ | test Δ | dev SC | test SC | matches expectation? |
|---|---|---|---|---|---|---|---|
| no_motion | recall | down | -0.0088 | -0.0227 | no | YES | YES (both splits) |
| no_time | wrong_merge_rate | up | +0.0002 | -0.0087 | no | no | partial (one split) |
| no_time | impossible_transition_rate | up | -0.0019 | -0.0006 | no | no | NO |
| no_state | wrong_merge_rate | up | -0.0121 | -0.0026 | no | no | NO |
| no_rel | recall | down | -0.0033 | -0.0041 | no | no | YES (both splits) |
| no_rel | fragmentation_rate | up | -0.0140 | -0.0005 | no | no | NO |
| no_src | precision | ? | +0.0148 | -0.0013 | YES | no | n/a (no directional prediction) |

- **per-source-pair precision is NOT present in `metrics.json`** (available keys: precision, recall, f1, wrong_merge_rate, fragmentation_rate, impossible_transition_rate, trajectory_consistency_rate, dangling_precision, dangling_recall). The `no_src` row therefore uses overall precision only; a source-pair-resolved precision check could not be performed with the stored metrics.

## 6. Caveats

- n=5 per variant → paired t-test df=4 has low power; a non-significant p does NOT establish absence of an effect, and a significant p at n=5 is fragile.
- 5 removal variants are compared against one baseline; **no multiple-comparison correction is applied** — with 5 tests at α=0.05 the family-wise false-positive risk is ≈23%.
- Selection was made on dev; test values are reported as the final outcome. All four new variants that were trained are reported here regardless of outcome; no variant was dropped.
- Absolute performance on amb160 is low for every model (test F1 ≈ 0.57 for m3_full); these are RELATIVE contrasts on hard data.