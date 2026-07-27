# amb160 main run — n=5 (seeds 19521006/07/08/09/10)

Data `data/hard_ambiguity/amb160` (~109 identities/sector). Decoder `num_slots=160`. epochs=40, full-sectors, decode_full scoring. Identical settings across all 5 seeds (seeds 06/07/08 reused from n=3; 09/10 added).
**n=5 → low statistical power; paired t-test df=4 is reported but interpretation leans on sign consistency (SC) across seeds and dev/test agreement, not p alone.**

## (a) F1  (mean±std)

| model | dev F1 | test F1 |
|---|---|---|
| m3_full | 0.5601±0.0100 | 0.5661±0.0132 |
| no_motion | 0.5539±0.0206 | 0.5457±0.0113 |
| term_gating | 0.5413±0.0680 | 0.5388±0.0569 |

### m3_full vs no_motion — reversal: does motion help under ambiguity?

| split | per-seed ΔF1 (06/07/08/09/10) | mean Δ | sign-consistent | t(df=4) | p(2-tail) |
|---|---|---|---|---|---|
| dev | +0.0198, +0.0363, -0.0261, +0.0147, -0.0135 | +0.0062 | no (+) | 0.55 | 0.613 |
| test | +0.0270, +0.0352, +0.0139, +0.0082, +0.0174 | +0.0203 | YES (+) | 4.24 | 0.013 |

### term_gating vs m3_full — does learned gating beat fixed-weight motion?

| split | per-seed ΔF1 (06/07/08/09/10) | mean Δ | sign-consistent | t(df=4) | p(2-tail) |
|---|---|---|---|---|---|
| dev | -0.0549, -0.0174, +0.0511, +0.0462, -0.1188 | -0.0188 | no (-) | -0.59 | 0.588 |
| test | -0.0299, -0.0167, +0.0244, +0.0216, -0.1357 | -0.0273 | no (-) | -0.94 | 0.402 |

## (b) recall & trade-offs (test, mean±std)

| model | recall | precision | wrong_merge | impossible | frag |
|---|---|---|---|---|---|
| m3_full | 0.4532±0.0126 | 0.7558±0.0161 | 0.2195±0.0088 | 0.2943±0.0069 | 0.8772±0.0259 |
| no_motion | 0.4305±0.0145 | 0.7482±0.0133 | 0.2219±0.0081 | 0.2935±0.0048 | 0.9010±0.0263 |
| term_gating | 0.4211±0.0635 | 0.7586±0.0102 | 0.2119±0.0157 | 0.2939±0.0158 | 0.8766±0.0175 |

- m3_full−no_motion recall Δ (test) per seed: +0.0352, +0.0425, +0.0136, +0.0089, +0.0133  (mean +0.0227, SC=YES)

## (c) term_gating gate distribution — motion gate, easy vs ambiguous (per seed)

| seed | motion easy (sim>0.5) | motion ambig (sim≤0.5) | Δ | sem easy | sem ambig |
|---|---|---|---|---|---|
| 19521006 | 0.650 | 0.966 | +0.315 | 0.248 | 0.010 |
| 19521007 | 0.663 | 0.982 | +0.319 | 0.243 | 0.006 |
| 19521008 | 0.674 | 0.973 | +0.298 | 0.323 | 0.014 |
| 19521009 | 0.612 | 0.947 | +0.335 | 0.312 | 0.011 |
| 19521010 | 0.644 | 0.970 | +0.326 | 0.290 | 0.015 |

- motion gate rises easy→ambiguous in all 3 seeds (mean 0.649→0.968); ambiguous-pair ratio (raw sim≤0.5) ≈ 0.508 (data-fixed, from seed-1).

## interpretation

- Reversal (m3_full>no_motion): dev SC=no, test SC=YES → PARTIAL — see per-seed signs.
- term_gating vs m3_full (test): mean Δ -0.0273, SC=no → term_gating does NOT beat m3_full (reproduced).
- **Absolute performance stays low**: test F1 ≈ 0.57, fragmentation ≈ 0.88 — amb160 is hard (4× hard-neg density vs frozen); report relative advantage separately from absolute quality.