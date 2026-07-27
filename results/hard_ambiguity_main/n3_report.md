# amb160 main run — n=3 (seeds 19521006/07/08)

Data `data/hard_ambiguity/amb160` (~109 identities/sector). Decoder `num_slots=160`. epochs=40, full-sectors, decode_full scoring. Identical settings across all 3 seeds.
**n=3 → low statistical power; paired t-test df=2 is reported but interpretation leans on sign consistency (SC) across seeds and dev/test agreement, not p alone.**

## (a) F1  (mean±std)

| model | dev F1 | test F1 |
|---|---|---|
| m3_full | 0.5642±0.0107 | 0.5650±0.0141 |
| no_motion | 0.5542±0.0222 | 0.5397±0.0078 |
| term_gating | 0.5571±0.0476 | 0.5576±0.0267 |

### m3_full vs no_motion — reversal: does motion help under ambiguity?

| split | per-seed ΔF1 (06/07/08) | mean Δ | sign-consistent | t(df=2) | p(2-tail) |
|---|---|---|---|---|---|
| dev | +0.0198, +0.0363, -0.0261 | +0.0100 | no (+) | 0.54 | 0.646 |
| test | +0.0270, +0.0352, +0.0139 | +0.0254 | YES (+) | 4.10 | 0.055 |

### term_gating vs m3_full — does learned gating beat fixed-weight motion?

| split | per-seed ΔF1 (06/07/08) | mean Δ | sign-consistent | t(df=2) | p(2-tail) |
|---|---|---|---|---|---|
| dev | -0.0549, -0.0174, +0.0511 | -0.0071 | no (-) | -0.23 | 0.841 |
| test | -0.0299, -0.0167, +0.0244 | -0.0074 | no (-) | -0.45 | 0.695 |

## (b) recall & trade-offs (test, mean±std)

| model | recall | precision | wrong_merge | impossible | frag |
|---|---|---|---|---|---|
| m3_full | 0.4532±0.0156 | 0.7520±0.0089 | 0.2175±0.0082 | 0.2915±0.0079 | 0.8797±0.0326 |
| no_motion | 0.4227±0.0123 | 0.7489±0.0163 | 0.2200±0.0070 | 0.2960±0.0045 | 0.8987±0.0211 |
| term_gating | 0.4427±0.0339 | 0.7563±0.0040 | 0.2085±0.0206 | 0.2890±0.0191 | 0.8656±0.0120 |

- m3_full−no_motion recall Δ (test) per seed: +0.0352, +0.0425, +0.0136  (mean +0.0304, SC=YES)

## (c) term_gating gate distribution — motion gate, easy vs ambiguous (per seed)

| seed | motion easy (sim>0.5) | motion ambig (sim≤0.5) | Δ | sem easy | sem ambig |
|---|---|---|---|---|---|
| 19521006 | 0.650 | 0.966 | +0.315 | 0.248 | 0.010 |
| 19521007 | 0.663 | 0.982 | +0.319 | 0.243 | 0.006 |
| 19521008 | 0.674 | 0.973 | +0.298 | 0.323 | 0.014 |

- motion gate rises easy→ambiguous in all 3 seeds (mean 0.663→0.974); ambiguous-pair ratio (raw sim≤0.5) ≈ 0.508 (data-fixed, from seed-1).

## interpretation

- Reversal (m3_full>no_motion): dev SC=no, test SC=YES → PARTIAL — see per-seed signs.
- term_gating vs m3_full (test): mean Δ -0.0074, SC=no → term_gating does NOT beat m3_full (reproduced).
- **Absolute performance stays low**: test F1 ≈ 0.57, fragmentation ≈ 0.88 — amb160 is hard (4× hard-neg density vs frozen); report relative advantage separately from absolute quality.