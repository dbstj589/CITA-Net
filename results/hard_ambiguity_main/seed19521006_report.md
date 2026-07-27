# amb160 main run — Stage 1 (seed 19521006 only)

Data: `data/hard_ambiguity/amb160` (~109 identities/sector, m1 gate F1 0.903 vs frozen 0.970).
Decoder: `num_slots=160` (fixes the 48-slot starvation that collapsed F1 to ~0.30).
epochs=40, full-sectors, decode_full scoring, seed 19521006. **1 seed → direction only, no statistics.**

## (a) F1 — does the constraint (motion) model beat no_motion?

| model | dev F1 | test F1 |
|---|---|---|
| m3_full | 0.5643 | 0.5581 |
| no_motion | 0.5445 | 0.5310 |
| term_gating | 0.5094 | 0.5282 |

**Reversal direction (test F1):**
- frozen (easy): no_motion 0.7034 > m3_full 0.6520  → motion HURTS (+5.1pt for no_motion)
- amb160 (hard): m3_full 0.5581 > no_motion 0.5310  → motion HELPS (+2.7pt for m3_full)
- **Direction reverses as hypothesised.** term_gating 0.5282: does NOT beat m3_full; ~ties no_motion (test), below no_motion (dev).

## (b) recall & trade-offs (test)

| model | recall | precision | wrong_merge | impossible_trans | frag |
|---|---|---|---|---|---|
| m3_full | 0.4439 | 0.7521 | 0.2206 | 0.2889 | 0.8816 |
| no_motion | 0.4086 | 0.7609 | 0.2156 | 0.2939 | 0.8831 |
| term_gating | 0.4058 | 0.7574 | 0.1864 | 0.2711 | 0.8741 |

- m3_full: best recall (0.444) → motion recovers ambiguous associations.
- term_gating: lowest recall (0.406) but lowest wrong_merge (0.186) & impossible (0.271) → most conservative.

## (c) term_gating gate distribution (amb160 test, per-term mean gate in (0,1))

ambiguous-pair ratio (raw semantic sim ≤ 0.5) = **0.508** (half of all candidate pairs).

| bucket | N | sem | time | motion | state | rel | src |
|---|---|---|---|---|---|---|---|
| sim>0.5 (easy) | 1.21M | 0.248 | 0.953 | **0.650** | 1.000 | 0.291 | 0.807 |
| sim≤0.5 (ambig) | 1.25M | 0.010 | 0.953 | **0.966** | 1.000 | 0.022 | 0.997 |

→ On ambiguous pairs the gate turns motion UP (0.650→0.966) and semantics DOWN (0.248→0.010): 'when content can't separate them, use kinematics'.

### ambiguity is genuinely higher than before (label-based, encoder-free)

| data | identities/sector | cand pairs/sector | neg:pos | negatives/identity |
|---|---|---|---|---|
| frozen | 28 | 13,361 | 9.4:1 | 437 |
| amb80 | 53 | 45,427 | 17.5:1 | 815 |
| **amb160** | **109** | **188,947** | **36.4:1** | **1,689** |

## caveat
- 1 seed: report DIRECTION only; no mean±std / significance claims.
- decode_full scoring (same as prior ablations); 48-slot results discarded (starvation artifact).
- Absolute F1 depressed vs frozen because amb160 is genuinely harder (4× hard-neg density).