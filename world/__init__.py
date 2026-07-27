"""Omniscient ground-truth STKG world generator (Hill 395).

Implements pipeline stages P1 (world state), P2 (terrain/visibility layers),
P3 (serialisation), P4 (integrity validation) and P8 (reproducibility) of the
"데이터셋 구축 방식" spec. Sensor projection (P5), error injection (P6), the
alignment/defect ledger (P7) and merge experiments (P9) are out of scope here.

The GT is sensor-neutral: it contains only the *true* world, so any later sensor
combination (drone-drone, wolf-robot pairs, or mixed) can be projected from it.
No observation-only fields (source, cep_m, *_confidence, clock offset, noise)
appear anywhere in the GT.
"""
