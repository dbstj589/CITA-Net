# 백마고지(Hill 395) 실험 — 정량 결과

학습: `cita_full_hill395`, 전체 50 train 섹터 × 40 epoch, 최종 loss 1.63, best dev-F1 0.693.

## dev / test 집계 지표

| 지표 | dev | test | 기존 large suite(참고) |
|---|---|---|---|
| precision | 0.8711 | 0.8948 | 0.8 |
| recall | 0.5478 | 0.5611 | 0.4 |
| f1 | 0.6694 | 0.6858 | 0.52 |
| wrong_merge_rate | 0.1317 | 0.1214 | - |
| fragmentation_rate | 0.7302 | 0.6850 | - |
| trajectory_consistency_rate | 0.7567 | 0.7538 | - |
| impossible_transition_rate | 0.2433 | 0.2462 | - |
| dangling_precision | 0.6820 | 0.6205 | - |
| dangling_recall | 0.5823 | 0.5712 | - |

*기존 large suite 참고치는 README 인용(F1~0.52 / P~0.80 / R~0.40).*
