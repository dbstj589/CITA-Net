# 백마고지(Hill 395) STKG 실험 결과

- 데이터셋: `data/battlefield_hill395_large/` (1,005,101 트리플, 75 섹터, 51,310 관측)
- 모델/설정: Full CITA-Net, `configs/cita_full_hill395.yaml`, 50 train 섹터 × 40 epoch
- **test F1 0.686 / precision 0.895 / recall 0.561**

## 파일 목록
| 파일 | 내용 |
|---|---|
| `metrics_summary.md` / `.json` | dev/test 집계 지표 + 기존 baseline 비교 |
| `per_sector_metrics.csv` | 섹터별 지표(dev+test 25개 섹터) |
| `training_curve.csv` | epoch별 loss / best dev-F1 |
| `dataset_stats.json` | 트리플·관측 수, 타입/상태/소스 분포 |
| `qualitative_cases.md` / `.json` | 정성 사례(병합/오병합/단편화/dangling/불가능전이) |
| `type_confusion.csv` | 오병합이 잦은 타입쌍 |
| `samples/output_*.json` | 디코드된 Part-B 섹터 전문(예시 3개) |

## 정성 사례 수집 건수
- 올바른 병합 25 / 잘못된 병합 25 / 단편화 20
- dangling 정탐 15 / 미탐 15 / 불가능 전이 20
## 그래프 (figures/)
- `training_curve.png` — epoch별 loss & best dev F1
- `dev_test_metrics.png` — dev/test 9개 지표 막대그래프
- `baseline_compare.png` — 기존 large suite 대비 P/R/F1
- `per_sector_metrics.png` — 섹터별 P/R/F1
- `dataset_distributions.png` — 타입/상태/소스 관측 분포
- `type_confusion.png` — 오병합 잦은 타입쌍
