# realistic-v1 — conditional validity of all five constraint terms, n=1

Data `data/realistic_v1` (161 true identities/sector, 3.99M triples), decoder `num_slots=200`, blocking `dt_max_s=450 / r_err_floor_m=300`, epochs=40, `--full-sectors`, decode_full scoring. Seeds: 19521006.

> **PARTIAL — n=1 of the planned 5 seeds.** Directional reading only.

**n=1 → low statistical power. The paired t-test (df=0) is reported for completeness; interpretation leans on per-seed sign consistency (SC) and dev/test agreement, not p alone. Holm-adjusted p is a reference column; the pre-registered verdict uses the uncorrected p together with SC.**

Convention: Δ = variant − m3_full. **Negative ΔF1 ⇒ removing the term HURT ⇒ the term helps.** Where dev and test disagree in sign, the result is marked NOT TRUSTWORTHY.

Analyses reported: **표준 디코드 (사전 등록 주 분석)**  
> 2차(revive) 분석은 `scripts/revive_realistic_v1.py` 미실행으로 아직 없음. `wrong_merge_rate`의 분모도 그 패스에서 복구된다.

## 1. [표준 디코드 (사전 등록 주 분석)] 전체 지표 패널 (mean, 분모 병기)

### dev

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.7055 | 0.1746 | 0.2792 | 0.1815 | 0.9774 | 0.2694 | 0.7306 | 0.2602 | 0.7849 |
| no_motion | 0.7424 | 0.2104 | 0.3270 | 0.1835 | 0.9855 | 0.2568 | 0.7432 | 0.2806 | 0.7308 |
| no_time | 0.7138 | 0.1615 | 0.2612 | 0.1973 | 0.9693 | 0.2601 | 0.7399 | 0.2513 | 0.7937 |
| no_state | 0.6514 | 0.1338 | 0.2202 | 0.2160 | 1.0262 | 0.2800 | 0.7200 | 0.2446 | 0.7900 |
| no_rel | 0.6805 | 0.1864 | 0.2912 | 0.2308 | 1.0014 | 0.2941 | 0.7059 | 0.2743 | 0.7606 |
| no_src | 0.7209 | 0.1976 | 0.3087 | 0.2134 | 0.9875 | 0.2615 | 0.7385 | 0.2664 | 0.7625 |
| *분모(평균)* | 26.8 | 109.9 | — | n/a (revive 미실행) | 109.9 | 1091.8 | 1091.8 | 154.7 | 51.1 |   ← m3_full 기준

### test

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.6994 | 0.1751 | 0.2782 | 0.2038 | 0.9775 | 0.2761 | 0.7239 | 0.2541 | 0.7422 |
| no_motion | 0.7263 | 0.2099 | 0.3231 | 0.1780 | 0.9734 | 0.2473 | 0.7527 | 0.2805 | 0.7182 |
| no_time | 0.7312 | 0.1618 | 0.2629 | 0.1846 | 0.9558 | 0.2513 | 0.7487 | 0.2485 | 0.7852 |
| no_state | 0.6566 | 0.1295 | 0.2149 | 0.2185 | 1.0072 | 0.2825 | 0.7175 | 0.2374 | 0.7799 |
| no_rel | 0.7122 | 0.1920 | 0.2998 | 0.1984 | 0.9701 | 0.2708 | 0.7292 | 0.2639 | 0.7319 |
| no_src | 0.7072 | 0.1786 | 0.2831 | 0.1929 | 0.9828 | 0.2545 | 0.7455 | 0.2489 | 0.7238 |
| *분모(평균)* | 27.7 | 108.8 | — | n/a (revive 미실행) | 108.8 | 1095.9 | 1095.9 | 153.5 | 52.2 |   ← m3_full 기준

## 2. (a) ΔF1 vs m3_full — 방향·부호일관성·paired t·Holm

### [표준 디코드 (사전 등록 주 분석)]

**dev**

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=0) | p | p(Holm) | reading |
|---|---|---|---|---|---|---|---|---|
| no_motion | +0.0478 | +0.0478 | removal helps/ties | n/a (n=1) | n/a | n/a | n/a | term does NOT help |
| no_time | -0.0180 | -0.0180 | removal hurts | n/a (n=1) | n/a | n/a | n/a | term HELPS (removal hurts) |
| no_state | -0.0590 | -0.0590 | removal hurts | n/a (n=1) | n/a | n/a | n/a | term HELPS (removal hurts) |
| no_rel | +0.0120 | +0.0120 | removal helps/ties | n/a (n=1) | n/a | n/a | n/a | term does NOT help |
| no_src | +0.0295 | +0.0295 | removal helps/ties | n/a (n=1) | n/a | n/a | n/a | term does NOT help |

**test**

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=0) | p | p(Holm) | reading |
|---|---|---|---|---|---|---|---|---|
| no_motion | +0.0449 | +0.0449 | removal helps/ties | n/a (n=1) | n/a | n/a | n/a | term does NOT help |
| no_time | -0.0153 | -0.0153 | removal hurts | n/a (n=1) | n/a | n/a | n/a | term HELPS (removal hurts) |
| no_state | -0.0633 | -0.0633 | removal hurts | n/a (n=1) | n/a | n/a | n/a | term HELPS (removal hurts) |
| no_rel | +0.0216 | +0.0216 | removal helps/ties | n/a (n=1) | n/a | n/a | n/a | term does NOT help |
| no_src | +0.0049 | +0.0049 | removal helps/ties | n/a (n=1) | n/a | n/a | n/a | term does NOT help |

## 3. (b) amb160 대조 — realistic-v1에서 새로 유효해진 항이 있는가

amb160은 불확실성이 깨끗한 벤치마크였고 운동학만 역전했다. 사전 등록 가설이 맞다면 새로 주입한 4종 불확실성(시각 오프셋·보고지연, 상태 동역학, 편제 관계 지문, 출처 오차 이질성)을 겨냥한 항들이 여기서 음수로 돌아서야 한다.

### [표준 디코드 (사전 등록 주 분석)]

| variant | realistic-v1 dev ΔF1 | realistic-v1 test ΔF1 | amb160 test ΔF1 | amb160 SC | 판정 |
|---|---|---|---|---|---|
| no_motion | +0.0478 | +0.0449 | -0.0203 | YES | no |
| no_time | -0.0180 | -0.0153 | -0.0001 | no | YES — 새로 유효 |
| no_state | -0.0590 | -0.0633 | -0.0111 | no | YES — 새로 유효 |
| no_rel | +0.0120 | +0.0216 | -0.0014 | no | no |
| no_src | +0.0295 | +0.0049 | -0.0123 | no | no |

- 새로 유효: **no_time, no_state**
- 신뢰 불가(dev/test 충돌): **none**

## 4. (d) 운동학이 realistic-v1에서도 유효한가

- amb160(대조): test ΔF1 -0.0203, SC YES, p 0.0133 → 운동학이 도움이 됐다.
- [표준 디코드 (사전 등록 주 분석)] dev ΔF1 +0.0478 (SC n/a (n=1)), test ΔF1 +0.0449 (SC n/a (n=1), t n/a, p n/a) → **no**

## 5. recall·precision 채널 — ΔF1이 어느 경로에서 나오는가

특히 `no_motion`의 ΔF1이 재현율 경로에서 나오는지 확인한다.

### [표준 디코드 (사전 등록 주 분석)]

| variant | split | ΔF1 | ΔRecall | ΔPrecision | 주 경로 |
|---|---|---|---|---|---|
| no_motion | dev | +0.0478 | +0.0359 | +0.0369 | both |
| no_motion | test | +0.0449 | +0.0348 | +0.0268 | both |
| no_time | dev | -0.0180 | -0.0130 | +0.0084 | both |
| no_time | test | -0.0153 | -0.0132 | +0.0318 | precision |
| no_state | dev | -0.0590 | -0.0408 | -0.0541 | both |
| no_state | test | -0.0633 | -0.0456 | -0.0429 | both |
| no_rel | dev | +0.0120 | +0.0118 | -0.0250 | precision |
| no_rel | test | +0.0216 | +0.0170 | +0.0128 | both |
| no_src | dev | +0.0295 | +0.0230 | +0.0154 | both |
| no_src | test | +0.0049 | +0.0036 | +0.0078 | precision |

## 6. (c) 각 항의 겨냥 지표가 특이적으로 나빠지는가 (분모 병기)

### [표준 디코드 (사전 등록 주 분석)]

| variant | 겨냥 지표 | 제거 시 기대 | dev Δ | test Δ | dev SC | test SC | 분모(변형/기준, test) | 일치? |
|---|---|---|---|---|---|---|---|---|
| no_motion | recall | down | +0.0359 | +0.0348 | n/a (n=1) | n/a (n=1) | 108.8 / 108.8 | NO |
| no_time | wrong_merge_rate | up | +0.0158 | -0.0192 | n/a (n=1) | n/a (n=1) | n/a (revive 미실행) / n/a (revive 미실행) | partial (한쪽) |
| no_time | impossible_transition_rate | up | -0.0093 | -0.0248 | n/a (n=1) | n/a (n=1) | 966.5 / 1095.9 | NO |
| no_state | wrong_merge_rate | up | +0.0345 | +0.0147 | n/a (n=1) | n/a (n=1) | n/a (revive 미실행) / n/a (revive 미실행) | YES (양쪽) |
| no_rel | recall | down | +0.0118 | +0.0170 | n/a (n=1) | n/a (n=1) | 108.8 / 108.8 | NO |
| no_rel | fragmentation_rate | up | +0.0239 | -0.0074 | n/a (n=1) | n/a (n=1) | 108.8 / 108.8 | partial (한쪽) |
| no_src | precision | ? | +0.0154 | +0.0078 | n/a (n=1) | n/a (n=1) | 27.3 / 27.7 | n/a (방향 예측 없음) |

- **출처쌍별(per-source-pair) 정밀도는 `metrics.json`에 없다** (보유 키: precision, recall, f1, wrong_merge_rate, fragmentation_rate, impossible_transition_rate, trajectory_consistency_rate, dangling_precision, dangling_recall). `no_src` 행은 전체 정밀도로만 대리하며, 출처쌍 분해 정밀도 점검은 **수행 불가**다.

## 7. 한계

- n=1 → paired t(df=0)는 저검정력. 비유의 p가 효과 부재를 뜻하지 않고, 이 n에서의 유의 p도 취약하다.
- 5개 변형을 하나의 기준선과 비교하며 **다중비교 미보정**(α=0.05에서 FWER ≈23%). Holm 보정 p를 참고 열로 병기했으나 주 판정은 사전 등록대로 미보정 + 부호일관성.
- dev로 선택하고 test를 보고한다. 학습된 모든 변형을 결과와 무관하게 보고했고 제외한 변형은 없다.
- 절대 성능이 모든 모델에서 낮다 (m3_full test F1 ≈ 0.278, recall ≈ 0.175). 디코더가 대부분의 관측을 ∅ 슬롯으로 보내므로 항 효과가 **저재현율 영역**에서 측정된다. 2차(revive) 분석이 이 영역 의존성을 점검한다.
- **`fragmentation_rate`는 비율이 아니다.** 분자가 `fragmented + unrecovered`라 1.0을 넘을 수 있다(실측 최대 1.13). 0~1로 해석하면 안 된다.
- `aggregate`는 섹터별 **비율의 매크로 평균**(카운트 합산 후 재계산이 아님)이라 분모가 작은 섹터도 동일 가중을 받는다. 그래서 분모 평균을 함께 싣는다.
- `wrong_merge_rate`의 분모는 `metrics.json`에 저장되지 않는다(eval.py:136의 `len(result.identities)`). revive 패스의 margin-0 재디코드에서 복구했으며, margin-0은 표준 디코드와 수학적으로 동일하므로 근사가 아니라 정확값이다 — **아직 미실행이라 이 표에서는 n/a**.
- 40에폭은 amb160 대조와 맞춘 고정 예산이며 수렴점이 아니다(dev F1이 40에폭에서도 상승 중). 변형들은 수렴이 아니라 **동일 예산**에서 비교된다. 80에폭 탐색 런이 이 예산 의존성을 별도로 점검한다(본 집계 제외).
- 모든 수치는 통제된 합성 suite에 대한 것이며 실제 전장 데이터 일반화 성능이 아니다.